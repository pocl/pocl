/* OpenCL runtime library: in-process JIT linking and loading of kernel object
   files via LLVM ORC + JITLink.

   Copyright (c) 2026 PoCL developers

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#include "config.h"

#include "pocl_debug.h"
#include "pocl_dynlib.h"
#include "pocl_llvm_orc.h"

#include <llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/TargetParser/Triple.h>

/* absoluteSymbols() moved from Core.h to its own header around LLVM 20. */
#if __has_include(<llvm/ExecutionEngine/Orc/AbsoluteSymbols.h>)
#include <llvm/ExecutionEngine/Orc/AbsoluteSymbols.h>
#endif

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>

using namespace llvm;
using namespace llvm::orc;

/* Latched (under JITMutex) when LLJIT creation fails, after which
   pocl_jit_initialize() fails fast and pocl_cpu_device_uses_jit() reports the
   device as not using the JIT, routing kernels through the linker path
   instead. Never reset: a creation failure (unsupported triple, executor
   setup) is not transient. Read without the lock by the gate; the flag only
   ever flips 0 -> 1, so a stale read at worst routes one more kernel into
   pocl_jit_initialize()'s locked fail-fast path. */
extern "C" int pocl_jit_unavailable = 0;

#ifdef ENABLE_PRINTF_IMMEDIATE_FLUSH
/* Host-side printf flush callback referenced by kernels built with immediate
   flush; defined in libpocl (lib/CL/devices/printf_buffer.c). */
extern "C" void pocl_flush_printf_buffer(char *buffer, uint32_t buffer_size);
#endif

#ifdef __MINGW32__
/* MinGW kernel objects reference the libgcc/compiler-rt stack probe
   ___chkstk_ms (emitted for frames > 4KB), which no Windows system DLL
   exports, unlike MSVC's __chkstk, which ntdll/kernel32 do export. libpocl
   is itself linked against a libgcc/compiler-rt that provides ___chkstk_ms, so
   defineHostSymbols() hands that copy to JIT'd kernels as an absolute symbol.
   (The mem* helpers kernels also call resolve from msvcrt via the JIT's
   default process-symbol search.) Both are reachable because the COFF JIT
   emits kernels with the large code model; see GetTargetMachine() in
   pocl_llvm_wg.cc. */
extern "C" void ___chkstk_ms(void);
#endif

namespace {

/* The process-global JIT that loads all kernel objects, created lazily on
   the first pocl_jit_initialize() call. ORC's ExecutionSession is internally
   synchronized, so a single instance is shared across all CPU host devices.
   Access is additionally serialized by JITMutex so initialization races and
   the (already serialized) PoCL dlhandle cache stay consistent.

   Deliberately a raw pointer that is never deleted: a static destructor
   (at exit() or libpocl dlclose) would end the ExecutionSession and unmap all
   JIT'd kernel code, while the pthread/TBB scheduler threads are only joined
   under POCL_ENABLE_UNINIT (default off) and so may still be executing that
   code -- a shutdown SIGSEGV the linker path cannot hit, since it never
   dlcloses the dlhandle cache's kernel .so handles either (not even under
   POCL_ENABLE_UNINIT: the cache, including its work-group function pointers,
   survives device uninit/reinit). Leaking the JIT gives the exact same
   code-stays-mapped-through-exit guarantee. */
LLJIT *TheJIT = nullptr;
std::mutex JITMutex;

/* JITDylib names must be unique within an ExecutionSession; a monotonic
   counter guarantees that regardless of the caller-supplied name. Only ever
   touched under JITMutex, so a plain integer suffices. */
uint64_t JDCounter = 0;

/* Make a dynamic library's symbols (e.g. libmvec's _ZGVdN8v_expf, or
   libpocl's own dependency chain) resolvable by JIT'd kernels. Attaches a
   DynamicLibrarySearchGenerator for the library to the process-symbols
   JITDylib, which every kernel JITDylib links against (LLJIT links each
   JITDylib it creates against the process-symbols JITDylib by default).
   Lookups are then served from the library's own handle through the JIT's
   generator chain; on POSIX that dlsym scope includes the library's
   dependencies. Returns false (and leaves the JIT usable) if the library
   cannot be loaded. */
static bool addLibrarySearchGenerator(const char *Library) {
  if (!Library || !Library[0])
    return false;
  JITDylibSP PSJD = TheJIT->getProcessSymbolsJITDylib();
  if (!PSJD)
    return false;
  Expected<std::unique_ptr<DynamicLibrarySearchGenerator>> Gen =
      DynamicLibrarySearchGenerator::Load(
          Library, TheJIT->getDataLayout().getGlobalPrefix());
  if (!Gen) {
    consumeError(Gen.takeError());
    return false;
  }
  PSJD->addGenerator(std::move(*Gen));
  return true;
}

/* Make a static archive's members resolvable by JIT'd kernels. JITLink pulls in
   archive members lazily, materializing only those that satisfy a referenced
   symbol (the same semantics as a static link). Used for SVML, whose __svml_*
   symbols ship only in Intel's static libsvml.a (with helper symbols in
   libirc.a). Attaches a StaticLibraryDefinitionGenerator for the archive to the
   process-symbols JITDylib that every kernel JITDylib links against. Returns
   false (and leaves the JIT usable) if the archive cannot be loaded. */
static bool loadStaticArchive(const char *Path) {
  if (!Path || !Path[0])
    return false;
  JITDylibSP PSJD = TheJIT->getProcessSymbolsJITDylib();
  if (!PSJD)
    return false;
  Expected<std::unique_ptr<StaticLibraryDefinitionGenerator>> Gen =
      StaticLibraryDefinitionGenerator::Load(TheJIT->getObjLinkingLayer(),
                                             Path);
  if (!Gen) {
    consumeError(Gen.takeError());
    return false;
  }
  PSJD->addGenerator(std::move(*Gen));
  return true;
}

/* Bookkeeping for a single loaded kernel object. */
struct PoclJITModule {
  JITDylib *JD;
};

/* Inject symbols that kernel objects reference but that are linked into libpocl
   rather than exported by a process library: PoCL's own host callbacks and, on
   Windows, the libgcc/compiler-rt stack probe. Defining them as absolute
   symbols makes resolution independent of libpocl's (deliberately hidden)
   dynamic-symbol visibility. They are constant for the process, so this is done
   once at init into the process-symbols JITDylib that every kernel JITDylib
   links against. Everything else (libc/libm/compiler-rt and msvcrt's mem*
   helpers) resolves via the JIT's default process-symbol search, backed up on
   POSIX by a search of libpocl's own handle (see pocl_jit_initialize). */
void defineHostSymbols(JITDylib &JD) {
  SymbolMap Syms;
#ifdef ENABLE_PRINTF_IMMEDIATE_FLUSH
  Syms[TheJIT->mangleAndIntern("pocl_flush_printf_buffer")] = {
      ExecutorAddr::fromPtr(&pocl_flush_printf_buffer),
      JITSymbolFlags::Exported | JITSymbolFlags::Callable};
#endif
#ifdef __MINGW32__
  Syms[TheJIT->mangleAndIntern("___chkstk_ms")] = {
      ExecutorAddr::fromPtr(&___chkstk_ms),
      JITSymbolFlags::Exported | JITSymbolFlags::Callable};
#endif
  if (Syms.empty())
    return;
  if (Error Err = JD.define(absoluteSymbols(std::move(Syms))))
    POCL_MSG_ERR("pocl_jit: failed to define host symbols: %s\n",
                 toString(std::move(Err)).c_str());
}

} // namespace

int pocl_jit_initialize(const char *TripleStr, const char *CPU) {
  std::lock_guard<std::mutex> Lock(JITMutex);
  if (TheJIT)
    return 0;
  if (pocl_jit_unavailable)
    return -1;

  /* The native target and asm printer are already initialized by
     InitializeLLVM() during device setup; kernel codegen (which uses the same
     target) runs before any kernel object is loaded. */
  LLJITBuilder Builder;

  /* Force the JITLink-based object-linking layer (LLJIT defaults to
     RuntimeDyld). JITLink resolves relocations, maps code into executable
     memory, and registers EH frames entirely in-process, so no external
     linker is needed. */
  /* The object-linking-layer creator callback signature has changed across
     LLVM releases:
        LLVM <= 20 : (ExecutionSession&, const Triple&)
        LLVM 21, 22: (ExecutionSession&)
        LLVM >= 23 : (ExecutionSession&, jitlink::JITLinkMemoryManager&)
     Verified directly against LLVM 18, 20, 21 and 22 (22.1's
     ObjectLinkingLayerCreator is the single-arg form); the >=23 form matches an
     LLVM 23 (trunk) checkout, so the 23 boundary is approximate. In every case
     we construct an ObjectLinkingLayer, which uses the ExecutorProcessControl's
     in-process memory manager when not given one explicitly. */
  Builder.setObjectLinkingLayerCreator(
#if LLVM_MAJOR >= 23
      [](ExecutionSession &ES, jitlink::JITLinkMemoryManager &MM)
          -> Expected<std::unique_ptr<ObjectLayer>> {
        return std::make_unique<ObjectLinkingLayer>(ES, MM);
      }
#elif LLVM_MAJOR >= 21
      [](ExecutionSession &ES) -> Expected<std::unique_ptr<ObjectLayer>> {
        return std::make_unique<ObjectLinkingLayer>(ES);
      }
#else
      [](ExecutionSession &ES,
         const llvm::Triple &) -> Expected<std::unique_ptr<ObjectLayer>> {
        return std::make_unique<ObjectLinkingLayer>(ES);
      }
#endif
  );

  JITTargetMachineBuilder JTMB{llvm::Triple(TripleStr ? TripleStr : "")};
  if (CPU && CPU[0])
    JTMB.setCPU(CPU);
  Builder.setJITTargetMachineBuilder(std::move(JTMB));

  Expected<std::unique_ptr<LLJIT>> JIT = Builder.create();
  /* Test hook: simulate an environment where the JIT cannot be brought up
     (unsupported triple, executor setup failure) to exercise the linker
     fallback; see test_jit_fallback.jl. */
  if (JIT && getenv("POCL_FAULT_INJECT_JIT"))
    JIT = createStringError(inconvertibleErrorCode(),
                            "injected failure (POCL_FAULT_INJECT_JIT)");
  if (!JIT) {
    POCL_MSG_WARN("pocl_jit: LLJIT creation failed: %s;"
                  " falling back to linking kernel binaries\n",
                  toString(JIT.takeError()).c_str());
    pocl_jit_unavailable = 1;
    return -1;
  }
  TheJIT = JIT->release();

  /* Define PoCL's host callbacks (and the Windows stack probe) once, into the
     process-symbols JITDylib that every kernel JITDylib links against. */
  if (JITDylibSP PSJD = TheJIT->getProcessSymbolsJITDylib())
    defineHostSymbols(*PSJD);

#ifdef HAVE_DLFCN_H
  /* The JIT's default process-symbol search resolves through the global dlsym
     scope, which misses libpocl's private dependencies (libgcc_s for compiler
     builtins like __aeabi_uldivmod or __extendhfsf2, libm before glibc 2.34)
     when libpocl was dlopen'd RTLD_LOCAL, as ICD loaders do. The Clang-driver
     link path satisfied those references from each kernel .so's implicit
     -lgcc and own DT_NEEDED entries; restore that visibility by also searching
     libpocl's own dlopen handle, whose dlsym scope includes its dependency
     chain. This pins libpocl in memory, which the process-lifetime JIT
     effectively does anyway. On failure (e.g. libpocl linked statically into
     the executable) kernels fall back to the plain process-global lookup. */
  const char *SelfPath =
      pocl_dynlib_pathname(reinterpret_cast<void *>(&pocl_jit_initialize));
  if (!addLibrarySearchGenerator(SelfPath))
    POCL_MSG_PRINT_LLVM(
        "pocl_jit: could not add libpocl's own libraries ('%s') to the symbol "
        "search; kernels needing libpocl's private dependencies (libgcc_s, "
        "libm) may fail to resolve symbols if those are not process-global\n",
        SelfPath ? SelfPath : "<unknown>");
#endif

  /* CPU codegen vectorizes libm calls (expf, sinf, ...) into a vector-math
     library's symbols (e.g. _ZGVdN8v_expf, __svml_expf8). The library is chosen
     at configure time, matching codegen's veclib selection in pocl_llvm_wg.cc;
     expose that same one to the JIT so the symbols resolve. */
#if defined(ENABLE_HOST_CPU_VECTORIZE_LIBMVEC)
  const char *VecMathLib = "libmvec.so.1";
  if (!addLibrarySearchGenerator(VecMathLib))
    POCL_MSG_PRINT_LLVM(
        "pocl_jit: could not load vector-math library '%s'; vectorized math "
        "kernels may fail to resolve their symbols\n",
        VecMathLib);
#elif defined(ENABLE_HOST_CPU_VECTORIZE_SLEEF)
  const char *VecMathLib = nullptr;
  bool VecMathLoaded = false;
#ifdef HOST_CPU_SLEEF_LIBRARY
  VecMathLib = HOST_CPU_SLEEF_LIBRARY;
  VecMathLoaded = addLibrarySearchGenerator(VecMathLib);
#endif
#ifdef HOST_CPU_SLEEF_LIBRARY_FALLBACK
  if (!VecMathLoaded) {
    VecMathLib = HOST_CPU_SLEEF_LIBRARY_FALLBACK;
    VecMathLoaded = addLibrarySearchGenerator(VecMathLib);
  }
#endif
  if (!VecMathLoaded) {
    VecMathLib = "libsleef.so";
    VecMathLoaded = addLibrarySearchGenerator(VecMathLib);
  }
  if (!VecMathLoaded)
    POCL_MSG_PRINT_LLVM(
        "pocl_jit: could not load vector-math library '%s'; vectorized math "
        "kernels may fail to resolve their symbols\n",
        VecMathLib);
#elif defined(ENABLE_HOST_CPU_VECTORIZE_SVML)
  /* libsvml members reference libirc helpers, so load both archives; either
     generator resolves symbols for the other lazily as members materialize. */
  bool VecMathLoaded = true;
#ifdef HOST_CPU_IRC_LIBRARY
  VecMathLoaded &= loadStaticArchive(HOST_CPU_IRC_LIBRARY);
#endif
#ifdef HOST_CPU_SVML_LIBRARY
  VecMathLoaded &= loadStaticArchive(HOST_CPU_SVML_LIBRARY);
#endif
  if (!VecMathLoaded)
    POCL_MSG_PRINT_LLVM(
        "pocl_jit: could not load the SVML static archives; vectorized math "
        "kernels may fail to resolve their symbols\n");
#endif

  return 0;
}

void *pocl_jit_load_object(const char *Path, const char *UniqName) {
  std::lock_guard<std::mutex> Lock(JITMutex);
  if (!TheJIT) {
    POCL_MSG_ERR("pocl_jit: load_object called before initialize\n");
    return nullptr;
  }

  ErrorOr<std::unique_ptr<MemoryBuffer>> Buf = MemoryBuffer::getFile(
      Path, /*IsText=*/false, /*RequiresNullTerminator=*/false);
  if (!Buf) {
    POCL_MSG_ERR("pocl_jit: cannot read object '%s': %s\n", Path,
                 Buf.getError().message().c_str());
    return nullptr;
  }

  /* Give each loaded object its own JITDylib so symbol names never collide
     across kernels/specializations and can be unloaded independently. */
  std::string Name = std::string(UniqName ? UniqName : "kernel") + "#" +
                     std::to_string(JDCounter++);
  Expected<JITDylib &> JD = TheJIT->createJITDylib(std::move(Name));
  if (!JD) {
    POCL_MSG_ERR("pocl_jit: createJITDylib failed: %s\n",
                 toString(JD.takeError()).c_str());
    return nullptr;
  }

  if (Error Err = TheJIT->addObjectFile(*JD, std::move(*Buf))) {
    POCL_MSG_ERR("pocl_jit: addObjectFile('%s') failed: %s\n", Path,
                 toString(std::move(Err)).c_str());
    if (Error RmErr = TheJIT->getExecutionSession().removeJITDylib(*JD))
      consumeError(std::move(RmErr));
    return nullptr;
  }

  return new PoclJITModule{&*JD};
}

void *pocl_jit_lookup(void *Handle, const char *SymbolName) {
  std::lock_guard<std::mutex> Lock(JITMutex);
  PoclJITModule *M = static_cast<PoclJITModule *>(Handle);
  if (!TheJIT || M == nullptr || M->JD == nullptr)
    return nullptr;

  /* ORC mangles the unmangled name for the target (e.g. adds the leading
     underscore on Mach-O), so we pass the plain C symbol name. Linking and
     relocation of the object happen here, on first lookup. */
  Expected<ExecutorAddr> Addr = TheJIT->lookup(*M->JD, SymbolName);
  if (!Addr) {
    /* Not necessarily fatal: callers may probe alternative names. */
    consumeError(Addr.takeError());
    return nullptr;
  }
  return reinterpret_cast<void *>(Addr->getValue());
}

int pocl_jit_unload(void *Handle) {
  std::lock_guard<std::mutex> Lock(JITMutex);
  PoclJITModule *M = static_cast<PoclJITModule *>(Handle);
  if (M == nullptr)
    return 0;
  if (TheJIT && M->JD) {
    if (Error Err = TheJIT->getExecutionSession().removeJITDylib(*M->JD))
      POCL_MSG_ERR("pocl_jit: removeJITDylib failed: %s\n",
                   toString(std::move(Err)).c_str());
  }
  delete M;
  return 1;
}
