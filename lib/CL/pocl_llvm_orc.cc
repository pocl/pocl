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
#include "pocl_file_util.h"
#include "pocl_llvm.h"
#include "pocl_llvm_orc.h"
#include "pocl_util.h"

#include <llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
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
#include <unordered_map>

using namespace llvm;
using namespace llvm::orc;

#ifdef ENABLE_PRINTF_IMMEDIATE_FLUSH
/* Host-side printf flush callback referenced by kernels built with immediate
   flush; defined in libpocl (lib/CL/devices/printf_buffer.c). */
extern "C" void pocl_flush_printf_buffer(char *buffer, uint32_t buffer_size);
#endif

#ifdef __MINGW32__
/* MinGW kernel objects reference the libgcc/compiler-rt stack probe
   ___chkstk_ms (emitted for frames > 4KB), which no Windows system DLL
   exports. libpocl links against a libgcc/compiler-rt that provides it, so
   defineHostSymbols() hands that copy to kernels as an absolute symbol. The
   COFF JIT's large code model keeps it reachable; see GetTargetMachine() in
   pocl_llvm_wg.cc. */
extern "C" void ___chkstk_ms(void);
#endif

namespace {

/* Process-global JIT shared by all CPU host devices, created on the first
   pocl_jit_initialize() call. JITMutex serializes access. Intentionally never
   deleted: a destructor would unmap the JIT'd code at exit, but scheduler
   threads (joined only under POCL_ENABLE_UNINIT) may still be running it. The
   linker path leaks its kernel .so handles for the same reason. */
LLJIT *TheJIT = nullptr;
std::mutex JITMutex;

/* Set when LLJIT creation fails, after which pocl_jit_initialize() fails fast.
   Never reset, since the failure (unsupported triple, executor setup) is not
   transient. */
bool JITFailed = false;

/* Keeps each JITDylib name unique within the ExecutionSession; touched only
   under JITMutex. */
uint64_t JDCounter = 0;

/* Most recent pocl_jit_lookup() failure text, returned by
   pocl_jit_last_error() (a dlerror() analogue). Guarded by JITMutex. */
std::string LastLookupError;

/* Kernel-library bitcode modules loaded into the JIT. These JITDylibs are kept
   alive for the process lifetime, like TheJIT itself, so later kernels using
   the same CPU variant can reuse already parsed definitions. Guarded by
   JITMutex. */
std::unordered_map<std::string, JITDylib *> BuiltinJITDylibs;

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

/* Make a static archive's members resolvable by JIT'd kernels. JITLink pulls
   members in lazily, materializing only those that satisfy a referenced symbol,
   like a static link. Used for SVML, whose __svml_* symbols ship only in
   Intel's libsvml.a (helpers in libirc.a). Attaches a
   StaticLibraryDefinitionGenerator to the process-symbols JITDylib that every
   kernel JITDylib links against. Returns false if the archive cannot be loaded,
   leaving the JIT usable. */
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

/* Load an archive by absolute path, or by filename from PoCL's private data dir. */
static bool loadPrivateStaticArchive(const char *Path) {
  if (!Path || !Path[0])
    return false;
  if (pocl_exists(Path))
    return loadStaticArchive(Path);

  char PrivateDir[POCL_MAX_PATHNAME_LENGTH];
  pocl_get_private_datadir(PrivateDir);
  std::string FullPath = std::string(PrivateDir) + "/" + Path;
  if (pocl_exists(FullPath.c_str()))
    return loadStaticArchive(FullPath.c_str());

  return loadStaticArchive(Path);
}

/* Make the selected PoCL kernel-library bitcode visible to a kernel JITDylib.
   The final object normally contains already-specialized code, but LLVM native
   codegen can leave late calls to helpers such as FP16 vector builtins. The
   old shared-library final link could satisfy those calls from the kernel
   library; adding the library IR module gives JITLink the same fallback. */
static JITDylib *getOrLoadBuiltinJITDylib(const char *BitcodePath) {
  if (!BitcodePath || !BitcodePath[0])
    return nullptr;

  std::string Path(BitcodePath);
  auto It = BuiltinJITDylibs.find(Path);
  if (It != BuiltinJITDylibs.end())
    return It->second;

  std::string Name = "pocl-kernellib#" + std::to_string(JDCounter++);
  Expected<JITDylib &> JD = TheJIT->createJITDylib(std::move(Name));
  if (!JD) {
    POCL_MSG_ERR("pocl_jit: createJITDylib for kernel library '%s' failed: %s\n",
                 BitcodePath, toString(JD.takeError()).c_str());
    return nullptr;
  }

  auto Ctx = std::make_unique<LLVMContext>();
  SMDiagnostic Diag;
  std::unique_ptr<Module> M = parseIRFile(BitcodePath, Diag, *Ctx);
  if (!M) {
    POCL_MSG_ERR("pocl_jit: cannot parse kernel library '%s': %s\n",
                 BitcodePath, Diag.getMessage().str().c_str());
    if (Error Err = TheJIT->getExecutionSession().removeJITDylib(*JD))
      consumeError(std::move(Err));
    return nullptr;
  }

  M->setDataLayout(TheJIT->getDataLayout());
  if (Error Err = TheJIT->addIRModule(
          *JD, ThreadSafeModule(std::move(M), std::move(Ctx)))) {
    POCL_MSG_ERR("pocl_jit: addIRModule('%s') failed: %s\n", BitcodePath,
                 toString(std::move(Err)).c_str());
    if (Error RmErr = TheJIT->getExecutionSession().removeJITDylib(*JD))
      consumeError(std::move(RmErr));
    return nullptr;
  }

  BuiltinJITDylibs.emplace(std::move(Path), &*JD);
  return &*JD;
}

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
  if (JITFailed)
    return -1;

  /* Called from device init, which runs before clCreateContext's
     InitializeLLVM() call, so register the native target (and the rest of
     the one-time LLVM setup) here. InitializeLLVM() is idempotent. */
  InitializeLLVM();

  LLJITBuilder Builder;

  /* Force the JITLink object-linking layer (LLJIT defaults to RuntimeDyld).
     JITLink resolves relocations, maps code into executable memory, and
     registers EH frames in-process, so no external linker is needed. The
     creator callback signature varies by LLVM release:
        LLVM <= 20 : (ExecutionSession&, const Triple&)
        LLVM 21, 22: (ExecutionSession&)
        LLVM >= 23 : (ExecutionSession&, jitlink::JITLinkMemoryManager&)
     In each case we build an ObjectLinkingLayer, which uses the
     ExecutorProcessControl's in-process memory manager by default. */
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

  /* Normalize the triple (e.g. "x86_64-w64-mingw32" -> "...-windows-gnu") so the
     JIT's target machine and data layout use the COFF/Win64 form that matches the
     kernel objects codegen emits; see GetTargetMachine() in pocl_llvm_wg.cc. */
  JITTargetMachineBuilder JTMB{
      llvm::Triple(llvm::Triple::normalize(TripleStr ? TripleStr : ""))};
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
    JITFailed = true;
    return -1;
  }
  TheJIT = JIT->release();

  /* Define PoCL's host callbacks (and the Windows stack probe) once, into the
     process-symbols JITDylib that every kernel JITDylib links against. */
  if (JITDylibSP PSJD = TheJIT->getProcessSymbolsJITDylib())
    defineHostSymbols(*PSJD);

#ifdef HOST_CPU_COMPILER_RT_LIBRARY
  if (!loadPrivateStaticArchive(HOST_CPU_COMPILER_RT_LIBRARY))
    POCL_MSG_WARN(
        "pocl_jit: could not load the compiler runtime archive '%s'; kernels "
        "needing compiler-rt/libgcc helpers may fail to resolve symbols\n",
        HOST_CPU_COMPILER_RT_LIBRARY);
#endif

#ifdef HAVE_DLFCN_H
  /* The JIT's default process-symbol search uses the global dlsym scope, which
     misses libpocl's private dependencies (libgcc_s for builtins like
     __aeabi_uldivmod or __extendhfsf2, libm before glibc 2.34) when an ICD
     loader dlopen'd libpocl RTLD_LOCAL. Also search libpocl's own handle, whose
     dlsym scope covers its dependency chain, the way each kernel .so's implicit
     -lgcc and DT_NEEDED entries do on the link path. This pins libpocl in
     memory, which the process-lifetime JIT does anyway. On failure (e.g.
     libpocl linked statically) kernels fall back to the process-global
     lookup. */
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
  /* The configured SONAME first (e.g. SLEEF's libsleefgnuabi.so.3 when LIBMVEC was pointed
     at it, so we don't depend on a system libmvec), then the configure-time path, then the
     plain glibc SONAME as a last resort. */
  const char *Candidates[] = {
#ifdef HOST_CPU_LIBMVEC_LIBRARY
      HOST_CPU_LIBMVEC_LIBRARY,
#endif
#ifdef HOST_CPU_LIBMVEC_LIBRARY_FALLBACK
      HOST_CPU_LIBMVEC_LIBRARY_FALLBACK,
#endif
      "libmvec.so.1",
  };
  bool VecMathLoaded = false;
  for (const char *VecMathLib : Candidates)
    if ((VecMathLoaded = addLibrarySearchGenerator(VecMathLib)))
      break;
  if (!VecMathLoaded)
    POCL_MSG_WARN(
        "pocl_jit: could not load the libmvec vector-math library; vectorized "
        "math kernels may fail to resolve their symbols\n");
#elif defined(ENABLE_HOST_CPU_VECTORIZE_SLEEF)
  /* The configured SONAME first, then the configure-time path, then the
     plain library name as a last resort. */
  const char *Candidates[] = {
#ifdef HOST_CPU_SLEEF_LIBRARY
      HOST_CPU_SLEEF_LIBRARY,
#endif
#ifdef HOST_CPU_SLEEF_LIBRARY_FALLBACK
      HOST_CPU_SLEEF_LIBRARY_FALLBACK,
#endif
      "libsleef.so",
  };
  bool VecMathLoaded = false;
  for (const char *VecMathLib : Candidates)
    if ((VecMathLoaded = addLibrarySearchGenerator(VecMathLib)))
      break;
  if (!VecMathLoaded)
    POCL_MSG_WARN(
        "pocl_jit: could not load the SLEEF vector-math library; vectorized "
        "math kernels may fail to resolve their symbols\n");
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
    POCL_MSG_WARN(
        "pocl_jit: could not load the SVML static archives; vectorized math "
        "kernels may fail to resolve their symbols\n");
#endif

  return 0;
}

void *pocl_jit_load_object(const char *Path, const char *UniqName,
                           const char *KernelLibraryBitcodePath) {
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
     across kernels/specializations and can be unloaded independently. The
     JITDylib pointer doubles as the opaque module handle. */
  std::string Name = std::string(UniqName ? UniqName : "kernel") + "#" +
                     std::to_string(JDCounter++);
  Expected<JITDylib &> JD = TheJIT->createJITDylib(std::move(Name));
  if (!JD) {
    POCL_MSG_ERR("pocl_jit: createJITDylib failed: %s\n",
                 toString(JD.takeError()).c_str());
    return nullptr;
  }

  if (JITDylib *BuiltinJD =
          getOrLoadBuiltinJITDylib(KernelLibraryBitcodePath))
    JD->addToLinkOrder(*BuiltinJD);

  if (Error Err = TheJIT->addObjectFile(*JD, std::move(*Buf))) {
    POCL_MSG_ERR("pocl_jit: addObjectFile('%s') failed: %s\n", Path,
                 toString(std::move(Err)).c_str());
    if (Error RmErr = TheJIT->getExecutionSession().removeJITDylib(*JD))
      consumeError(std::move(RmErr));
    return nullptr;
  }

  return &*JD;
}

void *pocl_jit_lookup(void *Handle, const char *SymbolName) {
  std::lock_guard<std::mutex> Lock(JITMutex);
  JITDylib *JD = static_cast<JITDylib *>(Handle);
  if (!TheJIT || JD == nullptr)
    return nullptr;

  /* ORC mangles the unmangled name for the target (e.g. adds the leading
     underscore on Mach-O), so we pass the plain C symbol name. Linking and
     relocation of the object happen here, on first lookup. */
  Expected<ExecutorAddr> Addr = TheJIT->lookup(*JD, SymbolName);
  if (!Addr) {
    /* Leave reporting to the caller, which retrieves the diagnostic through
       pocl_jit_last_error() (the dlerror() pattern), so only record it
       instead of printing an error here. */
    LastLookupError = toString(Addr.takeError());
    POCL_MSG_PRINT_LLVM("pocl_jit: lookup('%s') failed: %s\n", SymbolName,
                        LastLookupError.c_str());
    return nullptr;
  }
  return reinterpret_cast<void *>(Addr->getValue());
}

const char *pocl_jit_last_error(void) {
  std::lock_guard<std::mutex> Lock(JITMutex);
  return LastLookupError.empty() ? nullptr : LastLookupError.c_str();
}

int pocl_jit_unload(void *Handle) {
  std::lock_guard<std::mutex> Lock(JITMutex);
  JITDylib *JD = static_cast<JITDylib *>(Handle);
  if (JD == nullptr)
    return 0;
  if (TheJIT) {
    if (Error Err = TheJIT->getExecutionSession().removeJITDylib(*JD))
      POCL_MSG_ERR("pocl_jit: removeJITDylib failed: %s\n",
                   toString(std::move(Err)).c_str());
  }
  return 1;
}
