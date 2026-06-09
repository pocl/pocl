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

#include "pocl_llvm_orc.h"
#include "pocl_debug.h"

#include <llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/TargetParser/Triple.h>

/* absoluteSymbols() moved from Core.h to its own header around LLVM 20. */
#if __has_include(<llvm/ExecutionEngine/Orc/AbsoluteSymbols.h>)
#include <llvm/ExecutionEngine/Orc/AbsoluteSymbols.h>
#endif

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

using namespace llvm;
using namespace llvm::orc;

#ifdef ENABLE_PRINTF_IMMEDIATE_FLUSH
/* Host-side printf flush callback referenced by kernels built with immediate
   flush; defined in libpocl (lib/CL/devices/printf_buffer.c). */
extern "C" void pocl_flush_printf_buffer (char *buffer, uint32_t buffer_size);
#endif

namespace
{

/* The process-global JIT used to load all kernel objects, created lazily on
   the first pocl_jit_initialize() call. ORC's ExecutionSession is internally
   synchronized, so a single instance is shared across all CPU host devices.
   Access is additionally serialized by JITMutex so initialization races and
   the (already serialized) PoCL dlhandle cache stay consistent. */
std::unique_ptr<LLJIT> TheJIT;
std::mutex JITMutex;

/* JITDylib names must be unique within an ExecutionSession; a monotonic
   counter guarantees that regardless of the caller-supplied name. */
std::atomic<uint64_t> JDCounter{ 0 };

static bool
loadPermanentLibrary (const char *Library)
{
  return Library && Library[0]
         && !sys::DynamicLibrary::LoadLibraryPermanently (Library);
}

#ifdef _WIN32
/* Windows kernel objects are compiled freestanding (no CRT), so they reference
   a handful of compiler-emitted runtime symbols that no process library
   exports: the stack-probe (___chkstk_ms on MinGW, __chkstk on MSVC) and the
   mem* helpers (memcpy/memset/memmove/memcmp/strlen). PoCL ships these as two
   relocatable objects (lib/kernel/host/libchkstk.S + libmemory.c). We JIT-link
   them once into a shared "runtime" JITDylib and add it to every kernel JD's
   link order, the COFF analog of the absolute-symbol injection used for the
   host callbacks below. (On ELF/Mach-O these symbols resolve from libc in the
   process, so no runtime JD is needed there.) */
JITDylib *RuntimeJD = nullptr;
#endif

/* Bookkeeping for a single loaded kernel object. */
struct PoclJITModule
{
  JITDylib *JD;
};

/* Inject host-side symbols that kernel objects reference but that are defined
   in libpocl (the running process). Defining them as absolute symbols makes
   resolution independent of libpocl's dynamic-symbol visibility and of the
   mode it was dlopen()ed with. Process symbols (libc/libm/compiler-rt) are
   resolved automatically via the JIT's default process-symbols search order,
   so only PoCL's own host callbacks need to be defined here. */
void
defineHostSymbols (JITDylib &JD)
{
#ifdef ENABLE_PRINTF_IMMEDIATE_FLUSH
  SymbolMap Syms;
  Syms[TheJIT->mangleAndIntern ("pocl_flush_printf_buffer")]
      = { ExecutorAddr::fromPtr (&pocl_flush_printf_buffer),
          JITSymbolFlags::Exported | JITSymbolFlags::Callable };
  if (Error Err = JD.define (absoluteSymbols (std::move (Syms))))
    POCL_MSG_ERR ("pocl_jit: failed to define host symbols: %s\n",
                  toString (std::move (Err)).c_str ());
#else
  (void)JD;
#endif
}

#ifdef _WIN32
/* JIT-link a relocatable object file from 'Path' into 'JD'. Used to populate
   the shared runtime JITDylib with the freestanding helper objects. */
bool
loadObjectFileInto (JITDylib &JD, const std::string &Path)
{
  ErrorOr<std::unique_ptr<MemoryBuffer> > Buf = MemoryBuffer::getFile (
      Path.c_str (), /*IsText=*/false, /*RequiresNullTerminator=*/false);
  if (!Buf)
    {
      POCL_MSG_ERR ("pocl_jit: cannot read runtime object '%s': %s\n",
                    Path.c_str (), Buf.getError ().message ().c_str ());
      return false;
    }
  if (Error Err = TheJIT->addObjectFile (JD, std::move (*Buf)))
    {
      POCL_MSG_ERR ("pocl_jit: addObjectFile('%s') failed: %s\n", Path.c_str (),
                    toString (std::move (Err)).c_str ());
      return false;
    }
  return true;
}
#endif

} // namespace

int
pocl_jit_initialize (const char *TripleStr, const char *CPU,
                     const char *RuntimeLibDir)
{
  std::lock_guard<std::mutex> Lock (JITMutex);
  if (TheJIT)
    return 0;

  /* The native target and asm printer are already initialized by
     InitializeLLVM() during device setup; kernel codegen (which uses the same
     target) runs before any kernel object is loaded. */
  LLJITBuilder Builder;

  /* Force the JITLink-based object-linking layer (LLJIT still defaults to
     RuntimeDyld). JITLink resolves relocations, maps code into executable
     memory, and registers EH frames entirely in-process: this is what
     replaces the external linker. */
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
  Builder.setObjectLinkingLayerCreator (
#if LLVM_MAJOR >= 23
      [] (ExecutionSession &ES, jitlink::JITLinkMemoryManager &MM)
          -> Expected<std::unique_ptr<ObjectLayer> > {
        return std::make_unique<ObjectLinkingLayer> (ES, MM);
      }
#elif LLVM_MAJOR >= 21
      [] (ExecutionSession &ES) -> Expected<std::unique_ptr<ObjectLayer> > {
        return std::make_unique<ObjectLinkingLayer> (ES);
      }
#else
      [] (ExecutionSession &ES, const llvm::Triple &)
          -> Expected<std::unique_ptr<ObjectLayer> > {
        return std::make_unique<ObjectLinkingLayer> (ES);
      }
#endif
  );

  JITTargetMachineBuilder JTMB{ llvm::Triple (TripleStr ? TripleStr : "") };
  if (CPU && CPU[0])
    JTMB.setCPU (CPU);
  Builder.setJITTargetMachineBuilder (std::move (JTMB));

  Expected<std::unique_ptr<LLJIT> > JIT = Builder.create ();
  if (!JIT)
    {
      POCL_MSG_ERR ("pocl_jit: LLJIT creation failed: %s\n",
                    toString (JIT.takeError ()).c_str ());
      return -1;
    }
  TheJIT = std::move (*JIT);

  /* When CPU codegen vectorizes libm calls (expf, sinf, ...) it lowers them to
     a vector-math library's symbols (e.g. libmvec's _ZGVdN8v_expf). Which
     library is chosen at configure time, matching the codegen veclib selection
     in pocl_llvm_wg.cc. On the old link path that library was pulled in as a
     NEEDED lib of the kernel .so; the JIT does no such link, so load it into the
     process here (LoadLibraryPermanently uses RTLD_GLOBAL, so its symbols join
     the scope the JIT searches). SVML is intentionally not handled: it is a
     static-only library incompatible with the JIT, so SVML builds keep the link
     path (HOST_CPU_ENABLE_JIT is off there). */
#if defined(ENABLE_HOST_CPU_VECTORIZE_LIBMVEC)
  const char *VecMathLib = "libmvec.so.1";
  if (!loadPermanentLibrary (VecMathLib))
    POCL_MSG_PRINT_LLVM (
        "pocl_jit: could not load vector-math library '%s'; vectorized math "
        "kernels may fail to resolve their symbols\n",
        VecMathLib);
#elif defined(ENABLE_HOST_CPU_VECTORIZE_SLEEF)
  const char *VecMathLib = nullptr;
  bool VecMathLoaded = false;
#ifdef HOST_CPU_SLEEF_LIBRARY
  VecMathLib = HOST_CPU_SLEEF_LIBRARY;
  VecMathLoaded = loadPermanentLibrary (VecMathLib);
#endif
#ifdef HOST_CPU_SLEEF_LIBRARY_FALLBACK
  if (!VecMathLoaded)
    {
      VecMathLib = HOST_CPU_SLEEF_LIBRARY_FALLBACK;
      VecMathLoaded = loadPermanentLibrary (VecMathLib);
    }
#endif
  if (!VecMathLoaded)
    {
      VecMathLib = "libsleef.so";
      VecMathLoaded = loadPermanentLibrary (VecMathLib);
    }
  if (!VecMathLoaded)
    POCL_MSG_PRINT_LLVM (
        "pocl_jit: could not load vector-math library '%s'; vectorized math "
        "kernels may fail to resolve their symbols\n",
        VecMathLib);
#endif

#ifdef _WIN32
  /* Build the shared runtime JITDylib from the freestanding helper objects.
     A failure here is not fatal at init time (a kernel that doesn't reference
     these symbols still links), but kernels that do reference them will fail
     to look up later, so report it. */
  if (RuntimeLibDir && RuntimeLibDir[0])
    {
      Expected<JITDylib &> RJD = TheJIT->createJITDylib ("pocl_runtime");
      if (!RJD)
        {
          POCL_MSG_ERR ("pocl_jit: creating runtime JITDylib failed: %s\n",
                        toString (RJD.takeError ()).c_str ());
        }
      else
        {
          RuntimeJD = &*RJD;
          std::string Dir (RuntimeLibDir);
          loadObjectFileInto (*RuntimeJD, Dir + "/libchkstk.obj");
          loadObjectFileInto (*RuntimeJD, Dir + "/libmemory.obj");
        }
    }
#else
  (void)RuntimeLibDir;
#endif

  return 0;
}

void *
pocl_jit_load_object (const char *Path, const char *UniqName)
{
  std::lock_guard<std::mutex> Lock (JITMutex);
  if (!TheJIT)
    {
      POCL_MSG_ERR ("pocl_jit: load_object called before initialize\n");
      return nullptr;
    }

  ErrorOr<std::unique_ptr<MemoryBuffer> > Buf = MemoryBuffer::getFile (
      Path, /*IsText=*/false, /*RequiresNullTerminator=*/false);
  if (!Buf)
    {
      POCL_MSG_ERR ("pocl_jit: cannot read object '%s': %s\n", Path,
                    Buf.getError ().message ().c_str ());
      return nullptr;
    }

  /* Give each loaded object its own JITDylib so symbol names never collide
     across kernels/specializations and can be unloaded independently. */
  std::string Name = std::string (UniqName ? UniqName : "kernel") + "#"
                     + std::to_string (JDCounter.fetch_add (1));
  Expected<JITDylib &> JD = TheJIT->createJITDylib (std::move (Name));
  if (!JD)
    {
      POCL_MSG_ERR ("pocl_jit: createJITDylib failed: %s\n",
                    toString (JD.takeError ()).c_str ());
      return nullptr;
    }

  defineHostSymbols (*JD);

#ifdef _WIN32
  /* Resolve the freestanding runtime helpers (stack probe, mem*) from the
     shared runtime JITDylib, and crucially do so *before* the process symbols.
     A kernel reaches these helpers with a direct call (PCRel32, +-2GB range);
     the process CRT's copies live in a loaded DLL far outside that range, so a
     direct call to them overflows the relocation (COFF/x86-64 JITLink does not
     insert a far-call stub the way the ELF/Mach-O path does). The runtime JD's
     copies are JIT-allocated next to the kernel code, so the call is in range.
     The kernel JD's link order after createJITDylib is [self, <process>,
     <platform>]; insert the runtime JD right after self so it wins over the
     process CRT. */
  if (RuntimeJD)
    {
      JITDylibSearchOrder Order;
      (*JD).withLinkOrderDo (
          [&] (const JITDylibSearchOrder &O) { Order = O; });
      Order.insert (Order.begin () + (Order.empty () ? 0 : 1),
                    { RuntimeJD, JITDylibLookupFlags::MatchExportedSymbolsOnly });
      (*JD).setLinkOrder (std::move (Order),
                          /*LinkAgainstThisJITDylibFirst=*/false);
    }
#endif

  if (Error Err = TheJIT->addObjectFile (*JD, std::move (*Buf)))
    {
      POCL_MSG_ERR ("pocl_jit: addObjectFile('%s') failed: %s\n", Path,
                    toString (std::move (Err)).c_str ());
      if (Error RmErr = TheJIT->getExecutionSession ().removeJITDylib (*JD))
        consumeError (std::move (RmErr));
      return nullptr;
    }

  return new PoclJITModule{ &*JD };
}

void *
pocl_jit_lookup (void *Handle, const char *SymbolName)
{
  std::lock_guard<std::mutex> Lock (JITMutex);
  PoclJITModule *M = static_cast<PoclJITModule *> (Handle);
  if (!TheJIT || M == nullptr || M->JD == nullptr)
    return nullptr;

  /* ORC mangles the unmangled name for the target (e.g. adds the leading
     underscore on Mach-O), so we pass the plain C symbol name. Linking and
     relocation of the object happen here, on first lookup. */
  Expected<ExecutorAddr> Addr = TheJIT->lookup (*M->JD, SymbolName);
  if (!Addr)
    {
      /* Not necessarily fatal: callers may probe alternative names. */
      consumeError (Addr.takeError ());
      return nullptr;
    }
  return reinterpret_cast<void *> (Addr->getValue ());
}

int
pocl_jit_unload (void *Handle)
{
  std::lock_guard<std::mutex> Lock (JITMutex);
  PoclJITModule *M = static_cast<PoclJITModule *> (Handle);
  if (M == nullptr)
    return 0;
  if (TheJIT && M->JD)
    {
      if (Error Err = TheJIT->getExecutionSession ().removeJITDylib (*M->JD))
        POCL_MSG_ERR ("pocl_jit: removeJITDylib failed: %s\n",
                      toString (std::move (Err)).c_str ());
    }
  delete M;
  return 1;
}
