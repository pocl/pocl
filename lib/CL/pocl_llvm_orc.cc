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

#include <llvm/ADT/StringSet.h>
#include <llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectFileInterface.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h>
#include <llvm/Object/Archive.h>
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
#include <set>
#include <string>
#include <vector>

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

/* Same story for the FP16 soft-float conversion helpers codegen emits for
   double/float <-> _Float16: defined in libgcc, exported by no DLL. Loading
   libgcc.a into the JIT to supply them does NOT work here -- their COFF members
   carry .pdata/.xdata SEH unwind whose IMAGE_REL_AMD64_ADDR32NB fixups JITLink
   cannot relocate into the high-mapped kernel image ("out of range of Pointer32
   fixup"). Hand kernels libpocl's own statically-linked copies instead.

   But mingw's libgcc uses the legacy soft-float ABI for these: the half is
   carried in a GPR as an unsigned short, whereas LLVM's codegen calls them with
   the _Float16 ABI, passing/returning the half in the low 16 bits of an XMM
   register. Handing kernels libgcc's symbols directly therefore moves the half
   through the wrong register -- e.g. `fptrunc double to half` (emitted for the
   FP16 pow/atan2 builtins, which round a double result) reads a stale XMM0 and
   returns garbage. Wrap libgcc's routines in the _Float16 ABI and inject the
   wrappers. On f16c CPUs the half<->float pair is done in hardware, so only the
   double conversions strictly need this, but wrap all four for CPUs without
   f16c. (The Windows JIT build uses the Clang toolchain, which supports
   _Float16.) */
/* libgcc's routines, reached under private names via asm labels so we can give
   them their real (integer-GPR) prototypes without clashing with the compiler's
   builtin declarations. */
extern "C" {
unsigned short poclLibgccTruncSFHF2(float) __asm__("__truncsfhf2");
unsigned short poclLibgccTruncDFHF2(double) __asm__("__truncdfhf2");
float poclLibgccExtendHFSF2(unsigned short) __asm__("__extendhfsf2");
double poclLibgccExtendHFDF2(unsigned short) __asm__("__extendhfdf2");
}

static _Float16 poclTruncSFHF2(float A) {
  unsigned short H = poclLibgccTruncSFHF2(A);
  _Float16 R;
  __builtin_memcpy(&R, &H, sizeof R);
  return R;
}
static _Float16 poclTruncDFHF2(double A) {
  unsigned short H = poclLibgccTruncDFHF2(A);
  _Float16 R;
  __builtin_memcpy(&R, &H, sizeof R);
  return R;
}
static float poclExtendHFSF2(_Float16 A) {
  unsigned short H;
  __builtin_memcpy(&H, &A, sizeof H);
  return poclLibgccExtendHFSF2(H);
}
static double poclExtendHFDF2(_Float16 A) {
  unsigned short H;
  __builtin_memcpy(&H, &A, sizeof H);
  return poclLibgccExtendHFDF2(H);
}
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

/* ORC symbol source model.

   Kernel objects can only refer to symbols introduced by codegen, the linked
   kernel library, or PoCL's host ABI. Keep each source class mapped to one
   ORC mechanism, mirroring what the shared-library link path provided:

   - Process/global scope: LLJIT's default process-symbol generator. This is
     the JIT analogue of loader-visible process symbols.
   - libpocl and its private dependency closure: DynamicLibrarySearchGenerator
     on libpocl's own handle. This mirrors a kernel shared library's DT_NEEDED
     edge to libpocl and, on POSIX, libpocl's dependencies such as libgcc_s or
     libm when the ICD loader used RTLD_LOCAL.
   - Configure-time vector math library: DynamicLibrarySearchGenerator for
     shared veclibs (libmvec, SLEEF), or a request-driven archive generator for
     static SVML/libirc. These are the same absolute link inputs the non-JIT
     final link receives.
   - Compiler runtime helpers: StaticLibraryDefinitionGenerator for the
     installed compiler-rt/libgcc archive. This replaces the Clang driver's
     implicit runtime library link.
   - Host ABI callbacks and platform helper symbols: absoluteSymbols in the
     process-symbols JITDylib. This covers deliberately hidden libpocl entry
     points such as the printf flush callback and MinGW's stack probe.

   PoCL kernel-library builtins are not a runtime source: they are linked into
   the kernel object at IR time before codegen (see linkKernelLibraryForJIT in
   pocl_llvm_wg.cc), so the object is self-contained for them.

   A new unresolved CPU-kernel symbol should fit one of these buckets; if it
   does not, first identify which codegen or ABI rule introduced it and then
   add the corresponding generator here rather than treating it as a one-off. */

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

/* Unlike StaticLibraryDefinitionGenerator, this does not build an archive-wide
   symbol map at JIT initialization. SVML is only needed after vectorized
   builtins introduce a __svml_* reference, so keep the archive paths until
   that first lookup and then consult only the requested archive-index names. */
class LazySVMLDefinitionGenerator : public DefinitionGenerator {
  struct IndexedArchive {
    std::string Path;
    std::unique_ptr<MemoryBuffer> Buffer;
    std::unique_ptr<object::Archive> Archive;
    std::set<uint64_t> LoadedOffsets;
  };

  struct SelectedMember {
    IndexedArchive *Source;
    uint64_t Offset;
    MemoryBufferRef Buffer;
  };

public:
  LazySVMLDefinitionGenerator(ObjectLayer &Layer, StringRef IRCPath,
                              StringRef SVMLPath, char GlobalPrefix)
      : Layer(Layer), IRCPath(IRCPath), SVMLPath(SVMLPath),
        GlobalPrefix(GlobalPrefix) {}

  Error tryToGenerate(LookupState &, LookupKind K, JITDylib &JD,
                      JITDylibLookupFlags,
                      const SymbolLookupSet &Symbols) override {
    if (K != LookupKind::Static)
      return Error::success();

    if (!Active) {
      bool NeedsSVML = false;
      for (const auto &[Name, _] : Symbols)
        if (isSVMLName(*Name)) {
          NeedsSVML = true;
          break;
        }
      if (!NeedsSVML)
        return Error::success();

      /* Keep this before opening either archive: it is the deterministic test
         that unrelated JIT lookups leave the provider untouched. */
      if (getenv("POCL_FAULT_INJECT_JIT_SVML"))
        return createStringError(
            inconvertibleErrorCode(),
            "injected SVML provider failure (POCL_FAULT_INJECT_JIT_SVML)");

      if (Error Err = activate())
        return Err;
    }

    StringSet<> Requested;
    for (const auto &[Name, _] : Symbols)
      Requested.insert(*Name);

    std::vector<SelectedMember> Members;
    if (Error Err = selectMembers(*IRC, Requested, Members))
      return Err;
    if (!Requested.empty())
      if (Error Err = selectMembers(*SVML, Requested, Members))
        return Err;

    for (const SelectedMember &Member : Members)
      if (Error Err = addMember(JD, Member))
        return Err;

    return Error::success();
  }

private:
  bool isSVMLName(StringRef Name) const {
    if (GlobalPrefix != '\0') {
      if (Name.empty() || Name.front() != GlobalPrefix)
        return false;
      Name = Name.drop_front();
    }
    return Name.starts_with("__svml_");
  }

  static Error openArchive(std::unique_ptr<IndexedArchive> &Out,
                           StringRef Path) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer = MemoryBuffer::getFile(
        Path, /*IsText=*/false, /*RequiresNullTerminator=*/false);
    if (!Buffer)
      return createStringError(inconvertibleErrorCode(),
                               "pocl_jit: could not open SVML archive '%s': %s",
                               Path.str().c_str(),
                               Buffer.getError().message().c_str());

    Expected<std::unique_ptr<object::Archive>> Archive =
        object::Archive::create((*Buffer)->getMemBufferRef());
    if (!Archive)
      return createStringError(
          inconvertibleErrorCode(),
          "pocl_jit: could not parse SVML archive '%s': %s", Path.str().c_str(),
          toString(Archive.takeError()).c_str());
    if (!(*Archive)->hasSymbolTable())
      return createStringError(
          inconvertibleErrorCode(),
          "pocl_jit: SVML archive '%s' has no symbol table",
          Path.str().c_str());

    auto Indexed = std::make_unique<IndexedArchive>();
    Indexed->Path = Path.str();
    Indexed->Buffer = std::move(*Buffer);
    Indexed->Archive = std::move(*Archive);
    Out = std::move(Indexed);
    return Error::success();
  }

  Error activate() {
    if (Error Err = openArchive(IRC, IRCPath))
      return Err;
    if (Error Err = openArchive(SVML, SVMLPath))
      return Err;
    Active = true;
    POCL_MSG_PRINT_LLVM("pocl_jit: activating lazy SVML archive provider\n");
    return Error::success();
  }

  static Error selectMembers(IndexedArchive &Source, StringSet<> &Requested,
                             std::vector<SelectedMember> &Members) {
    std::set<uint64_t> SelectedOffsets;
    for (const object::Archive::Symbol &Symbol : Source.Archive->symbols()) {
      StringRef Name = Symbol.getName();
      if (Requested.find(Name) == Requested.end())
        continue;

      Expected<object::Archive::Child> Child = Symbol.getMember();
      if (!Child)
        return Child.takeError();
      uint64_t Offset = Child->getDataOffset();
      Requested.erase(Name);
      if (Source.LoadedOffsets.count(Offset) ||
          !SelectedOffsets.insert(Offset).second)
        continue;

      Expected<MemoryBufferRef> Buffer = Child->getMemoryBufferRef();
      if (!Buffer)
        return Buffer.takeError();
      Members.push_back({&Source, Offset, *Buffer});
    }
    return Error::success();
  }

  Error addMember(JITDylib &JD, const SelectedMember &Member) {
    std::string Identifier =
        (Member.Source->Path + "[" + std::to_string(Member.Offset) + "](" +
         Member.Buffer.getBufferIdentifier().str() + ")");
    std::unique_ptr<MemoryBuffer> Buffer =
        MemoryBuffer::getMemBuffer(Member.Buffer.getBuffer(), Identifier,
                                   /*RequiresNullTerminator=*/false);
    auto Interface = getObjectFileInterface(Layer.getExecutionSession(),
                                            Buffer->getMemBufferRef());
    if (!Interface)
      return Interface.takeError();
    if (Error Err = Layer.add(JD, std::move(Buffer), std::move(*Interface)))
      return Err;

    Member.Source->LoadedOffsets.insert(Member.Offset);
    POCL_MSG_PRINT_LLVM("pocl_jit: materializing SVML archive member %s\n",
                        Identifier.c_str());
    return Error::success();
  }

  ObjectLayer &Layer;
  std::string IRCPath;
  std::string SVMLPath;
  char GlobalPrefix;
  bool Active = false;
  std::unique_ptr<IndexedArchive> IRC;
  std::unique_ptr<IndexedArchive> SVML;
};

static bool addLazySVMLGenerator(const char *IRCPath, const char *SVMLPath) {
  if (!IRCPath || !IRCPath[0] || !SVMLPath || !SVMLPath[0])
    return false;
  JITDylibSP PSJD = TheJIT->getProcessSymbolsJITDylib();
  if (!PSJD)
    return false;
  PSJD->addGenerator(std::make_unique<LazySVMLDefinitionGenerator>(
      TheJIT->getObjLinkingLayer(), IRCPath, SVMLPath,
      TheJIT->getDataLayout().getGlobalPrefix()));
  return true;
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
  Syms[TheJIT->mangleAndIntern("__truncsfhf2")] = {
      ExecutorAddr::fromPtr(&poclTruncSFHF2),
      JITSymbolFlags::Exported | JITSymbolFlags::Callable};
  Syms[TheJIT->mangleAndIntern("__truncdfhf2")] = {
      ExecutorAddr::fromPtr(&poclTruncDFHF2),
      JITSymbolFlags::Exported | JITSymbolFlags::Callable};
  Syms[TheJIT->mangleAndIntern("__extendhfsf2")] = {
      ExecutorAddr::fromPtr(&poclExtendHFSF2),
      JITSymbolFlags::Exported | JITSymbolFlags::Callable};
  Syms[TheJIT->mangleAndIntern("__extendhfdf2")] = {
      ExecutorAddr::fromPtr(&poclExtendHFDF2),
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
  /* Test hook: setting POCL_FAULT_INJECT_JIT simulates an environment where
     the JIT cannot be brought up (unsupported triple, executor setup failure),
     to exercise the linker fallback. */
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
  /* Do not index either archive here. Most cached kernels do not need SVML,
     and the local generator opens both archives only after a __svml_* lookup.
     Once active it also handles the non-SVML libirc helpers introduced by a
     materialized libsvml member. */
  bool VecMathLoaded = false;
#if defined(HOST_CPU_IRC_LIBRARY) && defined(HOST_CPU_SVML_LIBRARY)
  VecMathLoaded =
      addLazySVMLGenerator(HOST_CPU_IRC_LIBRARY, HOST_CPU_SVML_LIBRARY);
#endif
  if (!VecMathLoaded)
    POCL_MSG_WARN(
        "pocl_jit: could not configure the lazy SVML archive provider; "
        "vectorized math kernels may fail to resolve their symbols\n");
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
