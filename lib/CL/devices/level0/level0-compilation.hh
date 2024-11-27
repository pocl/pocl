/// level0-compilation.hh - multithreaded compilation
/// for LevelZero Compute API devices.
///
/// Overall design:
///
/// A single Level0 driver can have:
///   any number of independent contexts
///   stable device handles that are valid across contexts
///
/// Level0Driver has a single instance of Level0CompilationJobScheduler.
/// Scheduler ows one Level0CompilerJobQueue and multiple instances of
/// Level0CompilerThread, which pick up jobs (Level0CompilationJob) from the
/// queue. The Level0CompilationJob holds an instance of Level0Build (build with
/// specialization), which at the end of compilation is moved into its
/// Level0Program.
///
/// Usage: users should only interact with the JobScheduler instance. This is
/// because it's possible to launch multiple jobs (with different
/// specialization) and let them run in background, without waiting for the
/// result. Each job keeps a shared_ptr to the Program (and the Kernel if it's a
/// JIT kernel job). Therefore the user does NOT own the objects returned by
/// JobScheduler's APIs like createProgram, createKernel etc, the owner is
/// JobScheduler.

/// Copyright (c) 2022-2023 Michal Babej / Intel Finland Oy
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to
/// deal in the Software without restriction, including without limitation the
/// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
/// sell copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
/// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
/// IN THE SOFTWARE.

#include <ze_api.h>

#include "config.h"

#ifdef ENABLE_NPU
#include <ze_graph_ext.h>
#include <ze_graph_profiling_ext.h>

#define GRAPH_EXT_NAME "ZE_extension_graph_1_5"
#define GRAPH_EXT_VERSION ZE_GRAPH_EXT_VERSION_1_5
typedef ze_graph_dditable_ext_t graph_dditable_ext_t;
#endif

#ifndef POCL_LIB_CL_DEVICES_LEVEL0_LEVEL0_COMPILATION_HH
#define POCL_LIB_CL_DEVICES_LEVEL0_LEVEL0_COMPILATION_HH

#include <condition_variable>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace pocl {

/// specialization flags

/// true = -ze-opt-level=2, false = -ze-opt-disable
// bool Optimized;
/// adds -ze-opt-greater-than-4GB-buffer-required;
/// only meaningful if the device supports 64bit buffers
// bool LargeOffsets;
/// true = "-g"
// bool Debug;
/// true = ' -ze-opt-large-register-file'
// bool SmallWGSize;

struct BuildSpecialization {
  bool Optimize;
  bool LargeOffsets;
  bool Debug;
  bool SmallWGSize;

  bool operator<(const BuildSpecialization &Other) const {
    if (Optimize < Other.Optimize)
      return true;
    if (LargeOffsets < Other.LargeOffsets)
      return true;
    if (SmallWGSize < Other.SmallWGSize)
      return true;
    if (Debug < Other.Debug)
      return true;
    return false;
  }

  bool operator==(const BuildSpecialization &Other) const {
    return (Optimize == Other.Optimize) &&
           (LargeOffsets == Other.LargeOffsets) &&
           (SmallWGSize == Other.SmallWGSize) &&
           (Debug == Other.Debug);
  }

  bool operator!=(const BuildSpecialization &Other) const {
    return !(*this == Other);
  }
};

class Level0Kernel;
typedef std::shared_ptr<Level0Kernel> Level0KernelSPtr;

class Level0ProgramBase;
typedef std::shared_ptr<Level0ProgramBase> Level0ProgramBaseSPtr;

class Level0Program;
typedef std::shared_ptr<Level0Program> Level0ProgramSPtr;

class Level0BuildBase;
typedef std::unique_ptr<Level0BuildBase> Level0BuildBaseUPtr;

class Level0Build;
typedef std::unique_ptr<Level0Build> Level0BuildUPtr;

class Level0ProgramBuild;
typedef std::unique_ptr<Level0ProgramBuild> Level0ProgramBuildUPtr;

class Level0KernelBuild;
typedef std::unique_ptr<Level0KernelBuild> Level0KernelBuildUPtr;

class Level0JITProgramBuild;
typedef std::unique_ptr<Level0JITProgramBuild> Level0JITProgramBuildUPtr;

class Level0CompilationJob;
typedef std::shared_ptr<Level0CompilationJob> Level0CompilationJobSPtr;

class Level0CompilerThread;
class Level0CompilationJobScheduler;

#ifdef ENABLE_NPU
class Level0BuiltinKernel;
typedef std::shared_ptr<Level0BuiltinKernel> Level0BuiltinKernelSPtr;

class Level0BuiltinProgram;
typedef std::shared_ptr<Level0BuiltinProgram> Level0BuiltinProgramSPtr;

class Level0BuiltinProgramBuild;
typedef std::unique_ptr<Level0BuiltinProgramBuild>
    Level0BuiltinProgramBuildUPtr;
#endif

///
/// \brief Stores a map of Specializations to ZE kernel+module handles,
///        for a particular cl_kernel + device (= kernel->data[device_i])
///
/// New handles are dynamically created by getOrCreateForBuild() when they're
/// needed. If the program was created with lazy-compilation, this triggers a
/// new compilation
///
class Level0Kernel {

public:
  Level0Kernel(const std::string N);
  ~Level0Kernel();

  Level0Kernel(Level0Kernel const &) = delete;
  Level0Kernel& operator=(Level0Kernel const &) = delete;
  Level0Kernel(Level0Kernel const &&) = delete;
  Level0Kernel& operator=(Level0Kernel &&) = delete;

  /// this is necessary for the Level0Queue's run() function to lock the kernel
  std::mutex &getMutex() { return Mutex; }

  ///  returns any existing handle.
  /// Used only in pocl_level0_local_size_optimizer() for zeKernelSuggestGroupSize().
  /// For getting a handle for running a kernel, use Level0Program->getBestKernel() ///
  ze_kernel_handle_t getAnyCreated();

  const std::string &getName() { return Name; }
  const std::string &getCacheUUID() { return CacheUUID; }

  void setIndirectAccess(ze_kernel_indirect_access_flag_t AccessFlag,
                         bool Value);
  void setAccessedPointers(const std::map<void *, size_t> &Ptrs);
  ze_kernel_indirect_access_flags_t getIndirectFlags() {
    return IndirectAccessFlags;
  }
  const std::map<void *, size_t> &getAccessedPointers() {
    return AccessedPointers;
  }

  /// returns (or creates a new) ze_kernel_handle_t + ze_module_handle_t
  /// for a particular program build specialization
  ze_kernel_handle_t getOrCreateForBuild(Level0Build *Build);

private:
  std::mutex Mutex;
  /// map of program build specializations to Kernel handles
  std::map<BuildSpecialization, ze_kernel_handle_t> KernelHandles;
  /// for indirect access
  std::map<void *, size_t> AccessedPointers;

  std::string Name;
  std::string CacheUUID;
  ze_kernel_indirect_access_flags_t IndirectAccessFlags = 0;

  bool createForBuild(BuildSpecialization Spec, ze_module_handle_t Mod);
};

class Level0ProgramBase {
public:
  Level0ProgramBase(ze_context_handle_t Ctx, ze_device_handle_t Dev,
                    const char *CDir, const std::string &UUID)
      : CacheDir(CDir), CacheUUID(UUID), ContextH(Ctx), DeviceH(Dev) {}
  virtual ~Level0ProgramBase() {};

  virtual bool init() = 0;
  virtual bool addFinishedBuild(Level0BuildBaseUPtr Build) = 0;
  ze_device_handle_t getDevice() const { return DeviceH; }
  const std::string &getBuildLog() const { return BuildLog; }
  const std::string &getCacheDir() const { return CacheDir; }
  const std::string &getCacheUUID() const { return CacheUUID; }

protected:
  /// all data except Builds, Kernels, ExtractedSPIRV,
  /// ProgramLLVMCtx & BuildLog are const
  std::mutex Mutex;

  /// cl_program's pocl cache dir
  std::string CacheDir;
  /// UUID used to determine compatibility of native binaries in cache
  std::string CacheUUID;

  /// compilation output
  std::string BuildLog;

  /// these are the destination Context/Device, not the build thread's
  /// Context/Device (though the device is the same for both, context isn't)
  ze_context_handle_t ContextH;
  ze_device_handle_t DeviceH;
};

///
/// \brief Stores a set of specialized builds for a particular
/// cl_program + cl_device_id (=program->data[device_id]).
///
/// Note that this takes SPIRV + compiler options + SpecConstants combination
/// as input, and these are considered constant, so if the cl_program is rebuilt,
/// the instance of this class needs to be destroyed & new one recreated.
///
/// Since there might still be a compile job scheduled/running that will need
/// the instance, it needs to be handled properly (-> std::shared_ptr)
///
class Level0Program : public Level0ProgramBase {

public:
  Level0Program(ze_context_handle_t Ctx,
                ze_device_handle_t Dev,
                bool EnableJIT, bool Optimize,
                uint32_t NumSpecs, uint32_t *SpecIDs,
                const void **SpecValues, size_t *SpecValSizes,
                std::vector<uint8_t> &SpvData,
                std::vector<char> &ProgramBCData,
                const char* CDir,
                const std::string &UUID);
  virtual bool init() override;
  virtual ~Level0Program();

  Level0Program(Level0Program const &) = delete;
  Level0Program& operator=(Level0Program const &) = delete;
  Level0Program(Level0Program const &&) = delete;
  Level0Program& operator=(Level0Program &&) = delete;

  const std::vector<uint8_t> &getSPIRV() const { return SPIRV; }
  const std::vector<uint8_t> &getLinkinSPIRV() const { return LinkinSPIRV; }
  ze_module_constants_t getSpecConstants() const { return SpecConstants; }
  bool isJITCompiled() const { return JITCompilation; }
  bool isOptimized() const { return Optimize; }

  /// for cl_kernel creation device->ops callback
  Level0Kernel *createKernel(const std::string Name);
  /// for cl_kernel deletion device->ops callback
  bool releaseKernel(Level0Kernel *Kernel);

  ///
  /// \brief returns the best available specialization of a Kernel,
  ///        for the given set of specialization options (currently just one).
  /// \param [in] Kernel the Level0Kernel to search for
  /// \param [in] LargeOffset specialization option
  /// \param [out] Mod the ze_module_handle_t of the found specialization, or null
  /// \param [out] Ker the ze_kernel_handle_t of the found specialization, or null
  /// \returns false if can't find any build specialization
  ///
  bool getBestKernel(Level0Kernel *Kernel, bool MustUseLargeOffsets,
                     bool CanBeSmallWG, ze_module_handle_t &Mod,
                     ze_kernel_handle_t &Ker);

  ///
  /// \brief addFinishedBuild adds a finished program/kernel build to the
  ///        member holding builds (if the build was successful), or in case
  ///        of error, updates the program build log and drops the Build
  /// \param [in] Build the build to process
  ///
  virtual bool addFinishedBuild(Level0BuildBaseUPtr Build) override;

  ///
  /// \brief extractKernelSPIRV extracts the SPIR-V of a single kernel from the
  ///        whole program's SPIR-V. Required for JIT compilation of kernels
  /// \param [in] Kernel the kernel to extract
  /// \param [out] SPIRV variable to hold the extracted SPIR-V
  /// \return true if successful
  ///
  bool extractKernelSPIRV(std::string &KernelName, std::vector<uint8_t> &SPIRV);

  Level0JITProgramBuild *getLinkinBuild(BuildSpecialization Spec);

private:
  /// full program builds with specializations
  std::list<Level0ProgramBuildUPtr> ProgBuilds;
  /// kernel builds with specializations, for JIT compilation
  std::list<Level0KernelBuildUPtr> KernBuilds;
  /// JIT link-in builds with specializations
  std::list<Level0JITProgramBuildUPtr> JITProgBuilds;

  /// map of kernel names to extracted SPIR-V; this is not using
  /// Level0Kernel as map key, because there can be multiple kernel
  /// object with the same name, for which we only need one key-val
  std::map<std::string, std::vector<uint8_t>> ExtractedKernelSPIRVCache;

  /// pocl::ProgramWithContext object, has its own lock
  void *ProgramLLVMCtx;

  std::list<Level0KernelSPtr> Kernels;

  ////////////////////////////////////////////////////////////////////////

  /// SPIR-V binary (= compilation input)
  std::vector<uint8_t> SPIRV;
  /// the LLVM binary bitcode
  std::vector<char> ProgramBC;

  /// SPIR-V binary for the "linkin" module
  std::vector<uint8_t> LinkinSPIRV;

  /// spec constants used for compilation
  ze_module_constants_t SpecConstants;
  /// storage for actual data of ze_module_constants_t
  std::vector<uint32_t> ConstantIds;
  std::vector<const void*> ConstantVoidPtrs;
  std::vector<std::vector<uint8_t>> ConstantValues;

  /// true = compile each kernel separately before launch
  /// false = compile the whole program once
  bool JITCompilation;
  bool Optimize;

  /// setup the ze_module_constants_t from cl_program's Spec constant data.
  void setupSpecConsts(uint32_t NumSpecs, const uint32_t *SpecIDs,
                      const void **SpecValues, size_t *SpecValSizes);
};

#ifdef ENABLE_NPU
class Level0BuiltinKernel {
public:
  Level0BuiltinKernel(const std::string N); //, ze_graph_handle_t G);
  ~Level0BuiltinKernel() {};

  Level0BuiltinKernel(Level0BuiltinKernel const &) = delete;
  Level0BuiltinKernel &operator=(Level0BuiltinKernel const &) = delete;
  Level0BuiltinKernel(Level0BuiltinKernel const &&) = delete;
  Level0BuiltinKernel &operator=(Level0BuiltinKernel &&) = delete;

  /// this is necessary for the Level0Queue's run() function to lock the kernel
  std::mutex &getMutex() { return Mutex; }

  const std::string &getName() { return Name; }
  const std::string &getCacheUUID() { return CacheUUID; }

private:
  std::mutex Mutex;
  ze_graph_handle_t GraphH;

  std::string Name;
  std::string CacheUUID;
};

class Level0BuiltinProgram : public Level0ProgramBase {
public:
  Level0BuiltinProgram(ze_context_handle_t Ctx, ze_device_handle_t Dev,
                       size_t NumBuiltinKernels, char **BuiltinKernelNames,
                       void *BuiltinKernelIDs,    // IDs for DBKs
                       void **BuiltinKernelAttrs, // Attrs for DBKs
                       const char *CDir, const std::string &UUID);
  virtual bool init() override;
  virtual ~Level0BuiltinProgram();

  Level0BuiltinProgram(Level0BuiltinProgram const &) = delete;
  Level0BuiltinProgram &operator=(Level0BuiltinProgram const &) = delete;
  Level0BuiltinProgram(Level0BuiltinProgram const &&) = delete;
  Level0BuiltinProgram &operator=(Level0BuiltinProgram &&) = delete;

  const std::vector<std::string> &getKernelNames() { return KernelNames; }
  const std::vector<unsigned> &getKernelIDs() { return KernelIDs; }
  const std::vector<void *> &getKernelAttrs() { return KernelAttrs; }
  bool isDBK() const { return IsDBK; }

  /// for cl_kernel creation device->ops callback
  Level0BuiltinKernel *
  createKernel(const std::string Name); //, ze_graph_handle_t G);
  /// for cl_kernel deletion device->ops callback
  bool releaseKernel(Level0BuiltinKernel *Kernel);

  ///
  /// \brief returns the best available specialization of a Kernel,
  ///        for the given set of specialization options
  /// \param [in] Kernel the Level0Kernel to search for
  /// \param [out] Ker the ze_graph_handle_t of the found specialization, or
  /// null
  /// \returns false if can't find any build for the kernel
  ///
  bool getBestKernel(Level0BuiltinKernel *MKernel, ze_graph_handle_t &Ker);

  ///
  /// \brief addFinishedBuild adds a finished program/kernel build to the
  ///        member holding builds (if the build was successful), or in case
  ///        of error, updates the program build log and drops the Build
  /// \param [in] Build the build object to process
  ///
  virtual bool addFinishedBuild(Level0BuildBaseUPtr Build) override;

private:
  std::vector<std::string> KernelNames;
  std::vector<unsigned> KernelIDs;
  std::vector<void *> KernelAttrs;

  Level0BuiltinProgramBuildUPtr FinishedBuild;

  std::list<Level0BuiltinKernelSPtr> Kernels;

  /// helper Queue in the destination Context, for initialization of graphs
  ze_command_queue_handle_t QueueH;
  ze_command_list_handle_t ListH;
  bool IsDBK;
};

void getNpuGraphModelsList(std::string &Out, unsigned &NumKernels);
#endif

class Level0BuildBase {
public:
  enum class BuildType { Kernel, Program, JITProgram, BuiltinProgram, Unknown };

  Level0BuildBase(bool S, BuildType T) : BuildSuccessful(S), Type(T) {};
  virtual ~Level0BuildBase() {};

  bool isSuccessful() const { return BuildSuccessful; }
  std::string &&getBuildLog() { return std::move(BuildLog); }
  BuildType getBuildType() { return Type; }

  // run the Build in the provided ZE Context.
  virtual void run(ze_context_handle_t ContextH) = 0;

  bool isEqual(Level0BuildBase *Other) {
    if (Type == Other->Type)
      return false;
    return compareSameClass(Other);
  };

protected:
  /// compares the instances of same subclass
  virtual bool compareSameClass(Level0BuildBase *Other) { return false; }
  bool BuildSuccessful;
  BuildType Type;
  /// build log for failed builds
  std::string BuildLog;
};

///
/// \brief Abstract class for a single build of a program or kernel (to a native
/// binary) with a particular set of specializations
///
class Level0Build : public Level0BuildBase {

public:
  Level0Build(BuildSpecialization S, Level0Program *Prog, BuildType T)
      : Level0BuildBase(false, T), Program(Prog), ModuleH(nullptr), Spec(S) {}
  virtual ~Level0Build();

  Level0Build(Level0Build const &) = delete;
  Level0Build &operator=(Level0Build const &) = delete;
  Level0Build(Level0Build const &&) = delete;
  Level0Build &operator=(Level0Build &&) = delete;

  bool isDebug() const { return Spec.Debug; }
  bool isOptimized() const { return Spec.Optimize; }
  bool isLargeOffset() const { return Spec.LargeOffsets; }
  bool isSmallWG() const { return Spec.SmallWGSize; }

  ze_module_handle_t getModuleHandle() { return ModuleH; }
  BuildSpecialization getSpec() { return Spec; }

  // Level0 specific (TODO make generic virtual)
  /// loads the built Native Binary, in a particular context & device
  virtual bool loadBinary(ze_context_handle_t ContextH,
                          ze_device_handle_t DeviceH);

  virtual bool compareSameClass(Level0BuildBase *Other) override;

protected:
  /// compiled binary in ZE native format
  std::vector<uint8_t> NativeBinary;

  /// assumes this pointer is valid & alive during the whole build duration,
  /// should be OK because CompilationJob keeps a shared_ptr
  /// TODO: HOWEVER we must invalidate it after the build is finished ???
  Level0Program *Program;

  ///  this handle is valid for the *target* (loadBinary) context,
  ///  not the compilation thread's context.
  ze_module_handle_t ModuleH;

  BuildSpecialization Spec;
};

#ifdef ENABLE_NPU
struct Level0BuiltinKernelBuildResult {
  /// VPU binary + Shave(VLIW) binary, in native format
  std::vector<uint8_t> VpuNativeBinary;
  std::vector<uint8_t> ShaveNativeBinary;

  ///  this handle is valid for the *target* (loadBinary) context,
  ///  not the compilation thread's context.
  ze_graph_handle_t GraphHFinal;
  graph_dditable_ext_t *GraphDDITable;

  Level0BuiltinKernelBuildResult(Level0BuiltinKernelBuildResult const &) =
      delete;
  Level0BuiltinKernelBuildResult &
  operator=(Level0BuiltinKernelBuildResult const &) = delete;

  Level0BuiltinKernelBuildResult(Level0BuiltinKernelBuildResult &&O) {
    VpuNativeBinary = std::move(O.VpuNativeBinary);
    ShaveNativeBinary = std::move(O.ShaveNativeBinary);
    GraphHFinal = O.GraphHFinal;
    O.GraphHFinal = nullptr;
    GraphDDITable = O.GraphDDITable;
    O.GraphDDITable = nullptr;
  }
  Level0BuiltinKernelBuildResult &
  operator=(Level0BuiltinKernelBuildResult &&O) {
    VpuNativeBinary = std::move(O.VpuNativeBinary);
    ShaveNativeBinary = std::move(O.ShaveNativeBinary);
    GraphHFinal = O.GraphHFinal;
    O.GraphHFinal = nullptr;
    GraphDDITable = O.GraphDDITable;
    O.GraphDDITable = nullptr;
    return *this;
  }

  Level0BuiltinKernelBuildResult(graph_dditable_ext_t *DDI)
      : GraphHFinal(nullptr), GraphDDITable(DDI) {}

  ~Level0BuiltinKernelBuildResult() {
    if (GraphHFinal) {
      GraphDDITable->pfnDestroy(GraphHFinal);
    }
  }
};

typedef bool (*instantiateModelTemplate_fn)(const void *KernelAttrs,
                                            std::string &ModelXMLInstance,
                                            std::string &BuildFlagsInstance);

struct Level0Model {
  std::string Name;
  unsigned DBK_ID;
  ze_graph_format_t Format;
  std::string NativeBin;
  std::string NativeShaveBin;
  std::string NGraphXml;
  std::string NGraphBin;
  const std::string BuildFlags;
  instantiateModelTemplate_fn instantiateModel;
};

class Level0BuiltinProgramBuild : public Level0BuildBase {
public:
  Level0BuiltinProgramBuild(Level0BuiltinProgram *Prog,
                            graph_dditable_ext_t *DDITable)
      : Level0BuildBase(false, BuildType::BuiltinProgram), Program(Prog),
        GraphDDITable(DDITable), VPUModel("MTL") {}
  virtual ~Level0BuiltinProgramBuild() { KernelBuilds.clear(); }

  Level0BuiltinProgramBuild(Level0BuiltinProgramBuild const &) = delete;
  Level0BuiltinProgramBuild &
  operator=(Level0BuiltinProgramBuild const &) = delete;
  Level0BuiltinProgramBuild(Level0BuiltinProgramBuild const &&) = delete;
  Level0BuiltinProgramBuild &operator=(Level0BuiltinProgramBuild &&) = delete;

  virtual void run(ze_context_handle_t ContextH) override;

  virtual bool compareSameClass(Level0BuildBase *Other) override;

  /// loads the built Native Binary, for each kernel separately,
  /// in a particular context & device (queue is req for initialization)
  bool loadBinaries(ze_context_handle_t ContextH, ze_device_handle_t DeviceH,
                    ze_command_queue_handle_t QueueH,
                    ze_command_list_handle_t ListH);

  ze_graph_handle_t getGraphHandle(std::string KernelName);

private:
  /// loads stored model from filesystem and puts it into out,
  /// optionally calling compileFromXmlBin if the model is in XML format
  bool loadModel(ze_context_handle_t ContextH, ze_device_handle_t DeviceH,
                 const Level0Model *M, const void *KernelAttrs,
                 Level0BuiltinKernelBuildResult &Out);

  /// loads a single Native Binary, in destination context & device
  bool loadBinary(ze_context_handle_t ContextH, ze_device_handle_t DeviceH,
                  ze_command_queue_handle_t QueueH,
                  ze_command_list_handle_t ListH,
                  Level0BuiltinKernelBuildResult &Out);

  /// compiles a graph from XML+Bin to Native VPU binary
  bool compileFromXmlBin(ze_context_handle_t ContextH,
                         ze_device_handle_t DeviceH,
                         const std::vector<uint8_t> &ModelXml,
                         const std::vector<uint8_t> &ModelBin,
                         const std::string &BuildFlags,
                         std::string ProgCachePath, std::string ProgNativeDir,
                         Level0BuiltinKernelBuildResult &Out);

  std::map<std::string, Level0BuiltinKernelBuildResult> KernelBuilds;

  /// assumes this pointer is valid & alive during the whole build duration,
  /// should be OK because CompilationJob keeps a shared_ptr
  Level0BuiltinProgram *Program;

  graph_dditable_ext_t *GraphDDITable;

  // TODO set this to something
  std::string VPUModel;
};
#endif

///
/// \brief a single build of a SPIRV program to a native ZE binary
///        with a particular set of specializations
///
class Level0ProgramBuild : public Level0Build {

public:
  Level0ProgramBuild(BuildSpecialization Spec, Level0Program *Prog)
      : Level0Build(Spec, Prog, BuildType::Program) {}
  ~Level0ProgramBuild() override {}

  Level0ProgramBuild(Level0ProgramBuild const &) = delete;
  Level0ProgramBuild &operator=(Level0ProgramBuild const &) = delete;
  Level0ProgramBuild(Level0ProgramBuild const &&) = delete;
  Level0ProgramBuild &operator=(Level0ProgramBuild &&) = delete;

  virtual void run(ze_context_handle_t ContextH) override;
};

///
/// \brief a single JIT build of a SPIRV program to a native ZE binary
///        with a particular set of specializations.
///        In this case, the native binary holds the program-scope variables
///
class Level0JITProgramBuild : public Level0ProgramBuild {

public:
  Level0JITProgramBuild(BuildSpecialization Spec, Level0Program *Prog)
      : Level0ProgramBuild(Spec, Prog) {
    Type = BuildType::JITProgram;
  }
  ~Level0JITProgramBuild() override {}

  Level0JITProgramBuild(Level0JITProgramBuild const &) = delete;
  Level0JITProgramBuild &operator=(Level0JITProgramBuild const &) = delete;
  Level0JITProgramBuild(Level0JITProgramBuild const &&) = delete;
  Level0JITProgramBuild &operator=(Level0JITProgramBuild &&) = delete;

  virtual void run(ze_context_handle_t ContextH) override;
};

///
/// \brief a single build of a SPIRV kernel to a native ZE binary
///        with a particular set of specializations (for JIT builds)
///
class Level0KernelBuild : public Level0Build {

public:
  Level0KernelBuild(BuildSpecialization Spec, std::string KernelName,
                    std::string CacheUUID, Level0Program *Prog)
      : Level0Build(Spec, Prog, BuildType::Kernel), KernelName(KernelName),
        KernelCacheUUID(CacheUUID), LinkinModuleH(nullptr) {}
  ~Level0KernelBuild() override {}

  Level0KernelBuild(Level0KernelBuild const &) = delete;
  Level0KernelBuild &operator=(Level0KernelBuild const &) = delete;
  Level0KernelBuild(Level0KernelBuild const &&) = delete;
  Level0KernelBuild &operator=(Level0KernelBuild &&) = delete;

  const std::string getKernelName() { return KernelName; }
  virtual void run(ze_context_handle_t ContextH) override;

  virtual bool loadBinary(ze_context_handle_t ContextH,
                          ze_device_handle_t DeviceH) override;

  virtual bool compareSameClass(Level0BuildBase *Other) override;

private:
  std::string KernelName;
  std::string KernelCacheUUID;
  ze_module_handle_t LinkinModuleH;
};

///
/// \brief A compilation job for the background compiler threads
///
/// The Level0Build instance will be moved into Program once build is finished.
///
///
class Level0CompilationJob {

public:
  Level0CompilationJob(bool HiPrio, Level0ProgramBaseSPtr Prog,
                       Level0BuildBaseUPtr BuildPtr)
      : Build(std::move(BuildPtr)), Program(Prog), HighPrio(HiPrio),
        Finished(false), Successful(false) {}
  ~Level0CompilationJob() = default;

  Level0CompilationJob(Level0CompilationJob const &) = delete;
  Level0CompilationJob& operator=(Level0CompilationJob const &) = delete;
  Level0CompilationJob(Level0CompilationJob const &&) = delete;
  Level0CompilationJob& operator=(Level0CompilationJob &&) = delete;

  bool isHighPrio() const { return HighPrio; }
  // for cancel_builds_for_program
  bool isForProgram(Level0ProgramBase *Prog) const {
    return Program.get() == Prog;
  }
  bool isForDevice(ze_device_handle_t Dev) const {
    return Program->getDevice() == Dev;
  }
  bool isForBuild(Level0BuildBase *B) const { return Build->isEqual(B); }
  // for preferred device comparison
  ze_device_handle_t getDevice() { return Program->getDevice(); }
  void signalFinished();
  bool isSuccessful() const { return Successful; }
  void waitForFinish();

  void compile(Level0CompilerThread *CThread);

private:
  std::mutex Mutex;
  std::condition_variable Cond;

  /// Level0Build instance
  Level0BuildBaseUPtr Build;

  /// needed to hand over the build when it's finished,
  /// and also to ensure Program is not freed while
  /// any build is in progress
  Level0ProgramBaseSPtr Program;

  bool HighPrio;
  bool Finished;
  bool Successful;
};


///
/// \brief A compilation job queue for the background compiler threads
///
/// A queue of compilation jobs with some extra features,
/// like priorities & cancellation of waiting jobs for a particular program
/// (for when a program is rebuilt).
///
///

class Level0CompilerJobQueue {
public:
  Level0CompilerJobQueue() = default;
  ~Level0CompilerJobQueue() = default;

  Level0CompilerJobQueue(Level0CompilerJobQueue const &) = delete;
  Level0CompilerJobQueue& operator=(Level0CompilerJobQueue const &) = delete;
  Level0CompilerJobQueue(Level0CompilerJobQueue const &&) = delete;
  Level0CompilerJobQueue& operator=(Level0CompilerJobQueue &&) = delete;

  /// push job into queue
  void pushWork(Level0CompilationJobSPtr Job);
  ///
  /// \brief returns a job waiting to be compiled. Order of search:
  ///        1) high-priority job for PreferredDevice;
  ///        2) any high-priority job;
  ///        3) low-priority job for PreferredDevice;
  ///        4) any low-priority job.
  /// \param [in] PreferredDevice the device to prefer
  /// \param [out] ShouldExit true if ExitRequested==true
  /// \returns nullptr if ShouldExit==true, otherwise blocks
  ///
  Level0CompilationJobSPtr getWorkOrWait(ze_device_handle_t PreferredDevice,
                                         bool &ShouldExit);

  void finishedWork(Level0CompilationJob *Job);

  Level0CompilationJobSPtr findOrCreateWork(bool HiPrio,
                                            Level0ProgramBaseSPtr Prog,
                                            Level0BuildBaseUPtr BuildU);
  /// cancels jobs for a program which are *not* yet running
  void cancelAllJobsForProgram(Level0ProgramBase *Program);
  /// cancel all jobs & signal an exit
  void clearAndExit();

private:
  std::list<Level0CompilationJobSPtr> LowPrioJobs;
  std::list<Level0CompilationJobSPtr> HighPrioJobs;
  std::list<Level0CompilationJobSPtr> InProgressJobs;
  std::mutex Mutex;
  std::condition_variable Cond;
  /// signal to compilationThreads that program is exiting
  bool ExitRequested = false;

  static Level0CompilationJobSPtr
      findJob(std::list<Level0CompilationJobSPtr> &Queue,
              ze_device_handle_t PreferredDevice);

  static Level0CompilationJobSPtr
  findJob2(std::list<Level0CompilationJobSPtr> &Queue, Level0ProgramBase *Prog,
           Level0BuildBase *Build);

  /// push job into queue, without acquiring mutex
  void pushWorkUnlocked(Level0CompilationJobSPtr Job);
};

///
/// \brief A background compiler thread
///
/// A single CPU thread with its own ZE context,
/// that picks up jobs from Level0CompilerJobQueue & compiles them.
/// Since ze_module_handle cannot be shared between contexts, but
/// native binaries can, native binaries are stored in Level0Build instances.
///
///
class Level0CompilerThread {

public:
  Level0CompilerThread(Level0CompilerJobQueue *Queue,
                       ze_device_handle_t PrefDev, ze_driver_handle_t Drv)
      : DriverH(Drv), PreferredDeviceH(PrefDev), JobQueue(Queue),
        ThreadContextH(nullptr) {}
  ~Level0CompilerThread();

  Level0CompilerThread(Level0CompilerThread const &) = delete;
  Level0CompilerThread& operator=(Level0CompilerThread const &) = delete;
  Level0CompilerThread(Level0CompilerThread const &&) = delete;
  Level0CompilerThread& operator=(Level0CompilerThread &&) = delete;

  ze_context_handle_t getContextHandle() { return ThreadContextH; }

  bool init();

private:
  ze_driver_handle_t DriverH;
  /// The thread will prefer jobs for this device. This is to avoid thread starvation.
  ze_device_handle_t PreferredDeviceH;
  std::thread Thread;
  Level0CompilerJobQueue* JobQueue;

  /// context specific to this thread
  ze_context_handle_t ThreadContextH;

  /// std::thread runs this method;
  /// infinte loop for picking jobs from JobQueue
  void run();
};

///
/// \brief A compilation job scheduler
///
/// main interface to the background compilation system. Owns a single
/// job queue, and manages NCPUTHREADS of background compilation threads.
///
///

class Level0CompilationJobScheduler {

public:
  Level0CompilationJobScheduler() = default;
  ~Level0CompilationJobScheduler();

  Level0CompilationJobScheduler(Level0CompilationJobScheduler const &) = delete;
  Level0CompilationJobScheduler& operator=(Level0CompilationJobScheduler const &) = delete;
  Level0CompilationJobScheduler(Level0CompilationJobScheduler const &&) = delete;
  Level0CompilationJobScheduler& operator=(Level0CompilationJobScheduler &&) = delete;

  bool init(ze_driver_handle_t H, std::vector<ze_device_handle_t> &DevicesH);

  Level0Program *createProgram(ze_context_handle_t Ctx, ze_device_handle_t Dev,
                               bool EnableJIT, std::string &BuildLog,
                               bool Optimize, bool DeviceSupports64bitBuffers,
                               uint32_t NumSpecs, uint32_t *SpecIDs,
                               const void **SpecValues, size_t *SpecValSizes,
                               std::vector<uint8_t> &SpvData,
                               std::vector<char> &ProgramBCData,
                               const char *CDir, const std::string &UUID);
  bool releaseProgram(Level0Program *Prog);

  Level0Kernel *createKernel(Level0Program *Prog, const char *Name);

  bool releaseKernel(Level0Program *Prog, Level0Kernel *Kernel);

  bool getBestKernel(Level0Program *Program,
                     Level0Kernel *Kernel,
                     bool MustUseLargeOffsets,
                     unsigned LocalWGSize,
                     ze_module_handle_t &Mod,
                     ze_kernel_handle_t &Ker);

#ifdef ENABLE_NPU
  Level0BuiltinProgram *
  createBuiltinProgram(ze_context_handle_t Ctx, ze_device_handle_t Dev,
                       std::string &BuildLog, size_t num_builtin_kernels,
                       char **builtin_kernel_names, void *builtin_kernel_ids,
                       void **builtin_kernel_attributes, const char *CDir,
                       const std::string &UUID);

  bool releaseBuiltinProgram(Level0BuiltinProgram *Prog);

  Level0BuiltinKernel *createBuiltinKernel(Level0BuiltinProgram *Prog,
                                           const char *Name);

  bool releaseBuiltinKernel(Level0BuiltinProgram *Prog,
                            Level0BuiltinKernel *Kernel);

  bool getBestBuiltinKernel(Level0BuiltinProgram *Program,
                            Level0BuiltinKernel *Kernel,
                            ze_graph_handle_t &Graph);
#endif

private:
  std::mutex ProgramsLock;
  ze_driver_handle_t DriverH = nullptr;
#ifdef ENABLE_NPU
  graph_dditable_ext_t *GraphDDITable;
  ze_graph_profiling_dditable_ext_t *GraphProfDDITable;
#endif
  std::vector<std::unique_ptr<Level0CompilerThread>> CompilerThreads;
  std::unique_ptr<Level0CompilerJobQueue> JobQueue;
  /// list of all programs that were built & not released yet
  std::list<Level0ProgramSPtr> Programs;
#ifdef ENABLE_NPU
  /// list of all builtin programs
  std::list<Level0BuiltinProgramSPtr> BuiltinPrograms;
  Level0BuiltinProgramSPtr findProgram(Level0BuiltinProgram *Prog);
#endif

  Level0ProgramSPtr findProgram(Level0Program *Prog);
  template <class P, class SPtr>
  SPtr findProgram(P *Prog, std::list<SPtr> &List, bool erase = false);

  void addCompilationJob(Level0CompilationJobSPtr Job);

  bool createAndWaitKernelJITBuilds(Level0ProgramSPtr Program,
                                    Level0Kernel *Kernel, bool LargeOffsets,
                                    bool SmallWG);

  ///
  /// \brief Creates a compilation job and waits for it to finish.
  ///
  /// If the DeviceSupports64bitBuffers is true, creates (and waits for) two
  /// jobs, the 2nd with the LargeOffset option. Uses
  ///
  /// \param [in] Program the program to build for
  /// \param [out] BuildLog contains the build log, if the build is a failure
  /// \param [in] DeviceSupports64bitBuffers specialization option
  /// \param [in] Optimize specialization option
  /// \returns nullptr if ShouldExit==true, otherwise blocks
  ///
  bool createProgramBuilds(Level0ProgramSPtr Program, std::string &BuildLog,
                           bool DeviceSupports64bitBuffers,
                           bool Optimize = true);

#ifdef ENABLE_NPU
  bool createAndWaitBuiltinProgramBuilds(Level0BuiltinProgramSPtr Program,
                                         std::string &BuildLog);
#endif

  bool createProgramBuildFullOptions(Level0ProgramSPtr Program,
                                     std::string &BuildLog, bool WaitForFinish,
                                     bool Optimize = true,
                                     bool LargeOffsets = false,
                                     bool SmallWG = false,
                                     bool HighPrio = false);

  /// adds an -O0 build (two if DeviceSupports64bitBuffers) and waits for finish
  bool createProgramBuildsO0(Level0ProgramSPtr &Program,
                             std::string &BuildLog,
                             bool DeviceSupports64bitBuffers) {
    return createProgramBuilds(Program, BuildLog, DeviceSupports64bitBuffers,
                               false);
  }

  /// adds an -O2 build (or two) but does not wait for finish
  bool createProgramBuildsO2(Level0ProgramSPtr &Program,
                             std::string &BuildLog,
                             bool DeviceSupports64bitBuffers) {
    return createProgramBuilds(Program, BuildLog, DeviceSupports64bitBuffers,
                               true);
  }
};

} // namespace pocl

#endif // POCL_LIB_CL_DEVICES_LEVEL0_LEVEL0_COMPILATION_HH
