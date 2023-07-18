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


#ifndef POCL_LIB_CL_DEVICES_LEVEL0_LEVEL0_COMPILATION_HH
#define POCL_LIB_CL_DEVICES_LEVEL0_LEVEL0_COMPILATION_HH

#include <ze_api.h>

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

class Level0Program;
typedef std::shared_ptr<Level0Program> Level0ProgramSPtr;

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
class Level0Program {

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
  bool init();
  ~Level0Program();

  Level0Program(Level0Program const &) = delete;
  Level0Program& operator=(Level0Program const &) = delete;
  Level0Program(Level0Program const &&) = delete;
  Level0Program& operator=(Level0Program &&) = delete;

  const std::vector<uint8_t> &getSPIRV() const { return SPIRV; }
  const std::vector<uint8_t> &getLinkinSPIRV() const { return LinkinSPIRV; }
  ze_module_constants_t getSpecConstants() const { return SpecConstants; }
  ze_device_handle_t getDevice() { return DeviceH; }
  const std::string &getBuildLog() { return BuildLog; }
  const std::string &getCacheDir() { return CacheDir; }
  const std::string &getCacheUUID() { return CacheUUID; }
  bool isJITCompiled() const { return JITCompilation; }
  bool isOptimized() const { return Optimize; }

  /// DISABLED: used by device->ops->setup_metadata to get kernel metadata
  // ze_module_handle_t getAnyHandle();

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
  bool addFinishedBuild(Level0BuildUPtr Build);

  ///
  /// \brief extractKernelSPIRV extracts the SPIR-V of a single kernel from the
  ///        whole program's SPIR-V. Required for JIT compilation of kernels
  /// \param [in] Kernel the kernel to extract
  /// \param [out] SPIRV variable to hold the extracted SPIR-V
  /// \return true if successful
  ///
  bool extractKernelSPIRV(std::string &KernelName, std::vector<uint8_t> &SPIRV);

  bool getKernelSPtr(Level0Kernel *Kernel, Level0KernelSPtr &KernelS);

  Level0JITProgramBuild *getLinkinBuild(BuildSpecialization Spec);

private:
  /// all data except Builds, Kernels, ExtractedSPIRV,
  /// ProgramLLVMCtx & BuildLog are const
  std::mutex Mutex;
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
  std::string BuildLog;

  ////////////////////////////////////////////////////////////////////////
  /// cl_program's pocl cache dir
  std::string CacheDir;
  /// UUID used to determine compatibility of native binaries in cache
  std::string CacheUUID;

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

  ze_context_handle_t ContextH;
  ze_device_handle_t DeviceH;

  /// true = compile each kernel separately before launch
  /// false = compile the whole program once
  bool JITCompilation;
  bool Optimize;

  /// setup the ze_module_constants_t from cl_program's Spec constant data.
  void setupSpecConsts(uint32_t NumSpecs, const uint32_t *SpecIDs,
                      const void **SpecValues, size_t *SpecValSizes);

  //  Level0Build *findBuild(bool LargeOffset, bool KernelBuilds);
};

///
/// \brief Abstract class for a single build of a program or kernel (to a native
/// binary)
///        with a particular set of specializations
///
class Level0Build {

public:
  Level0Build(BuildSpecialization S, Level0Program *Prog)
      : Program(Prog), ModuleH(nullptr), Spec(S), BuildSuccessful(false),
        Type(BuildType::Unknown) {
    DeviceH = Program->getDevice();
  }
  virtual ~Level0Build();

  enum class BuildType { Kernel, Program, JITProgram, Unknown };

  Level0Build(Level0Build const &) = delete;
  Level0Build &operator=(Level0Build const &) = delete;
  Level0Build(Level0Build const &&) = delete;
  Level0Build &operator=(Level0Build &&) = delete;

  bool isSuccessful() const { return BuildSuccessful; }
  bool isDebug() const { return Spec.Debug; }
  bool isOptimized() const { return Spec.Optimize; }
  bool isLargeOffset() const { return Spec.LargeOffsets; }
  bool isSmallWG() const { return Spec.SmallWGSize; }
  std::string &&getBuildLog() { return std::move(BuildLog); }
  BuildType getBuildType() { return Type; }

  ze_module_handle_t getModuleHandle() { return ModuleH; }
  BuildSpecialization getSpec() { return Spec; }
  ze_device_handle_t getDevice() { return DeviceH; }

  // TODO make generic
  virtual void run(ze_context_handle_t ContextH) = 0;

  // Level0 specific (TODO make generic virtual)
  /// loads the built Native Binary, in a particular context & device
  virtual bool loadBinary(ze_context_handle_t ContextH,
                          ze_device_handle_t DeviceH);

  virtual bool isEqual(Level0Build *Other);

protected:
  /// compiled binary in ZE native format
  std::vector<uint8_t> NativeBinary;
  /// build log for failed builds
  std::string BuildLog;

  /// assumes this pointer is valid & alive during the whole build duration,
  /// should be OK because CompilationJob keeps a shared_ptr
  /// TODO: HOWEVER we must invalidate it after the build is finished ???
  Level0Program *Program;

  ///  this handle is valid for the *target* (loadBinary) context,
  ///  not the compilation thread's context.
  ze_module_handle_t ModuleH;

  ze_device_handle_t DeviceH;

  BuildSpecialization Spec;
  bool BuildSuccessful;
  BuildType Type;
};

///
/// \brief a single build of a SPIRV program to a native ZE binary
///        with a particular set of specializations
///
class Level0ProgramBuild : public Level0Build {

public:
  Level0ProgramBuild(BuildSpecialization Spec, Level0Program *Prog)
      : Level0Build(Spec, Prog) {
    Type = BuildType::Program;
  }
  ~Level0ProgramBuild() override {}

  Level0ProgramBuild(Level0ProgramBuild const &) = delete;
  Level0ProgramBuild &operator=(Level0ProgramBuild const &) = delete;
  Level0ProgramBuild(Level0ProgramBuild const &&) = delete;
  Level0ProgramBuild &operator=(Level0ProgramBuild &&) = delete;

  void run(ze_context_handle_t ContextH) override;
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

  void run(ze_context_handle_t ContextH) override;
};

///
/// \brief a single build of a SPIRV kernel to a native ZE binary
///        with a particular set of specializations (for JIT builds)
///
class Level0KernelBuild : public Level0Build {

public:
  Level0KernelBuild(BuildSpecialization Spec,
                    std::string KernelName,
                    std::string CacheUUID,
                    Level0Program *Prog)
   : Level0Build(Spec, Prog), KernelName(KernelName),
     KernelCacheUUID(CacheUUID), LinkinModuleH(nullptr) {
    Type = BuildType::Kernel;
  }
  ~Level0KernelBuild() override {}

  Level0KernelBuild(Level0KernelBuild const &) = delete;
  Level0KernelBuild &operator=(Level0KernelBuild const &) = delete;
  Level0KernelBuild(Level0KernelBuild const &&) = delete;
  Level0KernelBuild &operator=(Level0KernelBuild &&) = delete;

  const std::string getKernelName() { return KernelName; }
  void run(ze_context_handle_t ContextH) override;

  virtual bool loadBinary(ze_context_handle_t ContextH,
                          ze_device_handle_t DeviceH) override;

  virtual bool isEqual(Level0Build *Other) override;

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
  Level0CompilationJob(bool HiPrio, Level0ProgramSPtr Prog,
                       Level0BuildUPtr BuildPtr)
      : Build(std::move(BuildPtr)), Program(Prog), HighPrio(HiPrio),
        Finished(false), Successful(false) {}
  ~Level0CompilationJob() = default;

  Level0CompilationJob(Level0CompilationJob const &) = delete;
  Level0CompilationJob& operator=(Level0CompilationJob const &) = delete;
  Level0CompilationJob(Level0CompilationJob const &&) = delete;
  Level0CompilationJob& operator=(Level0CompilationJob &&) = delete;

  bool isHighPrio() const { return HighPrio; }
  // for cancel_builds_for_program
  bool isForProgram(Level0Program *Prog) const { return Program.get() == Prog; }
  bool isBuildEqual(Level0Build *B) const { return Build->isEqual(B); }
  // for preferred device comparison
  ze_device_handle_t getDevice() { return Build->getDevice(); }
  void signalFinished();
  bool isSuccessful() const { return Successful; }
  void waitForFinish();

  void compile(Level0CompilerThread *CThread);

private:
  std::mutex Mutex;
  std::condition_variable Cond;

  /// Level0Build instance
  Level0BuildUPtr Build;
  /// needed to hand over the build when it's finished,
  /// and also to ensure Program is not freed while
  /// any build is in progress
  Level0ProgramSPtr Program;

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

  /// push job into queue, without acquiring mutex
  void pushWorkUnlocked(Level0CompilationJobSPtr Job);

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
                                            Level0ProgramSPtr &Prog,
                                            Level0BuildUPtr BuildU);
  /// cancels jobs for a program which are *not* yet running
  void cancelAllJobsForProgram(Level0Program *Program);
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
      findJob2(std::list<Level0CompilationJobSPtr> &Queue,
               Level0Program *Prog,
               Level0Build *Build);
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
  Level0CompilerThread(Level0CompilerJobQueue* Queue,
                       ze_device_handle_t PrefDev,
                       ze_driver_handle_t Drv)
    : DriverH(Drv), PreferredDeviceH(PrefDev), JobQueue(Queue), ThreadContextH(nullptr) { }
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

private:
  ze_driver_handle_t DriverH = nullptr;
  std::vector<std::unique_ptr<Level0CompilerThread>> CompilerThreads;
  std::unique_ptr<Level0CompilerJobQueue> JobQueue;
  /// list of all programs.
  std::list<Level0ProgramSPtr> Programs;
  std::mutex ProgramsLock;

  bool findProgram(Level0Program *Prog, Level0ProgramSPtr &Program);
  void addCompilationJob(Level0CompilationJobSPtr Job);

  bool createAndWaitKernelJITBuilds(Level0ProgramSPtr &Program,
                                    Level0Kernel *Kernel,
                                    bool LargeOffsets,
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
  bool createProgramBuilds(Level0ProgramSPtr &Program,
                           std::string &BuildLog,
                           bool DeviceSupports64bitBuffers,
                           bool Optimize = true);

  bool createProgramBuildFullOptions(Level0ProgramSPtr &Program,
                                     std::string &BuildLog,
                                     bool WaitForFinish,
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
