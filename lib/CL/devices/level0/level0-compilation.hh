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
/// Scheduler ows one Level0CompilerJobQueue and multiple instances of Level0CompilerThread,
/// which pick up jobs (Level0CompilationJob) from the queue.
/// The Level0CompilationJob holds an instance of Level0ProgramBuild (build with specialization),
/// which at the end of compilation is moved into its Level0Program.
///
/// Usage:
///  1) create Level0Program instance
///  2) use one of the createXYZ methods of JobScheduler with the program
///  3) use Program->getAnyHandle() for e.g. device->ops->setup_metadata
///  4) use the Program->getBestKernel() to get the best available Kernel to run
///

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

class Level0ProgramBuild;

///
/// \brief Stores a map of ProgramBuilds to 'ze_kernel_handle_t' handles,
///        for a particular cl_kernel + device (= kernel->data[device_i])
///
/// New handles are dynamically created by getOrCreateForBuild() when they're needed.
///
///
class Level0Kernel {

public:
  Level0Kernel(const char* N) : Name(N) {}
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

  /// for getOrCreateForBuild
  friend class Level0Program;

  void setIndirectAccess(ze_kernel_indirect_access_flag_t AccessFlag,
                         bool Value);
  void setAccessedPointers(const std::map<void *, size_t> &Ptrs);
  ze_kernel_indirect_access_flags_t getIndirectFlags() {
    return IndirectAccessFlags;
  }
  const std::map<void *, size_t> &getAccessedPointers() {
    return AccessedPointers;
  }

private:
  std::mutex Mutex;
  /// map of ProgramBuilds to Kernel handles
  std::map<Level0ProgramBuild*, ze_kernel_handle_t> KernelHandles;
  std::string Name;
  std::map<void *, size_t> AccessedPointers;
  ze_kernel_indirect_access_flags_t IndirectAccessFlags = 0;

  bool createForBuild(Level0ProgramBuild* Build);
  /// returns (or creates a new) ze_kernel_handle_t for a particular ProgramBuild
  ze_kernel_handle_t getOrCreateForBuild(Level0ProgramBuild* Build);
};

typedef std::unique_ptr<Level0Kernel> Level0KernelUPtr;

class Level0Program;


///
/// \brief A single build of a SPIRV program (to a native binary) with a particular set of
///        "specializations" / Level0 build options (e.g. +O2 -Debug +64bit-offsets)
///
/// Current available specializations: Optimization, large (64bit) offsets, Debug info.
///
///
class Level0ProgramBuild {

public:
  Level0ProgramBuild(bool Opt, bool LargeOfs, bool Dbg,
                     Level0Program *Prog)
   : Program(Prog), ModuleH(nullptr), Optimized(Opt),
     LargeOffsets(LargeOfs), Debug(Dbg), BuildSuccessful(false) {}
  ~Level0ProgramBuild();

  Level0ProgramBuild(Level0ProgramBuild const &) = delete;
  Level0ProgramBuild& operator=(Level0ProgramBuild const &) = delete;
  Level0ProgramBuild(Level0ProgramBuild const &&) = delete;
  Level0ProgramBuild& operator=(Level0ProgramBuild &&) = delete;

  /// loads the built Native Binary, in a particular context & device
  bool loadBinary(ze_context_handle_t Context,
                  ze_device_handle_t Device);
  /// builds the program's SPIRV to Native Binary, in a particular context & device
  bool compile(ze_context_handle_t Context,
               ze_device_handle_t Device);
  ze_module_handle_t getHandle() { return ModuleH; }
  bool isSuccessful() const { return BuildSuccessful; }
  bool isDebug() const { return Debug; }
  bool isOptimized() const { return Optimized; }
  bool isLargeOffset() const { return LargeOffsets; }
  std::string &&getBuildLog() { return std::move(BuildLog); }

private:
  /// compiled binary in ZE native format (ELF)
  std::vector<uint8_t> NativeBinary;
  /// build log for failed builds
  std::string BuildLog;
  /// assumes this pointer is valid & alive during the whole build duration
  Level0Program *Program;

  ///  this handle is valid for the *target* (loadBinary) context,
  ///  not the compilation thread's context.
  ze_module_handle_t ModuleH;

  /// specialization flags

  /// true = -ze-opt-level=2, false = -ze-opt-disable
  bool Optimized;
  /// adds -ze-opt-greater-than-4GB-buffer-required;
  /// only meaningful if the device supports 64bit buffers
  bool LargeOffsets;
  /// true = "-g"
  bool Debug;

  bool BuildSuccessful;
};

typedef std::unique_ptr<Level0ProgramBuild> Level0ProgramBuildUPtr;

class Level0CompilationJobScheduler;

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
                uint32_t NumSpecs, uint32_t *SpecIDs,
                const void **SpecValues, size_t *SpecValSizes,
                std::vector<uint8_t> &SpvData,
                const char* CDir,
                const std::string &UUID);
  ~Level0Program();

  Level0Program(Level0Program const &) = delete;
  Level0Program& operator=(Level0Program const &) = delete;
  Level0Program(Level0Program const &&) = delete;
  Level0Program& operator=(Level0Program &&) = delete;

  ///  called by Level0CompilationJob when finished,
  ///  to move Level0ProgramBuild into the program
  void addFinishedBuild(Level0ProgramBuildUPtr Build);

  std::vector<uint8_t>& getSPIRV() { return SPIRV; }
  ze_module_constants_t getSpecConstants() { return SpecConstants; }
  ze_device_handle_t getDevice() { return DeviceH; }
  /// used by device->ops->setup_metadata to get kernel metadata
  ze_module_handle_t getAnyHandle();
  /// for cl_kernel creation device->ops callback
  Level0Kernel *createKernel(const char* Name);
  ///
  /// \brief returns the best available specialization of a Kernel,
  ///        for the given set of specialization options (currently just one).
  /// \param [in] Kernel the Level0Kernel to search for
  /// \param [in] LargeOffset specialization option
  /// \param [out] Mod the ze_module_handle_t of the found specialization, or null
  /// \param [out] Ker the ze_kernel_handle_t of the found specialization, or null
  /// \returns false if can't find any build specialization
  ///
  bool getBestKernel(Level0Kernel *Kernel, bool LargeOffset,
                     ze_module_handle_t &Mod, ze_kernel_handle_t &Ker);
  /// for cl_kernel deletion device->ops callback
  bool releaseKernel(Level0Kernel *Kernel);
  std::string &&getBuildLog() { return std::move(BuildLog); }
  const std::string &getCacheDir() { return CacheDir; }
  const std::string &getCacheUUID() { return CacheUUID; }

private:
  /// all data except Builds, Kernels & BuildLog are const
  std::mutex Mutex;
  /// builds with specializations
  std::list<Level0ProgramBuildUPtr> Builds;
  std::list<Level0KernelUPtr> Kernels;
  std::string BuildLog;

  ze_context_handle_t ContextH;
  ze_device_handle_t DeviceH;
  /// cl_program's pocl cache dir
  std::string CacheDir;
  /// UUID used to determine compatibility of native binaries in cache
  std::string CacheUUID;

  /// SPIR-V binary (= compilation input)
  std::vector<uint8_t> SPIRV;
  /// spec constants used for compilation
  ze_module_constants_t SpecConstants;
  /// storage for actual data of ze_module_constants_t
  std::vector<uint32_t> ConstantIds;
  std::vector<const void*> ConstantVoidPtrs;
  std::vector<std::vector<uint8_t>> ConstantValues;

  /// setup the ze_module_constants_t from cl_program's Spec constant data.
  void setupSpecConsts(uint32_t NumSpecs, const uint32_t *SpecIDs,
                      const void **SpecValues, size_t *SpecValSizes);

};

typedef std::shared_ptr<Level0Program> Level0ProgramSPtr;

///
/// \brief A compilation job for the background compiler threads
///
/// The Leve0ProgramBuild instance will be moved into Program once build is finished.
///
///
class Level0CompilationJob {

public:
  Level0CompilationJob(bool HiPrio, Level0ProgramSPtr Prog, Level0ProgramBuildUPtr ProgB)
    : Program(Prog), Build(std::move(ProgB)),
      HighPrio(HiPrio), Finished(false), Successful(false) {}
  ~Level0CompilationJob() = default;

  Level0CompilationJob(Level0CompilationJob const &) = delete;
  Level0CompilationJob& operator=(Level0CompilationJob const &) = delete;
  Level0CompilationJob(Level0CompilationJob const &&) = delete;
  Level0CompilationJob& operator=(Level0CompilationJob &&) = delete;

  bool isHighPrio() const { return HighPrio; }
  ze_device_handle_t getDevice() { return Program->getDevice(); }
  Level0ProgramBuild* getBuild() { return Build.get(); }
  bool isForProgram(Level0Program *Prog) { return Program.get() == Prog; }
  void signalFinished();
  bool isSuccessful() const { return Successful; }
  void waitForFinish();

private:
  std::mutex Mutex;
  std::condition_variable Cond;

  Level0ProgramSPtr Program;
  Level0ProgramBuildUPtr Build;

  /// true = high priority
  bool HighPrio;
  bool Finished;
  bool Successful;
};

typedef std::shared_ptr<Level0CompilationJob> Level0CompilationJobSPtr;

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
  /// cancels jobs for a program which are *not* yet running
  void cancelAllJobsFor(Level0Program *Program);
  /// cancel all jobs & signal an exit
  void clearAndExit();

private:
  std::list<Level0CompilationJobSPtr> LowPrioJobs;
  std::list<Level0CompilationJobSPtr> HighPrioJobs;
  std::mutex Mutex;
  std::condition_variable Cond;
  /// signal to compilationThreads that program is exiting
  bool ExitRequested = false;

  static Level0CompilationJobSPtr findJob(std::list<Level0CompilationJobSPtr> &Queue,
                                ze_device_handle_t PreferredDevice);

};


///
/// \brief A background compiler thread
///
/// A single CPU thread with its own ZE context,
/// that picks up jobs from Level0CompilerJobQueue & compiles them.
/// Since ze_module_handle cannot be shared between contexts, but
/// native binaries can, native binaries are stored in Level0ProgramBuild instances.
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

  bool init();

private:
  ze_driver_handle_t DriverH;
  /// The thread will prefer jobs for this device. This is to avoid thread starvation.
  ze_device_handle_t PreferredDeviceH;
  std::thread Thread;
  Level0CompilerJobQueue* JobQueue;
  ze_context_handle_t ThreadContextH;
  /// std::thread runs this method
  void run();
  void compileJob(Level0CompilationJobSPtr Job);
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
  /// cancel all unstarted jobs for this Program
  void cancelAllJobsFor(Level0Program *Program);

  ///
  /// \brief Creates a compilation job and waits for it to finish.
  ///
  /// If the DeviceSupports64bitBuffers is true, creates (and waits for) two jobs,
  /// the 2nd with the LargeOffset option.
  ///
  /// \param [in] Program the program to build for
  /// \param [out] BuildLog contains the build log, if the build is a failure
  /// \param [in] DeviceSupports64bitBuffers specialization option
  /// \param [in] Optimize specialization option
  /// \returns nullptr if ShouldExit==true, otherwise blocks
  ///
  bool createAndWaitForExactBuilds(Level0ProgramSPtr Program,
                                   std::string &BuildLog,
                                   bool DeviceSupports64bitBuffers,
                                   bool Optimize);

  /// adds an -O0 build (two if DeviceSupports64bitBuffers) and waits for finish
  bool createAndWaitForO0Builds(Level0ProgramSPtr Program,
                                std::string &BuildLog,
                                bool DeviceSupports64bitBuffers);

  /// adds an -O2 build (or two) but does not wait for finish
  void createO2Builds(Level0ProgramSPtr Program,
                      bool DeviceSupports64bitBuffers);

private:
  ze_driver_handle_t DriverH = nullptr;
  std::vector<std::unique_ptr<Level0CompilerThread>> CompilerThreads;
  std::unique_ptr<Level0CompilerJobQueue> JobQueue;

  void addCompilationJob(Level0CompilationJobSPtr Job);
};

} // namespace pocl

#endif // POCL_LIB_CL_DEVICES_LEVEL0_LEVEL0_COMPILATION_HH
