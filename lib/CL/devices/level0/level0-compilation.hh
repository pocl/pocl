/* level0-compilation.hh - multithreaded compilation for LevelZero Compute API devices.

   Copyright (c) 2022-2023 Michal Babej / Intel Finland Oy

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


#ifndef LEVEL0COMPILATION_HH
#define LEVEL0COMPILATION_HH

#include <ze_api.h>

#include <condition_variable>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <queue>

class Level0ProgramBuild;

class Level0Kernel {
  std::mutex Mutex;
   // map of Builds (as void*) to Kernel handles
  std::map<void*, ze_kernel_handle_t> KernelHandles;
  std::string Name;
  bool createForBuild(Level0ProgramBuild* Build);

public:
  Level0Kernel(const char* N) : Name(N) {}
  ~Level0Kernel();
  // it is necessary for the queue's run function to lock the kernel
  // while setting arguments & appending to command list
  std::mutex &getMutex() { return Mutex; }
  ze_kernel_handle_t getOrCreateForBuild(Level0ProgramBuild* Build);
  ze_kernel_handle_t getAnyCreated();
};


typedef std::unique_ptr<Level0Kernel> Level0KernelUPtr;

class Level0Program;

// one build with a particular set of settings
class Level0ProgramBuild {
  std::vector<uint8_t> NativeBinary;
  std::string BuildLog;
  // assumes this pointer is valid & alive during the whole build duration
  Level0Program *Program;

  // this handle is valid for the *target* (loadBinary) context,
  // not the compilation context.
  ze_module_handle_t ModuleH;
  bool Optimized;
  bool LargeOffsets;
  bool Debug;
  bool BuildSuccessful;

public:
  Level0ProgramBuild(bool Opt, bool LargeOfs, bool Dbg,
                     Level0Program *Prog)
   : Program(Prog), ModuleH(nullptr), Optimized(Opt),
     LargeOffsets(LargeOfs), Debug(Dbg), BuildSuccessful(false) {}
  ~Level0ProgramBuild();
  // loads the built Native Binary, in a particular context & device
  bool loadBinary(ze_context_handle_t Context,
                  ze_device_handle_t Device);
  // builds the program's SPIRV to Native Binary, in a particular context & device
  bool compile(ze_context_handle_t Context,
               ze_device_handle_t Device);
  ze_module_handle_t getHandle() { return ModuleH; }
  bool isSuccessful() { return BuildSuccessful; }
  bool isDebug() { return Debug; }
  bool isOptimized() { return Optimized; }
  bool isLargeOffset() { return LargeOffsets; }
  std::string &&getBuildLog() { return std::move(BuildLog); }
};

typedef std::unique_ptr<Level0ProgramBuild> Level0ProgramBuildUPtr;

class Level0CompilationJobScheduler;

// a set of builds of a particular SPIRV+SpecConst
class Level0Program {
  // all data except Builds, Kernels & BuildLog are const
  std::mutex Mutex;
  // vectors of multiple builds with different set of build settings,
  // shared_ptr because outsiders can hold a build instance & wait for it to finish
  std::list<Level0ProgramBuildUPtr> Builds;
  std::list<Level0KernelUPtr> Kernels;
  std::string BuildLog;

  ze_context_handle_t ContextH;
  ze_device_handle_t DeviceH;
  std::string CacheDir;
  std::string CacheUUID;

  // SPIR-V binary data
  std::vector<uint8_t> SPIRV;
  // spec constants used for builds
  ze_module_constants_t SpecConstants;
  // storage for actual data of ze_module_constants_t
  std::vector<uint32_t> ConstantIds;
  std::vector<const void*> ConstantVoidPtrs;
  std::vector<std::vector<uint8_t>> ConstantValues;


  void setupSpecConsts(uint32_t NumSpecs, uint32_t* SpecIDs,
                      const void **SpecValues, size_t *SpecValSizes);

public:
  Level0Program(ze_context_handle_t Ctx,
                ze_device_handle_t Dev,
                uint32_t NumSpecs, uint32_t *SpecIDs,
                const void **SpecValues, size_t *SpecValSizes,
                std::vector<uint8_t> &SpvData,
                const char* CDir,
                const std::string &UUID);
  ~Level0Program();

  void addFinishedBuild(Level0ProgramBuildUPtr Build);

  std::vector<uint8_t>& getSPIRV() { return SPIRV; }
  ze_module_constants_t getSpecConstants() { return SpecConstants; }
  ze_device_handle_t getDevice() { return DeviceH; }
  ze_module_handle_t getAnyHandle(); // for setup_metadata
  Level0Kernel *createKernel(const char* Name);
  bool getBestKernel(Level0Kernel *Kernel, bool LargeOffset,
                        ze_module_handle_t &Mod, ze_kernel_handle_t &Ker);
  bool releaseKernel(Level0Kernel *Kernel);
  std::string &&getBuildLog() { return std::move(BuildLog); }
  const std::string &getCacheDir() { return CacheDir; }
  const std::string &getCacheUUID() { return CacheUUID; }
};

typedef std::shared_ptr<Level0Program> Level0ProgramSPtr;

class Level0CompilationJob {
  bool HighPrio;
  bool Finished;
  bool Successful;
  Level0ProgramSPtr Program;
  Level0ProgramBuildUPtr Build;

  std::mutex Mutex;
  std::condition_variable Cond;

public:
  Level0CompilationJob(bool HiP, Level0ProgramSPtr P, Level0ProgramBuildUPtr PB)
    : HighPrio(HiP), Finished(false), Successful(false),
      Program(P), Build(std::move(PB)) {}
  ~Level0CompilationJob() = default;
  bool isHighPrio() { return HighPrio; }
  ze_device_handle_t getDevice() { return Program->getDevice(); }
  Level0ProgramBuild* getBuild() { return Build.get(); }
  bool isForProgram(Level0Program *P) { return Program.get() == P; }
  void signalFinished();
  bool isSuccessful() { return Successful; }
  void waitForFinish();
};

typedef std::shared_ptr<Level0CompilationJob> Level0CompilationJobSPtr;

/* A queue of compilation jobs with some extra features,
 * like high-priority jobs & cancellation of existing requests
 * (for when a program is rebuilt) */
class Level0CompilerJobQueue {
  std::list<Level0CompilationJobSPtr> LowPrioJobs;
  std::list<Level0CompilationJobSPtr> HighPrioJobs;
  std::mutex Mutex;
  std::condition_variable Cond;
  bool ExitRequested;

  Level0CompilationJobSPtr findJob(std::list<Level0CompilationJobSPtr> &Queue,
                                ze_device_handle_t PreferredDevice);

public:
  Level0CompilerJobQueue() = default;
  ~Level0CompilerJobQueue() = default;
  void pushWork(Level0CompilationJobSPtr Job);
  Level0CompilationJobSPtr getWorkOrWait(ze_device_handle_t PreferredDevice,
                                         bool &ShouldExit);
  // cancels jobs for a program which are *not* yet running
  void cancelAllJobsFor(Level0Program *Program);
  void clearAndExit();
};


/* A single CPU thread with its own context that picks up jobs from
 * Level0CompilerJobQueue & compiles them */
class Level0CompilerThread {
  ze_driver_handle_t DriverH;
  ze_device_handle_t PreferredDeviceH;
  std::thread Thread;
  Level0CompilerJobQueue* JobQueue;
  ze_context_handle_t ThreadContextH;
  void run();
  void compileJob(Level0CompilationJobSPtr Job);

public:
  Level0CompilerThread(Level0CompilerJobQueue* Queue,
                       ze_device_handle_t PrefDev,
                       ze_driver_handle_t Drv)
    : DriverH(Drv), PreferredDeviceH(PrefDev), JobQueue(Queue), ThreadContextH(nullptr) { }
  ~Level0CompilerThread();
  bool init();
};

/* main interface to the background compilation system. Owns a single
 * job queue, and manages NCPUTHREADS of compilation threads. */

class Level0CompilationJobScheduler {
  ze_driver_handle_t DriverH = nullptr;
  std::vector<std::unique_ptr<Level0CompilerThread>> CompilerThreads;
  std::unique_ptr<Level0CompilerJobQueue> JobQueue;

  void addCompilationJob(Level0CompilationJobSPtr Job);

public:
  Level0CompilationJobScheduler() = default;
  ~Level0CompilationJobScheduler();
  bool init(ze_driver_handle_t H, std::vector<ze_device_handle_t> &DevicesH);
  void cancelAllJobsFor(Level0Program *Program);

/*
  bool createAndWaitForO0Builds(Level0ProgramSPtr Program,
                                std::string &BuildLog, bool DeviceSupports64bitBuffers);
  void createO2Builds(Level0ProgramSPtr Program, bool DeviceSupports64bitBuffers);
*/
  bool createAndWaitForExactBuilds(Level0ProgramSPtr Program,
                                   std::string &BuildLog,
                                   bool DeviceSupports64bitBuffers,
                                   bool Optimize);
};


#endif // LEVEL0COMPILATION_HH
