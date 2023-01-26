/* level0-compilation.cc - multithreaded compilation for LevelZero Compute API devices.

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

#include "level0-compilation.hh"

#include "common.h"
#include "common_driver.h"
#include "devices.h"
#include "pocl_cache.h"
#include "pocl_cl.h"
#include "pocl_file_util.h"
#include "pocl_timing.h"
#include "pocl_util.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <sstream>

using namespace pocl;

Level0Kernel::~Level0Kernel() {
  for (auto &Pair : KernelHandles) {
    ze_kernel_handle_t Kern = Pair.second;
    ze_result_t Res = zeKernelDestroy(Kern);
    assert(Res == ZE_RESULT_SUCCESS);
  }
}

bool Level0Kernel::createForBuild(Level0ProgramBuild *Build) {
  ze_kernel_handle_t hKernel = nullptr;
  ze_module_handle_t hModule = Build->getHandle();
  ze_kernel_desc_t KernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr,
                                 0, // flags
                                 Name.c_str()};
  ze_result_t Res = zeKernelCreate(hModule, &KernelDesc, &hKernel);
  if (Res != ZE_RESULT_SUCCESS) {
    return false;
  }

  KernelHandles[Build] = hKernel;
  return true;
}

ze_kernel_handle_t
Level0Kernel::getOrCreateForBuild(Level0ProgramBuild *Build) {
  std::lock_guard<std::mutex> LockGuard(Mutex);
  if (KernelHandles.find(Build) == KernelHandles.end()) {
    bool Res = createForBuild(Build);
    assert(Res == true);
  }

  return KernelHandles[Build];
}

ze_kernel_handle_t Level0Kernel::getAnyCreated() {
  std::lock_guard<std::mutex> LockGuard(Mutex);
  if (KernelHandles.empty()) {
    return nullptr;
  } else {
    return KernelHandles.begin()->second;
  }
}

bool Level0ProgramBuild::loadBinary(ze_context_handle_t Context,
                                    ze_device_handle_t Device) {
  POCL_MEASURE_START(load_binary);

  ze_module_desc_t ModuleDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                                 nullptr,
                                 ZE_MODULE_FORMAT_NATIVE,
                                 NativeBinary.size(),
                                 NativeBinary.data(),
                                 nullptr,
                                 nullptr}; // spec constants
  ze_module_handle_t TempModuleH = nullptr;
  ze_module_build_log_handle_t BuildLogH = nullptr;

  ze_result_t ZeRes =
      zeModuleCreate(Context, Device, &ModuleDesc, &TempModuleH, &BuildLogH);
  if (ZeRes != ZE_RESULT_SUCCESS) {
    size_t LogSize = 0;
    // should be null terminated.
    zeModuleBuildLogGetString(BuildLogH, &LogSize, nullptr);
    if (LogSize > 0) {
      BuildLog = "Output of zeModuleCreate:\n";
      char *Log = (char *)malloc(LogSize);
      assert(Log);
      zeModuleBuildLogGetString(BuildLogH, &LogSize, Log);
      zeModuleBuildLogDestroy(BuildLogH);
      BuildLog.append(Log);
      free(Log);
    }
    if (TempModuleH != nullptr) {
      zeModuleDestroy(TempModuleH);
    }
    return false;
  } else {
    zeModuleBuildLogDestroy(BuildLogH);
  }

  POCL_MEASURE_FINISH(load_binary);
  ModuleH = TempModuleH;
  return true;
}

Level0ProgramBuild::~Level0ProgramBuild() {
  if (ModuleH != nullptr) {
    zeModuleDestroy(ModuleH);
  }
}

bool Level0ProgramBuild::compile(ze_context_handle_t Context,
                                 ze_device_handle_t Device) {

  POCL_MEASURE_START(compilation);
  bool Res = false;
  ze_result_t ZeRes;
  ze_module_handle_t ModuleH = nullptr;
  ze_module_build_log_handle_t BuildLogH = nullptr;
  ze_module_build_log_handle_t LinkLogH = nullptr;
  size_t NativeSize = 0;
  ze_module_desc_t ModuleDesc;
  std::vector<uint8_t> SPIRV;
  ze_module_constants_t SpecConstants;

  std::string ProgCachePath(Program->getCacheDir());
  ProgCachePath.append("/program_");
  ProgCachePath.append(Program->getCacheUUID());

  std::string BuildFlags;
  if (Optimized) {
    BuildFlags.append("-ze-opt-level=2");
    ProgCachePath.append("_Opt");
  } else {
    BuildFlags = "-ze-opt-disable";
    ProgCachePath.append("_NoOpt");
  }

  if (LargeOffsets) {
    BuildFlags.append(" -ze-opt-greater-than-4GB-buffer-required");
    ProgCachePath.append("_64bit");
  } else {
    ProgCachePath.append("_32bit");
  }

  if (Debug) {
    BuildFlags.append(" -g");
    ProgCachePath.append("_Dbg");
  }

  ProgCachePath.append(".native");

  char *Binary = nullptr;
  uint64_t BinarySize = 0;
  if (pocl_exists(ProgCachePath.c_str()) != 0 &&
      pocl_read_file(ProgCachePath.c_str(), &Binary, &BinarySize) == 0) {

    POCL_MSG_PRINT_LEVEL0("Reading native binary: | %s \n",
                          ProgCachePath.c_str());
    NativeBinary.insert(NativeBinary.end(), (uint8_t *)Binary,
                        (uint8_t *)(Binary + BinarySize));
    POCL_MEM_FREE(Binary);
    Res = true;
    goto FINISH;
  }

  POCL_MSG_PRINT_LEVEL0("Storing native binary: | %s \n",
                        ProgCachePath.c_str());

  SPIRV = Program->getSPIRV();
  SpecConstants = Program->getSpecConstants();

  ModuleDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                nullptr,
                ZE_MODULE_FORMAT_IL_SPIRV,
                SPIRV.size(),
                SPIRV.data(),
                BuildFlags.c_str(),
                &SpecConstants};

  ZeRes = zeModuleCreate(Context, Device, &ModuleDesc, &ModuleH, &BuildLogH);

  if (ZeRes != ZE_RESULT_SUCCESS) {
    size_t LogSize = 0;
    // should be null terminated.
    zeModuleBuildLogGetString(BuildLogH, &LogSize, nullptr);
    if (LogSize > 0) {
      BuildLog = "Output of zeModuleCreate:\n";
      char *Log = (char *)malloc(LogSize);
      assert(Log);
      zeModuleBuildLogGetString(BuildLogH, &LogSize, Log);
      zeModuleBuildLogDestroy(BuildLogH);
      BuildLog.append(Log);
      free(Log);
    }
    goto FINISH;
  } else {
    zeModuleBuildLogDestroy(BuildLogH);
  }

  ZeRes = zeModuleDynamicLink(1, &ModuleH, &LinkLogH);

  if (ZeRes != ZE_RESULT_SUCCESS) {
    size_t LogSize = 0;
    // should be null terminated.
    zeModuleBuildLogGetString(LinkLogH, &LogSize, nullptr);
    if (LogSize > 0) {
      BuildLog = "Output of zeModuleDynamicLink:\n";
      char *Log = (char *)malloc(LogSize);
      assert(Log);
      zeModuleBuildLogGetString(LinkLogH, &LogSize, Log);
      zeModuleBuildLogDestroy(LinkLogH);
      BuildLog.append(Log);
      free(Log);
    }
    goto FINISH;
  } else {
    zeModuleBuildLogDestroy(LinkLogH);
  }

  ZeRes = zeModuleGetNativeBinary(ModuleH, &NativeSize, nullptr);
  if (ZeRes != ZE_RESULT_SUCCESS) {
    BuildLog.append("zeModuleGetNativeBinary() failed to return size\n");
    goto FINISH;
  }

  NativeBinary.resize(NativeSize);
  ZeRes = zeModuleGetNativeBinary(ModuleH, &NativeSize, NativeBinary.data());
  if (ZeRes != ZE_RESULT_SUCCESS) {
    BuildLog.append(
        "zeModuleGetNativeBinary() failed to return native binary\n");
    goto FINISH;
  }

  pocl_write_file(ProgCachePath.c_str(), (char *)NativeBinary.data(),
                  (uint64_t)NativeSize, 0, 1);
  Res = true;

FINISH:
  if (ModuleH != nullptr) {
    zeModuleDestroy(ModuleH);
  }

  POCL_MSG_PRINT_LEVEL0("Measuring compilation of %s\n",
                        (isOptimized() ? "O2" : "O0"));
  POCL_MEASURE_FINISH(compilation);

  BuildSuccessful = Res;
  return Res;
}


Level0Program::~Level0Program() {
  std::lock_guard<std::mutex> LockGuard(Mutex);
  Builds.clear();
}

Level0Program::Level0Program(ze_context_handle_t Ctx, ze_device_handle_t Dev,
                             uint32_t NumSpecs, uint32_t *SpecIDs,
                             const void **SpecValues, size_t *SpecValSizes,
                             std::vector<uint8_t> &SpvData, const char *CDir,
                             const std::string &UUID)
    : ContextH(Ctx), DeviceH(Dev), CacheDir(CDir), CacheUUID(UUID),
      SPIRV(SpvData) {
  assert(SPIRV.size() > 20);
  assert(CacheUUID.size() > 10);
  setupSpecConsts(NumSpecs, SpecIDs, SpecValues, SpecValSizes);
}

void Level0Program::addFinishedBuild(Level0ProgramBuildUPtr Build) {
  std::lock_guard<std::mutex> LockGuard(Mutex);
  if (!Build->isSuccessful()) {
    BuildLog = Build->getBuildLog();
    return;
  }
  if (!Build->loadBinary(ContextH, DeviceH)) {
    BuildLog = Build->getBuildLog();
    return;
  }
  Builds.push_back(std::move(Build));
}

ze_module_handle_t Level0Program::getAnyHandle() {
  std::lock_guard<std::mutex> LockGuard(Mutex);
  if (Builds.empty()) {
    POCL_MSG_WARN("getAnyHandle: no Builds available\n");
    return nullptr;
  }
  return Builds.front()->getHandle();
}

bool Level0Program::getBestKernel(Level0Kernel *Kernel, bool LargeOffset,
                                  ze_module_handle_t &Mod,
                                  ze_kernel_handle_t &Ker) {
  std::lock_guard<std::mutex> LockGuard(Mutex);
  Level0ProgramBuild *Build = nullptr;
  for (auto &B : Builds) {
    if (B->isOptimized() && B->isLargeOffset() == LargeOffset) {
      Build = B.get();
      break;
    }
  }
  if (Build == nullptr) {
    for (auto &B : Builds) {
      if (!B->isOptimized() && B->isLargeOffset() == LargeOffset) {
        Build = B.get();
        break;
      }
    }
  }
  if (Build == nullptr) {
    Mod = nullptr;
    Ker = nullptr;
    return false;
  }

  Mod = Build->getHandle();
  Ker = Kernel->getOrCreateForBuild(Build);
  return true;
}

Level0Kernel *Level0Program::createKernel(const char *Name) {
  std::lock_guard<std::mutex> LockGuard(Mutex);
  Level0KernelUPtr Kernel(new Level0Kernel(Name));
  Level0Kernel *Ret = Kernel.get();
  Kernels.push_back(std::move(Kernel));
  return Ret;
}

bool Level0Program::releaseKernel(Level0Kernel *Kernel) {
  std::lock_guard<std::mutex> LockGuard(Mutex);

  std::list<Level0KernelUPtr>::iterator Iter = std::find_if(
      Kernels.begin(), Kernels.end(),
      [&Kernel](Level0KernelUPtr &K) { return K.get() == Kernel; });

  if (Iter == Kernels.end())
    return false;
  Kernels.erase(Iter);
  return true;
}

void Level0Program::setupSpecConsts(uint32_t NumSpecs, const uint32_t *SpecIDs,
                                    const void **SpecValues,
                                    size_t *SpecValSizes) {
  if (NumSpecs == 0) {
    SpecConstants.numConstants = 0;
    SpecConstants.pConstantIds = nullptr;
    SpecConstants.pConstantValues = nullptr;
    return;
  }

  ConstantIds.resize(NumSpecs);
  ConstantValues.resize(NumSpecs);
  for (uint32_t i = 0; i < NumSpecs; ++i) {
    ConstantIds[i] = SpecIDs[i];
    ConstantValues[i].resize(SpecValSizes[i]);
    std::memcpy(ConstantValues[i].data(), SpecValues[i], SpecValSizes[i]);
    ConstantVoidPtrs[i] = ConstantValues[i].data();
  }

  SpecConstants.numConstants = NumSpecs;
  SpecConstants.pConstantIds = ConstantIds.data();
  SpecConstants.pConstantValues = ConstantVoidPtrs.data();
}


void Level0CompilationJob::signalFinished() {
  std::lock_guard<std::mutex> LockGuard(Mutex);
  Finished = true;
  Successful = Build->isSuccessful();
  // adds both successful & unsuccessful builds
  Program->addFinishedBuild(std::move(Build));
  Cond.notify_one();
}

void Level0CompilationJob::waitForFinish() {
  std::unique_lock<std::mutex> UniqLock(Mutex);
  while (!Finished) {
    Cond.wait(UniqLock);
  }
}


void Level0CompilerJobQueue::pushWork(Level0CompilationJobSPtr Job) {
  std::unique_lock<std::mutex> UniqLock(Mutex);
  if (Job->isHighPrio()) {
    HighPrioJobs.push_back(Job);
  } else {
    LowPrioJobs.push_back(Job);
  }
  Cond.notify_all();
}

Level0CompilationJobSPtr
Level0CompilerJobQueue::findJob(std::list<Level0CompilationJobSPtr> &Queue,
                                ze_device_handle_t PreferredDevice) {

  if (Queue.empty()) {
    return Level0CompilationJobSPtr(nullptr);
  }

  std::list<Level0CompilationJobSPtr>::iterator Iter =
      std::find_if(Queue.begin(), Queue.end(),
                   [&PreferredDevice](Level0CompilationJobSPtr &J) {
                     return J.get()->getDevice() == PreferredDevice;
                   });

  if (Iter == Queue.end()) {
    Iter = Queue.begin();
  }

  Level0CompilationJobSPtr Job(std::move(*Iter));
  Queue.erase(Iter);
  return Job;
}

Level0CompilationJobSPtr
Level0CompilerJobQueue::getWorkOrWait(ze_device_handle_t PreferredDevice,
                                      bool &ShouldExit) {

  Level0CompilationJobSPtr Job(nullptr);
  std::unique_lock<std::mutex> UniqLock(Mutex);
  do {
    ShouldExit = ExitRequested;

    Job = findJob(HighPrioJobs, PreferredDevice);
    if (Job.get() == nullptr) {
      Job = findJob(LowPrioJobs, PreferredDevice);
    }
    if (ShouldExit) {
      break;
    }
    if (Job.get() != nullptr) {
      UniqLock.unlock();
      return Job;
    } else {
      Cond.wait(UniqLock);
    }
  } while (!ShouldExit);
  UniqLock.unlock();
  return nullptr;
}

void Level0CompilerJobQueue::clearAndExit() {
  std::lock_guard<std::mutex> LockGuard(Mutex);
  HighPrioJobs.clear();
  LowPrioJobs.clear();
  ExitRequested = true;
  Cond.notify_all();
}

void Level0CompilerJobQueue::cancelAllJobsFor(Level0Program *Program) {
  std::lock_guard<std::mutex> LockGuard(Mutex);
  HighPrioJobs.remove_if([Program](Level0CompilationJobSPtr &J) {
    return J->isForProgram(Program);
  });
  LowPrioJobs.remove_if([Program](Level0CompilationJobSPtr &J) {
    return J->isForProgram(Program);
  });
}


bool Level0CompilerThread::init() {
  ze_context_desc_t ContextDescription = {};
  ContextDescription.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  ContextDescription.pNext = nullptr;
  ContextDescription.flags = 0;
  ze_result_t ZeRes =
      zeContextCreate(DriverH, &ContextDescription, &ThreadContextH);
  if (ZeRes != ZE_RESULT_SUCCESS) {
    POCL_MSG_ERR("Compiler thread: failed to create L0 Context\n");
    return false;
  }
  Thread = std::thread(&Level0CompilerThread::run, this);
  return true;
}

void Level0CompilerThread::run() {
  bool ShouldExit = false;
  do {
    Level0CompilationJobSPtr Job(nullptr);
    Job = JobQueue->getWorkOrWait(PreferredDeviceH, ShouldExit);
    if (Job.get() != nullptr) {
      compileJob(std::move(Job));
    }
  } while (!ShouldExit);
}

void Level0CompilerThread::compileJob(Level0CompilationJobSPtr Job) {
  ze_device_handle_t DeviceH = Job->getDevice();
  Level0ProgramBuild *Build = Job->getBuild();
  Build->compile(ThreadContextH, DeviceH);
  Job->signalFinished();
}

Level0CompilerThread::~Level0CompilerThread() {
  if (Thread.joinable()) {
    Thread.join();
  }
  if (ThreadContextH != nullptr) {
    ze_result_t Res = zeContextDestroy(ThreadContextH);
    if (Res != ZE_RESULT_SUCCESS) {
      POCL_MSG_ERR("Compiler thread: failed to destroy L0 Context\n");
    }
  }
}


bool Level0CompilationJobScheduler::init(
    ze_driver_handle_t H, std::vector<ze_device_handle_t> &DevicesH) {

  JobQueue.reset(new Level0CompilerJobQueue());
  DriverH = H;
  unsigned NumThreads = std::thread::hardware_concurrency();
  unsigned NumDevices = DevicesH.size();
  assert(NumDevices > 0);
  for (unsigned i = 0; i < NumThreads; ++i) {
    ze_device_handle_t PreferredDeviceH = DevicesH[i % NumDevices];
    CompilerThreads.emplace_back(
        new Level0CompilerThread(JobQueue.get(), PreferredDeviceH, DriverH));
  }
  for (unsigned i = 0; i < NumThreads; ++i) {
    if (!CompilerThreads[i]->init()) {
      POCL_MSG_ERR("Failed to initialize CompilerThread %u\n", i);
      return false;
    }
  }
  return true;
}

void Level0CompilationJobScheduler::cancelAllJobsFor(Level0Program *Program) {
  JobQueue->cancelAllJobsFor(Program);
}

void Level0CompilationJobScheduler::addCompilationJob(
    Level0CompilationJobSPtr Job) {
  JobQueue->pushWork(Job);
}

Level0CompilationJobScheduler::~Level0CompilationJobScheduler() {
  JobQueue->clearAndExit();
  CompilerThreads.clear();
}


bool Level0CompilationJobScheduler::createAndWaitForO0Builds(Level0ProgramSPtr
Program, std::string &BuildLog, bool DeviceSupports64bitBuffers) {

  Level0ProgramBuildUPtr O0SmallOfsBuild(
              new Level0ProgramBuild(false, false, false,
                                     Program.get()));
  Level0CompilationJobSPtr O0SmallOfsBuildJob(
              new Level0CompilationJob(true,
                    Program, std::move(O0SmallOfsBuild)));

  addCompilationJob(O0SmallOfsBuildJob);
  O0SmallOfsBuildJob->waitForFinish();

  if (!O0SmallOfsBuildJob->isSuccessful()) {
    BuildLog.append(Program->getBuildLog());
    return false;
  }

  // TODO submit both & wait for both
  if (DeviceSupports64bitBuffers) {
      Level0ProgramBuildUPtr O0LargeOfsBuild(
                  new Level0ProgramBuild(false, true, false,
                                         Program.get()));
      Level0CompilationJobSPtr O0LargeOfsBuildJob(
                  new Level0CompilationJob(true,
                        Program, std::move(O0LargeOfsBuild)));

      addCompilationJob(O0LargeOfsBuildJob);
      O0LargeOfsBuildJob->waitForFinish();

      if (!O0LargeOfsBuildJob->isSuccessful()) {
        BuildLog.append(Program->getBuildLog());
        return false;
      }
  }

  return true;
}

void Level0CompilationJobScheduler::createO2Builds(Level0ProgramSPtr Program,
                                                   bool DeviceSupports64bitBuffers) {

    Level0ProgramBuildUPtr O2SmallOfsBuild(
                new Level0ProgramBuild( true, false, false, Program.get()));
    Level0CompilationJobSPtr O2SmallOfsBuildJob(
                new Level0CompilationJob(true,
                      Program, std::move(O2SmallOfsBuild)));

    addCompilationJob(O2SmallOfsBuildJob);

    if (DeviceSupports64bitBuffers) {
        Level0ProgramBuildUPtr O2LargeOfsBuild(
                    new Level0ProgramBuild(true, true, false,
                                           Program.get()));
        Level0CompilationJobSPtr O2LargeOfsBuildJob(
                    new Level0CompilationJob(true,
                          Program, std::move(O2LargeOfsBuild)));

        addCompilationJob(O2LargeOfsBuildJob);
    }
}

bool Level0CompilationJobScheduler::createAndWaitForExactBuilds(
    Level0ProgramSPtr Program, std::string &BuildLog,
    bool DeviceSupports64bitBuffers, bool Optimize) {

  Level0ProgramBuildUPtr SmallOfsBuild(
              new Level0ProgramBuild(Optimize, // Opt
                                     false, // largeOfs
                                     false, // Dbg
                                     Program.get()));
  Level0CompilationJobSPtr SmallOfsBuildJob(
              new Level0CompilationJob(true, Program,
                                       std::move(SmallOfsBuild)));

  addCompilationJob(SmallOfsBuildJob);
  SmallOfsBuildJob->waitForFinish();

  if (!SmallOfsBuildJob->isSuccessful()) {
    BuildLog.append(Program->getBuildLog());
    return false;
  }

  // TODO submit both & wait for both
  if (DeviceSupports64bitBuffers) {
    Level0ProgramBuildUPtr LargeOfsBuild(
        new Level0ProgramBuild(Optimize, // Opt
                               true,     // largeOfs
                               false,    // Dbg
                               Program.get()));
    Level0CompilationJobSPtr LargeOfsBuildJob(
        new Level0CompilationJob(true, Program,
                                 std::move(LargeOfsBuild)));

    addCompilationJob(LargeOfsBuildJob);
    LargeOfsBuildJob->waitForFinish();

    if (!LargeOfsBuildJob->isSuccessful()) {
      BuildLog.append(Program->getBuildLog());
      return false;
    }
  }

  return true;
}
