/// level0-compilation.cc - multithreaded compilation
/// for LevelZero Compute API devices.
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

#include "level0-compilation.hh"

#include "common.h"
#include "common_driver.h"
#include "devices.h"
#include "pocl_cache.h"
#include "pocl_cl.h"
#include "pocl_file_util.h"
#include "pocl_hash.h"
#include "pocl_llvm.h"
#include "pocl_timing.h"
#include "pocl_util.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstring>
#include <functional>
#include <iomanip>
#include <memory>
#include <sstream>

using namespace pocl;

Level0Kernel::Level0Kernel(const std::string N) : Name(N) {
  SHA1_CTX Ctx;
  uint8_t Digest[SHA1_DIGEST_SIZE];
  pocl_SHA1_Init(&Ctx);
  pocl_SHA1_Update(&Ctx, (const uint8_t *)N.data(), N.size());
  pocl_SHA1_Final(&Ctx, Digest);
  for (unsigned i = 0; i < SHA1_DIGEST_SIZE; i++) {
    char Lo = (Digest[i] & 0x0F) + 65;
    CacheUUID.append(1, Lo);
    char Hi = ((Digest[i] & 0xF0) >> 4) + 65;
    CacheUUID.append(1, Hi);
  }
}

Level0Kernel::~Level0Kernel() {
  for (auto &Pair : KernelHandles) {
    ze_kernel_handle_t Kern = Pair.second;
    ze_result_t Res = zeKernelDestroy(Kern);
    if (Res != ZE_RESULT_SUCCESS) {
      POCL_MSG_ERR("Failed to destroy ZE kernel: %u\n", (unsigned)Res);
    }
  }
}

bool Level0Kernel::createForBuild(BuildSpecialization Spec,
                                  ze_module_handle_t Mod) {
  ze_kernel_handle_t hKernel = nullptr;
  ze_module_handle_t hModule = Mod;
  ze_kernel_desc_t KernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr,
                                 0, // flags
                                 Name.c_str()};
  //  POCL_MSG_WARN("Using kernel name: %s, MODULE: %p\n", Name.c_str(), Mod);
  ze_result_t Res = zeKernelCreate(hModule, &KernelDesc, &hKernel);
  if (Res != ZE_RESULT_SUCCESS) {
    POCL_MSG_ERR("Failed to create ZE kernel: %x\n", (unsigned)Res);
    return false;
  }

  KernelHandles[Spec] = hKernel;
  return true;
}

ze_kernel_handle_t Level0Kernel::getOrCreateForBuild(Level0Build *Build) {
  std::lock_guard<std::mutex> LockGuard(Mutex);
  BuildSpecialization Spec = Build->getSpec();
  ze_module_handle_t Mod = Build->getModuleHandle();
  assert(Mod != nullptr);
  if (KernelHandles.find(Spec) == KernelHandles.end()) {
    bool Res = createForBuild(Spec, Mod);
    if (!Res)
      return nullptr;
  }

  return KernelHandles[Spec];
}

ze_kernel_handle_t Level0Kernel::getAnyCreated() {
  std::lock_guard<std::mutex> LockGuard(Mutex);
  if (KernelHandles.empty()) {
    return nullptr;
  } else {
    return KernelHandles.begin()->second;
  }
}

void Level0Kernel::setIndirectAccess(
    ze_kernel_indirect_access_flag_t AccessFlag, bool Value) {
  std::lock_guard<std::mutex> LockGuard(Mutex);
  if (Value) { // set flag
    switch (AccessFlag) {
    case ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST:
    case ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED:
    case ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE:
      IndirectAccessFlags |= AccessFlag;
      break;
    default:
      break;
    }
  } else { // clear flag
    switch (AccessFlag) {
    case ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST:
    case ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED:
    case ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE:
      IndirectAccessFlags &= (~AccessFlag);
      break;
    default:
      break;
    }
  }
}

void Level0Kernel::setAccessedPointers(const std::map<void *, size_t> &Ptrs) {
  std::lock_guard<std::mutex> LockGuard(Mutex);
  AccessedPointers = Ptrs;
}

Level0Program::~Level0Program() {
  std::lock_guard<std::mutex> LockGuard(Mutex);
  assert(Kernels.empty());
  ProgBuilds.clear();
  KernBuilds.clear();
  JITProgBuilds.clear();
  if (JITCompilation) {
    pocl_llvm_release_context_for_program(ProgramLLVMCtx);
  }
}

Level0Program::Level0Program(ze_context_handle_t Ctx, ze_device_handle_t Dev,
                             bool EnableJIT, bool Optimize, uint32_t NumSpecs,
                             uint32_t *SpecIDs, const void **SpecValues,
                             size_t *SpecValSizes,
                             std::vector<uint8_t> &SpvData,
                             std::vector<char> &ProgramBCData, const char *CDir,
                             const std::string &UUID)
    : ProgramLLVMCtx(nullptr), CacheDir(CDir), CacheUUID(UUID), SPIRV(SpvData),
      ProgramBC(ProgramBCData), ContextH(Ctx), DeviceH(Dev),
      JITCompilation(EnableJIT), Optimize(Optimize) {
  setupSpecConsts(NumSpecs, SpecIDs, SpecValues, SpecValSizes);
}

bool Level0Program::init() {

  // InitializeLLVM();
  if (SPIRV.size() <= 20)
    return false;
  if (CacheUUID.size() <= 10)
    return false;

  if (JITCompilation) {
    if (ProgramBC.size() <= 20)
      return false;
    char *LinkinSpirvContent = nullptr;
    uint64_t LinkinSpirvSize = 0;
    ProgramLLVMCtx = pocl_llvm_create_context_for_program(ProgramBC.data(),
                                                          ProgramBC.size(),
                                                          &LinkinSpirvContent,
                                                          &LinkinSpirvSize);

    if (ProgramLLVMCtx == nullptr || LinkinSpirvSize == 0)
      return false;

    LinkinSPIRV.assign((uint8_t *)LinkinSpirvContent,
                       (uint8_t *)LinkinSpirvContent + LinkinSpirvSize);
    free(LinkinSpirvContent);
  }

  return true;
}

bool Level0Program::addFinishedBuild(Level0BuildUPtr Build) {

  std::lock_guard<std::mutex> LockGuard(Mutex);
  BuildLog.append(Build->getBuildLog());

  if (!Build->isSuccessful() || !Build->loadBinary(ContextH, DeviceH)) {
    POCL_MSG_ERR("build not successful or couldn't load binary\n");
    return false;
  }
  switch (Build->getBuildType()) {
  case Level0Build::BuildType::Kernel: {
    Level0KernelBuildUPtr P(static_cast<Level0KernelBuild *>(Build.release()));
    KernBuilds.push_back(std::move(P));
    return true;
  }
  case Level0Build::BuildType::Program: {
    Level0ProgramBuildUPtr P(
        static_cast<Level0ProgramBuild *>(Build.release()));
    ProgBuilds.push_back(std::move(P));
    return true;
  }
  case Level0Build::BuildType::JITProgram: {
    Level0JITProgramBuildUPtr P(
        static_cast<Level0JITProgramBuild *>(Build.release()));
    JITProgBuilds.push_back(std::move(P));
    return true;
  }
  default:
    assert(0 && "Unknown switch value in addFinishedBuild");
  }
}

template <class Build>
static Build *findBuild(bool MustUseLargeOffsets, bool CanBeSmallWG,
                        std::function<bool(Build *)> SkipMatchF,
                        std::list<std::unique_ptr<Build>> &List) {
  unsigned BestMatchProps = 0, CurrentMatchProps = 0;
  Build *BestRet = nullptr;
  Build *AnyRet = nullptr;

  // properties to match ordered by importance (most to least)
  enum Props { Opt = 0x4, LargeOfs = 0x2, SmallWg = 0x1 };

  for (auto &B : List) {

    if (SkipMatchF(B.get()))
      continue;

    if (MustUseLargeOffsets && !B->isLargeOffset()) {
      continue;
    }
    if (MustUseLargeOffsets == B->isLargeOffset()) {
      CurrentMatchProps |= Props::LargeOfs;
    }

    if (B->isOptimized())
      CurrentMatchProps |= Props::Opt;

    if (B->isSmallWG() && !CanBeSmallWG) {
      continue;
    }
    if (CanBeSmallWG == B->isSmallWG()) {
      CurrentMatchProps |= Props::SmallWg;
    }

    AnyRet = B.get();
    if (CurrentMatchProps > BestMatchProps) {
      BestMatchProps = CurrentMatchProps;
      BestRet = B.get();
    }
  }

  if (BestRet)
    return BestRet;
  else
    return AnyRet;
}

bool Level0Program::getBestKernel(Level0Kernel *Kernel,
                                  bool MustUseLargeOffsets,
                                  bool CanBeSmallWG,
                                  ze_module_handle_t &Mod,
                                  ze_kernel_handle_t &Ker) {
  std::lock_guard<std::mutex> LockGuard(Mutex);
  Level0Build *Build = nullptr;
  std::string KernelName = Kernel->getName();
  if (JITCompilation) {
    Build = findBuild<Level0KernelBuild>(
        MustUseLargeOffsets, CanBeSmallWG,
        [KernelName](Level0KernelBuild *B) {
          return B->getKernelName() != KernelName;
        },
        KernBuilds);
  } else {
    Build = findBuild<Level0ProgramBuild>(
        MustUseLargeOffsets, CanBeSmallWG,
        [](Level0ProgramBuild *B) { return false; }, ProgBuilds);
  }

  if (Build == nullptr) {
    Mod = nullptr;
    Ker = nullptr;
    return false;
  }

  Mod = Build->getModuleHandle();
  Ker = Kernel->getOrCreateForBuild(Build);
  return true;
}

Level0JITProgramBuild *Level0Program::getLinkinBuild(BuildSpecialization Spec) {
  std::lock_guard<std::mutex> LockGuard(Mutex);

  // try to find exact build first
  for (auto &B : JITProgBuilds) {
    if (B->isOptimized() == Spec.Optimize &&
        B->isLargeOffset() == Spec.LargeOffsets &&
        B->isSmallWG() == Spec.SmallWGSize && B->isDebug() == Spec.Debug) {
      return B.get();
    }
  }

  POCL_MSG_WARN("GetLinkinBuild: exact match not found\n");

  // TODO is it OK to return non-exact match ?
  return findBuild<Level0JITProgramBuild>(
      Spec.LargeOffsets, Spec.SmallWGSize,
      [](Level0JITProgramBuild *B) { return false; }, JITProgBuilds);
}

Level0Kernel *Level0Program::createKernel(const std::string Name) {
  std::lock_guard<std::mutex> LockGuard(Mutex);
  Level0KernelSPtr Kernel = std::make_shared<Level0Kernel>(Name);
  Kernels.push_back(Kernel);
  return Kernel.get();
}

bool Level0Program::releaseKernel(Level0Kernel *Kernel) {
  std::lock_guard<std::mutex> LockGuard(Mutex);

  auto Iter = std::find_if(Kernels.begin(), Kernels.end(),
      [&Kernel](Level0KernelSPtr &K) { return K.get() == Kernel; });

  if (Iter == Kernels.end())
    return false;

  Kernels.erase(Iter);
  return true;
}

bool Level0Program::getKernelSPtr(Level0Kernel *Kernel,
                                  Level0KernelSPtr &KernelS) {
  std::lock_guard<std::mutex> LockGuard(Mutex);

  auto Iter = std::find_if(Kernels.begin(), Kernels.end(),
      [&Kernel](Level0KernelSPtr &K) { return K.get() == Kernel; });

  if (Iter == Kernels.end())
    return false;

  KernelS = *Iter;
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
  ConstantVoidPtrs.resize(NumSpecs);
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

bool Level0Program::extractKernelSPIRV(std::string &KernelName,
                                       std::vector<uint8_t> &SPIRV) {
  {
    std::lock_guard<std::mutex> LockGuard(Mutex);
    auto Iter = ExtractedKernelSPIRVCache.find(KernelName);
    if (Iter != ExtractedKernelSPIRVCache.end()) {
      std::vector<uint8_t> &FoundSPIRV = Iter->second;
      SPIRV.insert(SPIRV.end(), FoundSPIRV.begin(), FoundSPIRV.end());
      return true;
    }
  }

  char *SpirvContent = nullptr;
  uint64_t SpirvSize = 0;
  // to avoid having to hold a lock on the Program, use separate BuildLog
  std::string ExtractBuildLog;
  int Res = pocl_llvm_extract_kernel_spirv(ProgramLLVMCtx, KernelName.c_str(),
                                           &ExtractBuildLog, &SpirvContent,
                                           &SpirvSize);

  {
    std::lock_guard<std::mutex> LockGuard(Mutex);
    BuildLog.append(ExtractBuildLog);
    if (Res == 0) {
      SPIRV.insert(SPIRV.end(), SpirvContent, SpirvContent + SpirvSize);
      ExtractedKernelSPIRVCache.emplace(KernelName, SPIRV);
      return true;
    } else {
      POCL_MSG_ERR("pocl_llvm_extract_kernel_spirv FAILED\n");
      return false;
    }
  }
}

Level0Build::~Level0Build() {
  if (ModuleH != nullptr) {
    zeModuleDestroy(ModuleH);
  }
}

static bool loadZeBinary(ze_context_handle_t ContextH,
                         ze_device_handle_t DeviceH,
                         const std::vector<uint8_t> &NativeBinary,
                         bool Finalize, ze_module_handle_t LinkinModuleH,
                         // output vars
                         std::string &BuildLog, ze_module_handle_t &ModuleH) {
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
  ze_module_build_log_handle_t LinkLogH = nullptr;

  ze_result_t ZeRes =
      zeModuleCreate(ContextH, DeviceH, &ModuleDesc, &TempModuleH, &BuildLogH);
  if (ZeRes != ZE_RESULT_SUCCESS) {
    BuildLog.append("zeModuleCreate failed with error: ");
    BuildLog.append(std::to_string(ZeRes));
    BuildLog.append("\n");
    size_t LogSize = 0;
    // should be null terminated.
    zeModuleBuildLogGetString(BuildLogH, &LogSize, nullptr);
    if (LogSize > 0) {
      BuildLog.append("Output of zeModuleCreate:\n");
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
    ModuleH = nullptr;
  } else {
    zeModuleBuildLogDestroy(BuildLogH);
    ModuleH = TempModuleH;
  }

  if (Finalize) {
    if (LinkinModuleH != nullptr) {
      ze_module_handle_t Temp[2] = {LinkinModuleH, ModuleH};
      ZeRes = zeModuleDynamicLink(2, Temp, &LinkLogH);
    } else {
      ZeRes = zeModuleDynamicLink(1, &ModuleH, &LinkLogH);
    }

    if (ZeRes != ZE_RESULT_SUCCESS) {
      BuildLog.append("zeModuleDynamicLink failed with error: ");
      BuildLog.append(std::to_string(ZeRes));
      BuildLog.append("\n");
      size_t LogSize = 0;
      // should be null terminated.
      zeModuleBuildLogGetString(LinkLogH, &LogSize, nullptr);
      if (LogSize > 0) {
        BuildLog.append("Output of zeModuleDynamicLink:\n");
        char *Log = (char *)malloc(LogSize);
        assert(Log);
        zeModuleBuildLogGetString(LinkLogH, &LogSize, Log);
        zeModuleBuildLogDestroy(LinkLogH);
        BuildLog.append(Log);
        free(Log);
      }
    } else {
      zeModuleBuildLogDestroy(LinkLogH);
    }
  }

  POCL_MEASURE_FINISH(load_binary);
  return (ZeRes == ZE_RESULT_SUCCESS);
}

static void getNativeCachePath(Level0Program *Program,
                               BuildSpecialization BSpec,
                               const std::string &KernelCacheUUID,
                               // output vars
                               std::string &BuildFlags,
                               std::string &ProgCachePath,
                               std::string &ProgNativeDir) {
  ProgCachePath = Program->getCacheDir();
  ProgCachePath.append("/native");
  ProgNativeDir = ProgCachePath;
  ProgCachePath.append("/");
  ProgCachePath.append(Program->getCacheUUID());
  ProgCachePath.append("_");
  if (Program->isJITCompiled()) {
    ProgCachePath.append("kernel_");
  } else {
    ProgCachePath.append("program_");
  }
  ProgCachePath.append(KernelCacheUUID);
  ProgCachePath.append("_");

  if (BSpec.Optimize) {
    BuildFlags.append("-ze-opt-level=2");
    ProgCachePath.append("_Opt");
  } else {
    BuildFlags.append("-ze-opt-disable");
    ProgCachePath.append("_NoOpt");
  }

  if (BSpec.LargeOffsets) {
    BuildFlags.append(" -ze-opt-greater-than-4GB-buffer-required");
    ProgCachePath.append("_64bit");
  } else {
    ProgCachePath.append("_32bit");
  }

  if (BSpec.SmallWGSize) {
    BuildFlags.append(" -ze-opt-large-register-file");
    ProgCachePath.append("_smallWG");
  } else {
    ProgCachePath.append("_largeWG");
  }

  if (BSpec.Debug) {
    BuildFlags.append(" -g");
    ProgCachePath.append("_Dbg");
  }

  ProgCachePath.append(".native");
}

static bool findInNativeCache(Level0Program *Program,
                              BuildSpecialization BSpec,
                              const std::string &KernelCacheUUID,
                              // output vars
                              std::string &BuildFlags,
                              std::string &ProgCachePath,
                              std::string &ProgNativeDir,
                              std::vector<uint8_t> &NativeBinary) {
  getNativeCachePath(Program, BSpec, KernelCacheUUID, BuildFlags,
                     ProgCachePath, ProgNativeDir);

  char *Binary = nullptr;
  uint64_t BinarySize = 0;
  if (pocl_exists(ProgCachePath.c_str()) != 0 &&
      pocl_read_file(ProgCachePath.c_str(), &Binary, &BinarySize) == 0) {

    POCL_MSG_PRINT_LEVEL0("Found native binary in cache:  %s \n",
                          ProgCachePath.c_str());
    NativeBinary.insert(NativeBinary.end(), (uint8_t *)Binary,
                        (uint8_t *)(Binary + BinarySize));
    POCL_MEM_FREE(Binary);
    return true;
  } else {
    POCL_MSG_PRINT_LEVEL0("Native binary not found in cache.\n");
  }

  return false;
}

static bool compileSPIRVtoNativeZE(Level0Program *Program,
                                   const std::vector<uint8_t>& SPIRV,
                                   ze_context_handle_t ContextH,
                                   std::string &BuildFlags,
                                   std::string &ProgCachePath,
                                   std::string &ProgNativeDir,
                                   // output vars
                                   std::string &BuildLog,
                                   std::vector<uint8_t> &NativeBinary
                                   ) {
  bool Res = false;
  ze_result_t ZeRes;
  ze_device_handle_t DeviceH = Program->getDevice();
  ze_module_handle_t ModuleH = nullptr;
  ze_module_build_log_handle_t BuildLogH = nullptr;
  size_t NativeSize = 0;
  ze_module_desc_t ModuleDesc;
  ze_module_constants_t SpecConstants;

  POCL_MSG_PRINT_LEVEL0("Compiling & saving into native binary:  %s \n",
                        ProgCachePath.c_str());

  SpecConstants = Program->getSpecConstants();

  ModuleDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                nullptr,
                ZE_MODULE_FORMAT_IL_SPIRV,
                SPIRV.size(),
                SPIRV.data(),
                BuildFlags.c_str(),
                &SpecConstants};

  ZeRes = zeModuleCreate(ContextH, DeviceH, &ModuleDesc, &ModuleH, &BuildLogH);

  if (ZeRes != ZE_RESULT_SUCCESS) {
    BuildLog.append("zeModuleCreate failed with error: ");
    BuildLog.append(std::to_string(ZeRes));
    BuildLog.append("\n");
    size_t LogSize = 0;
    // should be null terminated.
    zeModuleBuildLogGetString(BuildLogH, &LogSize, nullptr);
    if (LogSize > 0) {
      BuildLog.append("Output of zeModuleCreate:\n");
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

  ZeRes = zeModuleGetNativeBinary(ModuleH, &NativeSize, nullptr);
  if (ZeRes != ZE_RESULT_SUCCESS) {
    BuildLog.append("zeModuleGetNativeBinary failed with error: ");
    BuildLog.append(std::to_string(ZeRes));
    BuildLog.append("\nzeModuleGetNativeBinary() failed to return size\n");
    goto FINISH;
  }

  NativeBinary.resize(NativeSize);
  ZeRes = zeModuleGetNativeBinary(ModuleH, &NativeSize, NativeBinary.data());
  if (ZeRes != ZE_RESULT_SUCCESS) {
    BuildLog.append("zeModuleGetNativeBinary failed with error: ");
    BuildLog.append(std::to_string(ZeRes));
    BuildLog.append("\nzeModuleGetNativeBinary() failed to return binary\n");
    goto FINISH;
  }

  pocl_mkdir_p(ProgNativeDir.c_str());
  pocl_write_file(ProgCachePath.c_str(), (char *)NativeBinary.data(),
                  (uint64_t)NativeSize, 0);
  Res = true;

FINISH:
  if (ModuleH != nullptr) {
    zeModuleDestroy(ModuleH);
  }
  return Res;
}

bool Level0Build::loadBinary(ze_context_handle_t ContextH,
                             ze_device_handle_t DeviceH) {
  return loadZeBinary(ContextH, DeviceH, NativeBinary,
                      (Type != BuildType::JITProgram),
                      nullptr, BuildLog, ModuleH);
}

bool Level0Build::isEqual(Level0Build *Other) {
  if (Type != Other->Type)
    return false;
  if (DeviceH != Other->DeviceH)
    return false;
  if (Program != Other->Program)
    return false;
  if (Spec != Other->Spec)
    return false;
  return true;
}

bool Level0KernelBuild::loadBinary(ze_context_handle_t ContextH,
                                   ze_device_handle_t DeviceH) {
  assert(Type == BuildType::Kernel);
  assert(LinkinModuleH);
  return loadZeBinary(ContextH, DeviceH, NativeBinary,
                      true, LinkinModuleH, BuildLog, ModuleH);
}

bool Level0KernelBuild::isEqual(Level0Build *Other) {
  if (!Level0Build::isEqual(Other))
    return false;

  Level0KernelBuild *OtherK = static_cast<Level0KernelBuild *>(Other);
  if (KernelName != OtherK->KernelName)
    return false;
  if (KernelCacheUUID != OtherK->KernelCacheUUID)
    return false;
  return true;
}

void Level0ProgramBuild::run(ze_context_handle_t ContextH) {

  assert(Program != nullptr);
  POCL_MEASURE_START(compilation);

  std::string BuildFlags;
  std::string ProgCachePath;
  std::string ProgNativeDir;

  POCL_MSG_PRINT_LEVEL0(
      "Measuring Full Program compilation of %s | %s | %s | %s build\n",
      (Spec.Optimize ? "O2" : "O0"), (Spec.LargeOffsets ? "64bit" : "32bit"),
      (Spec.SmallWGSize ? "SmallWG" : "LargeWG"),
      (Spec.Debug ? "Debug" : "NoDebug"));

  assert(Program->isJITCompiled() == false);
  // build the full program
  bool Cached = findInNativeCache(Program, Spec, "", BuildFlags, ProgCachePath,
                                  ProgNativeDir, NativeBinary);

  assert(Program->getSPIRV().size() > 0);
  BuildSuccessful =
      Cached || compileSPIRVtoNativeZE(Program, Program->getSPIRV(), ContextH,
                                       BuildFlags, ProgCachePath, ProgNativeDir,
                                       BuildLog, NativeBinary);

  Program = nullptr;

  POCL_MEASURE_FINISH(compilation);
}

void Level0JITProgramBuild::run(ze_context_handle_t ContextH) {

  assert(Program != nullptr);
  POCL_MEASURE_START(compilation);

  std::string BuildFlags;
  std::string ProgCachePath;
  std::string ProgNativeDir;

  POCL_MSG_PRINT_LEVEL0(
      "Measuring JIT Program compilation of %s | %s | %s | %s build\n",
      (Spec.Optimize ? "O2" : "O0"), (Spec.LargeOffsets ? "64bit" : "32bit"),
      (Spec.SmallWGSize ? "SmallWG" : "LargeWG"),
      (Spec.Debug ? "Debug" : "NoDebug"));

  assert(Program->isJITCompiled() == true);
  // uses an invalid function name to avoid name clash with real kernels
  bool Cached = findInNativeCache(Program, Spec,
                                  "link.in",
                                  BuildFlags, ProgCachePath,
                                  ProgNativeDir, NativeBinary);

  BuildFlags.append(" -take-global-address -library-compilation");

  assert(Program->getLinkinSPIRV().size() > 0);
  BuildSuccessful =
      Cached || compileSPIRVtoNativeZE(Program, Program->getLinkinSPIRV(),
                                       ContextH, BuildFlags, ProgCachePath,
                                       ProgNativeDir, BuildLog, NativeBinary);
  Program = nullptr;

  POCL_MEASURE_FINISH(compilation);
}

void Level0KernelBuild::run(ze_context_handle_t ContextH) {

  assert(Program != nullptr);
  assert(Program->isJITCompiled() == true);
  POCL_MEASURE_START(compilation);

  std::string BuildFlags;
  std::string ProgCachePath;
  std::string ProgNativeDir;

  std::vector<uint8_t> SPIRV;

  POCL_MSG_PRINT_LEVEL0("Measuring Kernel compilation of %s | %s | %s build\n",
                        (Spec.Optimize ? "O2" : "O0"),
                        (Spec.LargeOffsets ? "64bit" : "32bit"),
                        (Spec.Debug ? "Debug" : "NoDebug"));

  bool Cached = findInNativeCache(Program, Spec, KernelCacheUUID, BuildFlags,
                                  ProgCachePath, ProgNativeDir, NativeBinary);

  BuildSuccessful =
      Cached || (Program->extractKernelSPIRV(KernelName, SPIRV) &&
                 compileSPIRVtoNativeZE(Program, SPIRV, ContextH, BuildFlags,
                                        ProgCachePath, ProgNativeDir, BuildLog,
                                        NativeBinary));

  // TODO assumes the LinkinModuleH (= its build) will be alive
  // during Kernel's loadBinary
  Level0JITProgramBuild *LinkinBuild = Program->getLinkinBuild(Spec);
  assert(LinkinBuild != nullptr);
  LinkinModuleH = LinkinBuild->getModuleHandle();
  assert(LinkinModuleH != nullptr);

  Program = nullptr;

  POCL_MEASURE_FINISH(compilation);
}

void Level0CompilationJob::signalFinished() {
  std::lock_guard<std::mutex> LockGuard(Mutex);
  Finished = true;
  // adds only successful builds to program
  Successful =
      Build->isSuccessful() && Program->addFinishedBuild(std::move(Build));
  Cond.notify_one();
}

void Level0CompilationJob::waitForFinish() {
  std::unique_lock<std::mutex> UniqLock(Mutex);
  while (!Finished) {
    Cond.wait(UniqLock);
  }
}

void Level0CompilationJob::compile(Level0CompilerThread *CThread) {
  ze_context_handle_t ContextH = CThread->getContextHandle();
  Build->run(ContextH);
  signalFinished();
}

void Level0CompilerJobQueue::pushWorkUnlocked(Level0CompilationJobSPtr Job) {
  if (Job->isHighPrio()) {
    HighPrioJobs.push_back(Job);
  } else {
    LowPrioJobs.push_back(Job);
  }
  Cond.notify_all();
}

void Level0CompilerJobQueue::pushWork(Level0CompilationJobSPtr Job) {
  std::unique_lock<std::mutex> UniqLock(Mutex);
  pushWorkUnlocked(Job);
}

Level0CompilationJobSPtr
Level0CompilerJobQueue::findJob(std::list<Level0CompilationJobSPtr> &Queue,
                                ze_device_handle_t PreferredDevice) {

  if (Queue.empty()) {
    return Level0CompilationJobSPtr(nullptr);
  }

  auto Iter = std::find_if(Queue.begin(), Queue.end(),
                           [&PreferredDevice](Level0CompilationJobSPtr &J) {
                             return J->getDevice() == PreferredDevice;
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
    Job = findJob(HighPrioJobs, PreferredDevice);
    if (!Job) {
      Job = findJob(LowPrioJobs, PreferredDevice);
    }
    if (Job) {
      InProgressJobs.push_back(Job);
      UniqLock.unlock();
      return Job;
    }
    ShouldExit = ExitRequested;
    if (!ShouldExit) {
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

void Level0CompilerJobQueue::finishedWork(Level0CompilationJob *Job) {
  std::lock_guard<std::mutex> LockGuard(Mutex);
  auto Iter = std::find_if(
      InProgressJobs.begin(), InProgressJobs.end(),
      [&Job](Level0CompilationJobSPtr &J) { return J.get() == Job; });
  if (Iter == InProgressJobs.end()) {
    POCL_MSG_ERR("In progress job not found\n");
    return;
  }

  InProgressJobs.erase(Iter);
}

Level0CompilationJobSPtr
Level0CompilerJobQueue::findJob2(std::list<Level0CompilationJobSPtr> &Queue,
                                 Level0Program *Prog, Level0Build *Build) {

  if (Queue.empty()) {
    return Level0CompilationJobSPtr(nullptr);
  }

  auto Iter = std::find_if(
      Queue.begin(), Queue.end(), [&Prog, &Build](Level0CompilationJobSPtr &J) {
        return J->isForProgram(Prog) && J->isBuildEqual(Build);
      });

  if (Iter == Queue.end()) {
    return Level0CompilationJobSPtr(nullptr);
  }

  return *Iter;
}

Level0CompilationJobSPtr
Level0CompilerJobQueue::findOrCreateWork(bool HiPrio, Level0ProgramSPtr &ProgS,
                                         Level0BuildUPtr BuildU) {

  std::unique_lock<std::mutex> UniqLock(Mutex);
  Level0Build *Build = BuildU.get();
  Level0Program *Prog = ProgS.get();

  Level0CompilationJobSPtr Res;
  Res = findJob2(InProgressJobs, Prog, Build);
  if (!Res)
    Res = findJob2(HighPrioJobs, Prog, Build);
  if (!Res)
    Res = findJob2(LowPrioJobs, Prog, Build);

  if (!Res) {
    // no in progess job, add new one
    Res = std::make_shared<Level0CompilationJob>(HiPrio, ProgS,
                                                 std::move(BuildU));

    pushWorkUnlocked(Res);
  }

  return Res;
}

void Level0CompilerJobQueue::cancelAllJobsForProgram(Level0Program *Program) {
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
    if (Job) {
      Job->compile(this);
      JobQueue->finishedWork(Job.get());
    }
  } while (!ShouldExit);
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

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

bool Level0CompilationJobScheduler::init(
    ze_driver_handle_t H, std::vector<ze_device_handle_t> &DevicesH) {

  JobQueue = std::make_unique<Level0CompilerJobQueue>();
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

void Level0CompilationJobScheduler::addCompilationJob(
    Level0CompilationJobSPtr Job) {
  JobQueue->pushWork(Job);
}

Level0CompilationJobScheduler::~Level0CompilationJobScheduler() {
  JobQueue->clearAndExit();
  CompilerThreads.clear();
}

Level0Program *Level0CompilationJobScheduler::createProgram(ze_context_handle_t Ctx,
                                                            ze_device_handle_t Dev,
                                                            bool EnableJIT,
                                                            std::string &BuildLog,
                                                            bool Optimize, bool DeviceSupports64bitBuffers,
                                                            uint32_t NumSpecs, uint32_t *SpecIDs,
                                                            const void **SpecValues, size_t *SpecValSizes,
                                                            std::vector<uint8_t> &SpvData,
                                                            std::vector<char> &ProgramBCData,
                                                            const char* CDir,
                                                            const std::string &UUID) {
  Level0ProgramSPtr Prog = std::make_shared<Level0Program>(Ctx, Dev, EnableJIT, Optimize,
                                           NumSpecs, SpecIDs,
                                           SpecValues, SpecValSizes, SpvData,
                                           ProgramBCData, CDir, UUID);
  if (!Prog->init()) {
    BuildLog.append("failed to initialize Level0Program\n");
    return nullptr;
  }

  bool Res =
      createProgramBuilds(Prog, BuildLog, DeviceSupports64bitBuffers, Optimize);
  if (!Res) {
    BuildLog.append("failed to build Level0Program\n");
    return nullptr;
  }

  std::lock_guard<std::mutex> Lock(ProgramsLock);
  Programs.push_back(Prog);
  return Prog.get();
}

bool Level0CompilationJobScheduler::releaseProgram(Level0Program *Prog) {
  JobQueue->cancelAllJobsForProgram(Prog);

  std::lock_guard<std::mutex> Lock(ProgramsLock);

  auto Iter =
      std::find_if(Programs.begin(), Programs.end(),
                   [&Prog](Level0ProgramSPtr &P) { return P.get() == Prog; });

  if (Iter == Programs.end())
    return false;

  Programs.erase(Iter);
  return true;
}

bool Level0CompilationJobScheduler::findProgram(Level0Program *Prog,
                                                Level0ProgramSPtr &Program) {
  std::lock_guard<std::mutex> Lock(ProgramsLock);

  auto Iter =
      std::find_if(Programs.begin(), Programs.end(),
                   [&Prog](Level0ProgramSPtr &P) { return P.get() == Prog; });

  if (Iter == Programs.end())
    return false;

  Program = *Iter;
  return true;
}

Level0Kernel *Level0CompilationJobScheduler::createKernel(Level0Program *Prog,
                                                          const char *Name) {
  Level0ProgramSPtr Program;
  if (!findProgram(Prog, Program)) {
    POCL_MSG_ERR("cannot find a program %p\n", Prog);
    return nullptr;
  }

  Level0Kernel *K = Program->createKernel(Name);

  // prebuild a 32bit small-WG specialization here
  // this might not be necessary but is useful for timing & catching errors
  // early
  if (pocl_get_bool_option("POCL_LEVEL0_JIT_PREBUILD", 0)) {
    if (K && Prog->isJITCompiled()) {
      POCL_MSG_PRINT_LEVEL0("JIT pre-compiling kernel %p %s\n", K,
                            K->getName().c_str());
      bool Res = createAndWaitKernelJITBuilds(Program, K,
                                              false, // 64bit ofs
                                              false);  // small WG size
      if (!Res) {
        const std::string &BL = Program->getBuildLog();
        POCL_MSG_ERR("Building JIT kernel failed with build log:\n%s",
                     BL.c_str());
        Program->releaseKernel(K);
        return nullptr;
      }
    }
  }

  return K;
}

bool Level0CompilationJobScheduler::releaseKernel(Level0Program *Prog,
                                                  Level0Kernel *Kernel) {
  Level0ProgramSPtr Program;
  if (!findProgram(Prog, Program)) {
    POCL_MSG_ERR("cannot find a program %p\n", Prog);
    return false;
  }
  return Program->releaseKernel(Kernel);
}


bool Level0CompilationJobScheduler::createProgramBuilds(Level0ProgramSPtr &Program,
                                                        std::string &BuildLog,
                                                        bool DeviceSupports64bitBuffers,
                                                        bool Optimize) {
  if (!createProgramBuildFullOptions(Program, BuildLog,
                                     true, // wait
                                     Optimize,
                                     false, // LargeOffsets
                                     false, // small WG
                                     true)) // high prio
    return false;

  if (DeviceSupports64bitBuffers &&
      (!createProgramBuildFullOptions(Program, BuildLog,
                                      true, // wait
                                      Optimize,
                                      true,    // LargeOffsets
                                      false,   // small WG
                                      false))) // high prio
    return false;

  return true;
}


bool Level0CompilationJobScheduler::createProgramBuildFullOptions(Level0ProgramSPtr &Program,
                                                                      std::string &BuildLog,
                                                                      bool WaitForFinish,
                                                                      bool Optimize,
                                                                      bool LargeOffsets,
                                                                      bool SmallWG,
                                                                      bool HighPrio) {

  BuildSpecialization Spec = { Optimize, LargeOffsets, false, SmallWG };
  Level0ProgramBuildUPtr ProgBuild;

  if (Program->isJITCompiled())
    ProgBuild = std::make_unique<Level0JITProgramBuild>(Spec, Program.get());
  else
    ProgBuild = std::make_unique<Level0ProgramBuild>(Spec, Program.get());

  Level0CompilationJobSPtr ProgBuildJob =
      std::make_shared<Level0CompilationJob>(HighPrio, Program,
                                             std::move(ProgBuild));

  addCompilationJob(ProgBuildJob);
  if (WaitForFinish) {
    ProgBuildJob->waitForFinish();

    if (!ProgBuildJob->isSuccessful()) {
      BuildLog.append(Program->getBuildLog());
      return false;
    }
  }

  return true;
}

bool Level0CompilationJobScheduler::createAndWaitKernelJITBuilds(Level0ProgramSPtr &Program,
                                                                 Level0Kernel *Kernel,
                                                                 bool LargeOffsets,
                                                                 bool SmallWG) {

  Level0KernelSPtr KernelS;
  bool Res = Program->getKernelSPtr(Kernel, KernelS);
  assert(Res);
  assert(KernelS);

  BuildSpecialization Spec = {Program->isOptimized(), LargeOffsets, false,
                              SmallWG};
  Level0KernelBuildUPtr KernBuild = std::make_unique<Level0KernelBuild>(
      Spec, Kernel->getName(), Kernel->getCacheUUID(), Program.get());
  Level0CompilationJobSPtr KernBuildJob =
      JobQueue->findOrCreateWork(true, Program, std::move(KernBuild));
  KernBuildJob->waitForFinish();
  return KernBuildJob->isSuccessful();
}



bool Level0CompilationJobScheduler::getBestKernel(Level0Program *Prog,
                                  Level0Kernel *Kernel,
                                  bool MustUseLargeOffsets,
                                  unsigned LocalWGSize,
                                  ze_module_handle_t &Mod,
                                  ze_kernel_handle_t &Ker)
{
  Level0ProgramSPtr Program;
  if (!findProgram(Prog, Program)) {
    POCL_MSG_ERR("cannot find a program %p\n", Prog);
    return false;
  }

  // this is optional; if a Large-WG build exists, it's also usable
  bool CanBeSmallWG = LocalWGSize < 32;

  bool Res;
  Res = Program->getBestKernel(Kernel, MustUseLargeOffsets, CanBeSmallWG, Mod,
                               Ker);
  if (!Program->isJITCompiled()) {
    return Res;
  }
  if (!Res) {
    // this can happen because for JIT-enabled programs,
    // the createKernel only precompiles one (small-offset, large-WG)
    // specialization
    Res = createAndWaitKernelJITBuilds(Program, Kernel, MustUseLargeOffsets,
                                       CanBeSmallWG);
    if (!Res) {
      std::string BL = Program->getBuildLog();
      POCL_MSG_ERR("Building JIT kernel failed with build log:\n%s",
                   BL.c_str());
      return false;
    }
    Res = Program->getBestKernel(Kernel, MustUseLargeOffsets, CanBeSmallWG, Mod,
                                 Ker);
  }
  return Res;
}
