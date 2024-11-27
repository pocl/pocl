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

#ifdef ENABLE_NPU
#include "npu_dbk.h"
#endif

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstring>
#include <functional>
#include <iomanip>
#include <memory>
#include <sstream>

using namespace pocl;

static std::string getStringHash(const uint8_t *Data, size_t Size) {
  std::string Out;
  SHA1_CTX Ctx;
  uint8_t Digest[SHA1_DIGEST_SIZE];
  pocl_SHA1_Init(&Ctx);
  pocl_SHA1_Update(&Ctx, Data, Size);
  pocl_SHA1_Final(&Ctx, Digest);
  for (unsigned i = 0; i < SHA1_DIGEST_SIZE; i++) {
    char Lo = (Digest[i] & 0x0F) + 65;
    Out.append(1, Lo);
    char Hi = ((Digest[i] & 0xF0) >> 4) + 65;
    Out.append(1, Hi);
  }
  return Out;
}

static std::string getStringHash(const std::string &Input) {
  return getStringHash((const uint8_t *)Input.data(), Input.size());
}

Level0Kernel::Level0Kernel(const std::string N) : Name(N) {
  CacheUUID = getStringHash(N);
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
  ze_kernel_handle_t KernelH = nullptr;
  ze_module_handle_t ModuleH = Mod;
  ze_kernel_desc_t KernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr,
                                 0, // flags
                                 Name.c_str()};
  //  POCL_MSG_WARN("Using kernel name: %s, MODULE: %p\n", Name.c_str(), Mod);
  ze_result_t Res = zeKernelCreate(ModuleH, &KernelDesc, &KernelH);
  if (Res != ZE_RESULT_SUCCESS) {
    POCL_MSG_ERR("Failed to create ZE kernel %s: %x\n", Name.c_str(),
                 (unsigned)Res);
    return false;
  }

  KernelHandles[Spec] = KernelH;
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
  Kernels.clear();
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
    : Level0ProgramBase(Ctx, Dev, CDir, UUID), ProgramLLVMCtx(nullptr),
      SPIRV(SpvData), ProgramBC(ProgramBCData), JITCompilation(EnableJIT),
      Optimize(Optimize) {
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

    if (ProgramLLVMCtx == nullptr) {
      POCL_MSG_ERR("Null ProgramLLVMCtx\n");
      return false;
    }
    if (LinkinSpirvSize == 0) {
      POCL_MSG_ERR("Null LinkinSpirvSize\n");
      return false;
    }

    LinkinSPIRV.assign((uint8_t *)LinkinSpirvContent,
                       (uint8_t *)LinkinSpirvContent + LinkinSpirvSize);
    free(LinkinSpirvContent);
  }

  return true;
}

bool Level0Program::addFinishedBuild(Level0BuildBaseUPtr B) {

  std::lock_guard<std::mutex> LockGuard(Mutex);
  BuildLog.append(B->getBuildLog());
  Level0BuildBase::BuildType BT = B->getBuildType();
  assert(BT == Level0BuildBase::BuildType::Kernel ||
         BT == Level0BuildBase::BuildType::JITProgram ||
         BT == Level0BuildBase::BuildType::Program);

  Level0Build *Build = static_cast<Level0Build *>(B.get());
  if (!Build->isSuccessful() || !Build->loadBinary(ContextH, DeviceH)) {
    BuildLog.append("Error: build not successful or "
                    "couldn't load built binary\n");
    return false;
  }
  switch (BT) {
  case Level0BuildBase::BuildType::Kernel: {
    Level0KernelBuildUPtr P(static_cast<Level0KernelBuild *>(B.release()));
    KernBuilds.push_back(std::move(P));
    return true;
  }
  case Level0BuildBase::BuildType::Program: {
    Level0ProgramBuildUPtr P(static_cast<Level0ProgramBuild *>(B.release()));
    ProgBuilds.push_back(std::move(P));
    return true;
  }
  case Level0BuildBase::BuildType::JITProgram: {
    Level0JITProgramBuildUPtr P(
        static_cast<Level0JITProgramBuild *>(B.release()));
    JITProgBuilds.push_back(std::move(P));
    return true;
  }
  default:
    assert(0 && "Unknown switch value in addFinishedBuild");
  }
  assert(!"UNREACHABLE!");
  return false;
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

bool Level0Build::compareSameClass(Level0BuildBase *Other) {
  Level0Build *OtherBuild = static_cast<Level0Build *>(Other);
  if (Program != OtherBuild->Program)
    return false;
  if (Spec != OtherBuild->Spec)
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

bool Level0KernelBuild::compareSameClass(Level0BuildBase *Other) {
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

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_NPU
bool Level0BuiltinProgramBuild::compareSameClass(Level0BuildBase *Other) {
  Level0BuiltinProgramBuild *OtherBuild =
      static_cast<Level0BuiltinProgramBuild *>(Other);
  if (Program != OtherBuild->Program)
    return false;
  return true;
}

static void getModelNativeCachePath(Level0BuiltinProgram *Program,
                                    const std::vector<uint8_t> &ModelXml,
                                    const std::vector<uint8_t> &ModelBin,
                                    const std::string &Name,
                                    const std::string &BuildFlags,
                                    std::string &ProgCachePath,
                                    std::string &ProgNativeDir) {
  ProgCachePath = Program->getCacheDir();
  ProgCachePath.append("/native/");
  ProgCachePath.append(Program->getCacheUUID());
  ProgNativeDir = ProgCachePath;

  ProgCachePath.append("/");
  ProgCachePath.append(Name);
  ProgCachePath.append("_");

  std::vector<uint8_t> Temp;
  Temp.insert(Temp.end(), Name.begin(), Name.end());
  if (!BuildFlags.empty())
    Temp.insert(Temp.end(), BuildFlags.begin(), BuildFlags.end());
  if (ModelXml.size())
    Temp.insert(Temp.end(), ModelXml.begin(), ModelXml.end());
  if (ModelBin.size())
    Temp.insert(Temp.end(), ModelBin.begin(), ModelBin.end());

  ProgCachePath.append(getStringHash(Temp.data(), Temp.size()));

  ProgCachePath.append(".vpu");
}

static bool findModelInNativeCache(
    Level0BuiltinProgram *Program, const std::vector<uint8_t> &ModelXml,
    const std::vector<uint8_t> &ModelBin, const std::string &BuildFlags,
    const std::string &Name, std::string &ProgCachePath,
    std::string &ProgNativeDir, Level0BuiltinKernelBuildResult &Out) {
  // TODO KernelCacheUUID ? NPU driver version ?
  getModelNativeCachePath(Program, ModelXml, ModelBin, Name, BuildFlags,
                          ProgCachePath, ProgNativeDir);

  char *Binary = nullptr;
  uint64_t BinarySize = 0;
  if (pocl_exists(ProgCachePath.c_str()) != 0 &&
      pocl_read_file(ProgCachePath.c_str(), &Binary, &BinarySize) == 0) {

    POCL_MSG_PRINT_LEVEL0("Found native Model binary in cache:  %s \n",
                          ProgCachePath.c_str());
    Out.VpuNativeBinary.insert(Out.VpuNativeBinary.end(), (uint8_t *)Binary,
                               (uint8_t *)(Binary + BinarySize));
    POCL_MEM_FREE(Binary);
    return true;
  } else {
    POCL_MSG_PRINT_LEVEL0("Native binary not found in cache.\n");
  }

  return false;
}

bool Level0BuiltinProgramBuild::loadBinary(
    ze_context_handle_t FinalContextH, ze_device_handle_t FinalDeviceH,
    ze_command_queue_handle_t QueueH, ze_command_list_handle_t ListH,
    Level0BuiltinKernelBuildResult &Out) {
  BuildLog.append("Creating graph");

  ze_activation_kernel_desc_t ActKernelDesc = {};
  if (!Out.ShaveNativeBinary.empty()) {
    ActKernelDesc = {.stype = ZE_STRUCTURE_TYPE_GRAPH_ACTIVATION_KERNEL,
                     .pNext = nullptr,
                     .kernelDataSize = Out.ShaveNativeBinary.size(),
                     .pKernelData = Out.ShaveNativeBinary.data()};
  }

  assert(!Out.VpuNativeBinary.empty());
  assert(BuildSuccessful);
  ze_graph_desc_t GraphDesc = {
      .stype = ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
      .pNext = !Out.ShaveNativeBinary.empty() ? &ActKernelDesc : nullptr,
      .format = ZE_GRAPH_FORMAT_NATIVE,
      .inputSize = Out.VpuNativeBinary.size(),
      .pInput = Out.VpuNativeBinary.data(),
      .pBuildFlags = nullptr};

  ze_graph_handle_t TempH = nullptr;
  ze_result_t Res =
      GraphDDITable->pfnCreate(FinalContextH, FinalDeviceH, &GraphDesc, &TempH);
  assert(Out.GraphHFinal == nullptr);
  Out.GraphHFinal = TempH;
  if (Res != ZE_RESULT_SUCCESS) {
    BuildLog.append("Graph create failed with error: ");
    BuildLog.append(std::to_string(Res));
    BuildLog.append("\n");
    return false;
  } else {
    BuildLog.append("Graph create SUCCESS\n ");
  }

  const uint64_t SyncTimeout = 2'000'000'000; // 2 seconds
  bool Success = (zeCommandListReset(ListH) == ZE_RESULT_SUCCESS);
  Res = GraphDDITable->pfnAppendGraphInitialize(ListH, Out.GraphHFinal, nullptr,
                                                0, nullptr);
  Success = Success && (Res == ZE_RESULT_SUCCESS);
  Success = Success && (zeCommandListClose(ListH) == ZE_RESULT_SUCCESS);
  Success = Success && (zeCommandQueueExecuteCommandLists(
                            QueueH, 1, &ListH, nullptr) == ZE_RESULT_SUCCESS);
  Success = Success && (zeCommandQueueSynchronize(QueueH, SyncTimeout) ==
                        ZE_RESULT_SUCCESS);

  if (Success) {
    POCL_MSG_PRINT_LEVEL0("Graph ready: %p\n", TempH);
    return true;
  } else {
    BuildLog.append("Graph compiled but initialization failed\n");
    return false;
  }
}

bool Level0BuiltinProgramBuild::compileFromXmlBin(
    ze_context_handle_t ContextH, ze_device_handle_t DeviceH,
    const std::vector<uint8_t> &ModelXml, const std::vector<uint8_t> &ModelBin,
    const std::string &BuildFlags, std::string ProgCachePath,
    std::string ProgNativeDir, Level0BuiltinKernelBuildResult &Out) {

  BuildLog.append("Starting BuiltinProgram Graph compilation\n");
  std::vector<uint8_t> Model;

  ze_device_graph_properties_t pDeviceGraphProperties;
  GraphDDITable->pfnDeviceGetGraphProperties(DeviceH, &pDeviceGraphProperties);

  ze_graph_compiler_version_info_t version = {
      .major = pDeviceGraphProperties.compilerVersion.major,
      .minor = pDeviceGraphProperties.compilerVersion.minor};

  uint64_t XmlLen = ModelXml.size();
  if (XmlLen <= 22) { // strlen(<?xml version="1.0"?>)
    BuildLog.append("broken XML file\n");
    return false;
  }
  uint64_t BinLen = ModelBin.size();

  uint32_t NumInputs = 2;
  uint64_t ModelSize = sizeof(version) + sizeof(NumInputs) + sizeof(XmlLen) +
                       XmlLen + sizeof(BinLen) + BinLen;

  Model.resize(ModelSize);

  uint64_t offset = 0;
  memcpy(Model.data(), &version, sizeof(version));
  offset += sizeof(version);

  memcpy(Model.data() + offset, &NumInputs, sizeof(NumInputs));
  offset += sizeof(NumInputs);

  memcpy(Model.data() + offset, &XmlLen, sizeof(XmlLen));
  offset += sizeof(XmlLen);

  memcpy(Model.data() + offset, ModelXml.data(), XmlLen);
  offset += XmlLen;

  memcpy(Model.data() + offset, &BinLen, sizeof(BinLen));
  offset += sizeof(BinLen);

  // Binaries are optional, XML is mandatory
  if (BinLen) {
    memcpy(Model.data() + offset, ModelBin.data(), BinLen);
    offset += BinLen;
  }

  assert(offset == ModelSize);

  ze_graph_desc_t GraphDesc = {.stype = ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                               .pNext = nullptr,
                               .format = ZE_GRAPH_FORMAT_NGRAPH_LITE,
                               .inputSize = Model.size(),
                               .pInput = Model.data(),
                               .pBuildFlags = BuildFlags.c_str()};

  auto Deleter = [this](ze_graph_handle_t ptr) {
    if (ptr)
      this->GraphDDITable->pfnDestroy(ptr);
  };

  std::unique_ptr<std::remove_pointer_t<ze_graph_handle_t>, decltype(Deleter)>
      TempGraphH(nullptr, Deleter);
  ze_graph_handle_t GraphH = nullptr;
  ze_result_t Res =
      GraphDDITable->pfnCreate(ContextH, DeviceH, &GraphDesc, &GraphH);
  TempGraphH.reset(GraphH);

  uint32_t logSize = 0;
  ze_result_t Res2 =
      GraphDDITable->pfnBuildLogGetString(TempGraphH.get(), &logSize, nullptr);
  if (Res2 == ZE_RESULT_SUCCESS && logSize > 0) {
    std::string TempBuildLog;
    TempBuildLog.resize(logSize + 1, 0);
    Res = GraphDDITable->pfnBuildLogGetString(TempGraphH.get(), &logSize,
                                              TempBuildLog.data());
    BuildLog.append(TempBuildLog);
  }

  if (Res != ZE_RESULT_SUCCESS) {
    char Msg[128];
    snprintf(Msg, 128, "BuiltinProgram Graph compilation failed with : %0x\n",
             Res);
    Msg[127] = 0;
    BuildLog.append(Msg);
    return false;
  }

  size_t NativeSize = 0;
  Res =
      GraphDDITable->pfnGetNativeBinary(TempGraphH.get(), &NativeSize, nullptr);
  if (Res != ZE_RESULT_SUCCESS || NativeSize == 0) {
    POCL_MSG_ERR("LevelZero: Failed to get Native binary SIZE for Graph\n");
    BuildLog.append("Failed to get Native binary SIZE for Graph\n");
    return false;
  }

  Out.VpuNativeBinary.resize(NativeSize, 0);
  Res = GraphDDITable->pfnGetNativeBinary(TempGraphH.get(), &NativeSize,
                                          Out.VpuNativeBinary.data());
  if (Res != ZE_RESULT_SUCCESS || NativeSize == 0) {
    BuildLog.append("Failed to get Native binary for Graph\n");
    return false;
  }

  Out.GraphHFinal = nullptr;

  if (pocl_mkdir_p(ProgNativeDir.c_str()) != 0) {
    BuildLog.append("Graph compilation: failed to create cache dir\n");
  }
  if (pocl_write_file(ProgCachePath.c_str(), (char *)Out.VpuNativeBinary.data(),
                      (uint64_t)NativeSize, 0) != 0) {
    BuildLog.append("Graph compilation: failed to write cache file\n");
  }

  BuildLog.append("BuiltinProgram Graph compilation successful\n");
  return true;
}

static void getArgTypeAndSize(ze_graph_argument_properties_t &graphArgProps,
                              size_t &TotalSize, std::string &TypeName) {

  TotalSize = 1;
  for (int i = 0; i < ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE; i++)
    TotalSize *= graphArgProps.dims[i];

  switch (graphArgProps.devicePrecision) {
  case ZE_GRAPH_ARGUMENT_PRECISION_FP64:
  case ZE_GRAPH_ARGUMENT_PRECISION_UINT64:
  case ZE_GRAPH_ARGUMENT_PRECISION_INT64:
    TotalSize *= sizeof(uint64_t);
    break;
  case ZE_GRAPH_ARGUMENT_PRECISION_FP32:
  case ZE_GRAPH_ARGUMENT_PRECISION_INT32:
  case ZE_GRAPH_ARGUMENT_PRECISION_UINT32:
    TotalSize *= sizeof(uint32_t);
    break;
  case ZE_GRAPH_ARGUMENT_PRECISION_BF16:
  case ZE_GRAPH_ARGUMENT_PRECISION_FP16:
  case ZE_GRAPH_ARGUMENT_PRECISION_INT16:
  case ZE_GRAPH_ARGUMENT_PRECISION_UINT16:
    TotalSize *= sizeof(uint16_t);
    break;
  case ZE_GRAPH_ARGUMENT_PRECISION_INT8:
  case ZE_GRAPH_ARGUMENT_PRECISION_UINT8:
    TotalSize *= sizeof(uint8_t);
    break;
  case ZE_GRAPH_ARGUMENT_PRECISION_INT4:
  case ZE_GRAPH_ARGUMENT_PRECISION_UINT4:
    TotalSize /= 2;
    break;
  default:
    POCL_MSG_ERR("Invalid Graph Argument Precision\n");
  }

  switch (graphArgProps.devicePrecision) {
  case ZE_GRAPH_ARGUMENT_PRECISION_FP64:
    TypeName = "double";
    break;
  case ZE_GRAPH_ARGUMENT_PRECISION_UINT64:
    TypeName = "ulong";
    break;
  case ZE_GRAPH_ARGUMENT_PRECISION_INT64:
    TypeName = "long";
    break;
  case ZE_GRAPH_ARGUMENT_PRECISION_FP32:
    TypeName = "float";
    break;
  case ZE_GRAPH_ARGUMENT_PRECISION_INT32:
    TypeName = "int";
    break;
  case ZE_GRAPH_ARGUMENT_PRECISION_UINT32:
    TypeName = "uint";
    break;
  case ZE_GRAPH_ARGUMENT_PRECISION_BF16:
    TypeName = "bfloat16";
    break;
  case ZE_GRAPH_ARGUMENT_PRECISION_FP16:
    TypeName = "half";
    break;
  case ZE_GRAPH_ARGUMENT_PRECISION_INT16:
    TypeName = "short";
    break;
  case ZE_GRAPH_ARGUMENT_PRECISION_UINT16:
    TypeName = "ushort";
    break;
  case ZE_GRAPH_ARGUMENT_PRECISION_INT8:
    TypeName = "char";
    break;
  case ZE_GRAPH_ARGUMENT_PRECISION_UINT8:
    TypeName = "uchar";
    break;
  case ZE_GRAPH_ARGUMENT_PRECISION_INT4:
  case ZE_GRAPH_ARGUMENT_PRECISION_UINT4:
    TypeName = "fourbit";
    break;
    break;
  default:
    POCL_MSG_ERR("Invalid Graph Argument Precision\n");
  }

  assert(TotalSize != 0);
}

constexpr unsigned NumLevel0GraphModels = 3;
static const Level0Model Level0GraphModels[NumLevel0GraphModels] = {
    Level0Model{
        .Name = "pocl.googlenet.v1.fp32",
        .DBK_ID = 0,
        .Format = ZE_GRAPH_FORMAT_NGRAPH_LITE,
        .NGraphXml = "googlenet-v1.xml",
        .NGraphBin = "googlenet-v1.bin",
        .BuildFlags =
            R"RAW(--inputs_precisions="data:U8" --inputs_layouts="data:NCHW"  --outputs_precisions="dot:FP16" --outputs_layouts="dot:NC" --config NPU_PLATFORM="3720" LOG_LEVEL="LOG_DEBUG")RAW",
    },
    Level0Model{.Name = "gemm_exp",
                .DBK_ID = CL_DBK_GEMM_EXP,
                .Format = ZE_GRAPH_FORMAT_NGRAPH_LITE,
                .NGraphXml = "",
                .NGraphBin = "",
                .BuildFlags = "",
                .instantiateModel = instantiateTemplateGEMM},
    Level0Model{.Name = "matmul_exp",
                .DBK_ID = CL_DBK_MATMUL_EXP,
                .Format = ZE_GRAPH_FORMAT_NGRAPH_LITE,
                .NGraphXml = "",
                .NGraphBin = "",
                .BuildFlags = "",
                .instantiateModel = instantiateTemplateMATMUL},
};

// returns semicolon separated list of recognized models (TODO: excluding DBKs
// ?)
void pocl::getNpuGraphModelsList(std::string &Out, unsigned &NumKernels) {
  for (unsigned I = 0; I < NumLevel0GraphModels; ++I) {
    if (I > 0)
      Out.append(";");
    if (Level0GraphModels[I].Name.size())
      Out.append(Level0GraphModels[I].Name);
  }
}

static bool readFile(const std::string &BasePath, const std::string &Filename,
                     std::vector<uint8_t> &Output, std::string &BLog) {
  char *Binary = nullptr;
  uint64_t BinarySize = 0;

  std::string FullPath(BasePath);
  FullPath.append("/");
  FullPath.append(Filename);
  if (pocl_read_file(FullPath.c_str(), &Binary, &BinarySize)) {
    BLog.append("Can't read file: ");
    BLog.append(FullPath.c_str());
    BLog.append("\n");
    return false;
  }
  Output.insert(Output.begin(), Binary, Binary + BinarySize);
  free(Binary);
  return true;
}

// TODO collision-free replacement
void replaceAllStringsInMap(std::string &Buffer, const ReplaceMapT RepMap) {
  for (auto &It : RepMap) {
    const std::string Old(It.first);
    const std::string &New = It.second;
    size_t Pos = std::string::npos;
    size_t Len = Old.size();
    while ((Pos = Buffer.find(Old)) != std::string::npos) {
      Buffer.replace(Pos, Len, New);
    }
  }
}

// converts cl_tensor_datatype to precision metadata
const char *dtype2precision(cl_tensor_datatype_exp dtype) {
  switch (dtype) {
  case CL_TENSOR_DTYPE_FP64_EXP:
    return "FP64";
  case CL_TENSOR_DTYPE_FP32_EXP:
    return "FP32";
  case CL_TENSOR_DTYPE_FP16_EXP:
    return "FP16";
  case CL_TENSOR_DTYPE_INT64_EXP:
    return "INT64";
  case CL_TENSOR_DTYPE_UINT64_EXP:
    return "UINT64";
  case CL_TENSOR_DTYPE_INT32_EXP:
    return "INT32";
  case CL_TENSOR_DTYPE_UINT32_EXP:
    return "UINT32";
  case CL_TENSOR_DTYPE_INT16_EXP:
    return "INT16";
  case CL_TENSOR_DTYPE_UINT16_EXP:
    return "UINT16";
  case CL_TENSOR_DTYPE_INT8_EXP:
    return "INT8";
  case CL_TENSOR_DTYPE_UINT8_EXP:
    return "UINT8";
  case CL_TENSOR_DTYPE_INT4_EXP:
    return "INT4";
  case CL_TENSOR_DTYPE_UINT4_EXP:
    return "UINT4";
  default:
  case CL_TENSOR_DTYPE_FP8E4M3_EXP: // return "F8E4M3";
  case CL_TENSOR_DTYPE_FP8E5M2_EXP: // return "F8E5M2";
  case CL_TENSOR_DTYPE_UNKNOWN:
    return "UNDEFINED";
  }
  return nullptr;
}

// converts cl_tensor_datatype to OpenVINO element tyep
const char *dtype2elemtype(cl_tensor_datatype_exp dtype) {
  switch (dtype) {
  case CL_TENSOR_DTYPE_FP64_EXP:
    return "F64";
  case CL_TENSOR_DTYPE_INT64_EXP:
    return "I64";
  case CL_TENSOR_DTYPE_UINT64_EXP:
    return "U64";
  case CL_TENSOR_DTYPE_FP32_EXP:
    return "F32";
  case CL_TENSOR_DTYPE_INT32_EXP:
    return "I32";
  case CL_TENSOR_DTYPE_UINT32_EXP:
    return "U32";
  case CL_TENSOR_DTYPE_FP16_EXP:
    return "F16";
  case CL_TENSOR_DTYPE_INT16_EXP:
    return "I16";
  case CL_TENSOR_DTYPE_UINT16_EXP:
    return "U16";
  case CL_TENSOR_DTYPE_FP8E4M3_EXP:
    return "F8E4M3";
  case CL_TENSOR_DTYPE_FP8E5M2_EXP:
    return "F8E5M2";
  case CL_TENSOR_DTYPE_INT8_EXP:
    return "I8";
  case CL_TENSOR_DTYPE_UINT8_EXP:
    return "U8";
  case CL_TENSOR_DTYPE_INT4_EXP:
    return "I4";
  case CL_TENSOR_DTYPE_UINT4_EXP:
    return "U4";
  default:
  case CL_TENSOR_DTYPE_UNKNOWN:
    return "UNDEFINED";
  }
  return nullptr;
}

const char *layout2str(cl_tensor_layout_ml_type_exp l) {
  switch (l) {
  case CL_TENSOR_LAYOUT_ML_C_EXP:
    return "C";
  case CL_TENSOR_LAYOUT_ML_NC_EXP:
    return "NC";
  case CL_TENSOR_LAYOUT_ML_CN_EXP:
    return "CN";
  case CL_TENSOR_LAYOUT_ML_HW_EXP:
    return "HW";
  case CL_TENSOR_LAYOUT_ML_WH_EXP:
    return "WH";
  case CL_TENSOR_LAYOUT_ML_CHW_EXP:
    return "CHW";
  case CL_TENSOR_LAYOUT_ML_NCHW_EXP:
    return "NCHW";
  case CL_TENSOR_LAYOUT_ML_NHWC_EXP:
    return "NHWC";
  default:
    return "NULL";
  }
}


/// @brief Loads a native model from disk cache, or builds an XML+BIN model,
/// or builds a DBK template model
bool Level0BuiltinProgramBuild::loadModel(ze_context_handle_t ContextH,
                                          ze_device_handle_t DeviceH,
                                          const Level0Model *M,
                                          const void *KernelAttrs,
                                          Level0BuiltinKernelBuildResult &Out) {
  char ModelDirectoryPath[POCL_MAX_PATHNAME_LENGTH];
  std::string PartialPath("/level0/");
  assert(!VPUModel.empty());
  PartialPath.append(VPUModel);
  PartialPath.append("/");
  PartialPath.append(M->Name);
  pocl_get_srcdir_or_datadir(ModelDirectoryPath, "/lib/CL/devices", "",
                             PartialPath.c_str());

  if (M->Format == ZE_GRAPH_FORMAT_NATIVE) {
    std::vector<uint8_t> NativeBin;
    std::vector<uint8_t> NativeShaveBin;
    BuildLog.append("Loading native Model\n");

    if (!readFile(ModelDirectoryPath, M->NativeBin, NativeBin, BuildLog)) {
      return false;
    }
    if (!M->NativeShaveBin.empty() &&
        !readFile(ModelDirectoryPath, M->NativeShaveBin, NativeShaveBin,
                  BuildLog)) {
      return false;
    }
    Out.GraphHFinal = nullptr;
    Out.ShaveNativeBinary = std::move(NativeShaveBin);
    Out.VpuNativeBinary = std::move(NativeBin);
    BuildLog.append("Native Model loaded\n");
    return true;
  }

  if (M->Format == ZE_GRAPH_FORMAT_NGRAPH_LITE) {
    std::vector<uint8_t> ModelXml;
    std::vector<uint8_t> ModelBin;
    std::string BuildFlags;

    std::string ProgCachePath;
    std::string ProgNativeDir;

    if (M->DBK_ID != 0) {
      BuildLog.append("Creating Model XML & Bin from DBK Template\n");
      std::string ModelXMLInstance;
      std::string BuildFlagsInstance;
      assert(M->instantiateModel);
      if (!M->instantiateModel(KernelAttrs, ModelXMLInstance,
                               BuildFlagsInstance))
        return false;
      BuildLog.append("\n");
      BuildLog.append("@@@ MODEL: \n");
      BuildLog.append(ModelXMLInstance);
      BuildLog.append("\n");
      BuildLog.append("@@@ BUILD FLAGS: \n");
      BuildLog.append(BuildFlagsInstance);
      BuildLog.append("\n");
      ModelXml.insert(ModelXml.end(), ModelXMLInstance.begin(),
                      ModelXMLInstance.end());
      BuildFlags.insert(BuildFlags.end(), BuildFlagsInstance.begin(),
                        BuildFlagsInstance.end());
    } else {
      BuildLog.append("Loading Model XML & Bin\n");

      if (!readFile(ModelDirectoryPath, M->NGraphXml, ModelXml, BuildLog)) {
        return false;
      }
      if (!readFile(ModelDirectoryPath, M->NGraphBin, ModelBin, BuildLog)) {
        return false;
      }
      BuildFlags = M->BuildFlags;
    }

    if (findModelInNativeCache(Program, ModelXml, ModelBin, BuildFlags, M->Name,
                               ProgCachePath, ProgNativeDir, Out)) {
      BuildLog.append("Model XML & Bin loaded, found cached "
                      "Native Model, not building\n");
      return true;
    }

    BuildLog.append("Model XML & Bin loaded, compiling\n");
    return compileFromXmlBin(ContextH, DeviceH, ModelXml, ModelBin, BuildFlags,
                             ProgCachePath, ProgNativeDir, Out);
  }

  BuildLog.append("Unknown model format\n");
  return false;
}

bool Level0BuiltinProgramBuild::loadBinaries(ze_context_handle_t ContextH,
                                             ze_device_handle_t DeviceH,
                                             ze_command_queue_handle_t QueueH,
                                             ze_command_list_handle_t ListH) {
  for (auto &[KernelName, KernelBuild] : KernelBuilds) {
    if (!loadBinary(ContextH, DeviceH, QueueH, ListH, KernelBuild)) {
      BuildLog.append("Creating Graph from native binary for kernel ");
      BuildLog.append(KernelName);
      BuildLog.append(" in target context failed\n");
      return false;
    }
  }
  return true;
}

ze_graph_handle_t
Level0BuiltinProgramBuild::getGraphHandle(std::string KernelName) {
  auto It = KernelBuilds.find(KernelName);
  if (It == KernelBuilds.end()) {
    POCL_MSG_ERR("getGraphHandle: unknown kernel %s\n", KernelName.c_str());
    return nullptr;
  }
  return It->second.GraphHFinal;
}

void Level0BuiltinProgramBuild::run(ze_context_handle_t ContextH) {

  POCL_MEASURE_START(compilation);

  POCL_MSG_PRINT_LEVEL0("Measuring BuiltinProgram compilation\n");

  auto KerIDs = Program->getKernelIDs();
  auto KerNames = Program->getKernelNames();
  auto KerAttrs = Program->getKernelAttrs();
  size_t NumKernels = KerNames.size();
  std::vector<const Level0Model *> BK_Models;
  // DBK builtin kernels
  assert(NumKernels > 0);

  BuildSuccessful = true;
  for (unsigned KerIdx = 0; KerIdx < NumKernels; ++KerIdx) {
    unsigned KerID = KerIDs[KerIdx];
    const std::string &KName = KerNames[KerIdx];
    const Level0Model *Model = nullptr;
    for (unsigned i = 0; i < NumLevel0GraphModels; ++i) {
      if (Program->isDBK()) {
        if (Level0GraphModels[i].DBK_ID == KerID) {
          Model = &Level0GraphModels[i];
          break;
        }
      } else {
        if (Level0GraphModels[i].Name == KName) {
          Model = &Level0GraphModels[i];
          break;
        }
      }
    }
    if (Model == nullptr) {
      BuildLog.append("BUG: unknown DBK kernel with ID: ");
      BuildLog.append(std::to_string(KerIDs[KerIdx]));
      BuildLog.append(" Name: ");
      BuildLog.append(KName);
      BuildSuccessful = false;
      break;
    } else {
      BK_Models.emplace_back(Model);
    }
  }

  if (!BuildSuccessful)
    return;

  for (unsigned KerIdx = 0; KerIdx < NumKernels; ++KerIdx) {
    Level0BuiltinKernelBuildResult Temp{GraphDDITable};
    BuildSuccessful =
        loadModel(ContextH, Program->getDevice(), BK_Models[KerIdx],
                  Program->isDBK() ? KerAttrs[KerIdx] : nullptr, Temp);
    if (BuildSuccessful)
      KernelBuilds.emplace(KerNames[KerIdx], std::move(Temp));
    else
      break;
  }
  Program = nullptr;

  POCL_MEASURE_FINISH(compilation);
}

Level0BuiltinProgram::Level0BuiltinProgram(
    ze_context_handle_t Ctx, ze_device_handle_t Dev, size_t NumBuiltinKernels,
    char **BuiltinKernelNames,
    void *BuiltinKernelIDs,    // IDs for DBKs
    void **BuiltinKernelAttrs, // Attrs for DBKs
    const char *CDir, const std::string &UUID)
    : Level0ProgramBase(Ctx, Dev, CDir, UUID), QueueH(nullptr), ListH(nullptr) {

  unsigned *IDs = reinterpret_cast<unsigned *>(BuiltinKernelIDs);
  if (BuiltinKernelIDs) {
    IsDBK = true;
    assert(BuiltinKernelAttrs);
    for (size_t i = 0; i < NumBuiltinKernels; ++i) {
      KernelNames.emplace_back(BuiltinKernelNames[i]);
      KernelIDs.emplace_back(IDs[i]);
      KernelAttrs.emplace_back(BuiltinKernelAttrs[i]);
    }
  } else {
    IsDBK = false;
    for (size_t i = 0; i < NumBuiltinKernels; ++i) {
      KernelNames.emplace_back(BuiltinKernelNames[i]);
    }
  }
}

bool Level0BuiltinProgram::init() {
  ze_command_queue_desc_t CmdQueueDesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                          nullptr,
                                          0, // TODO ordinal hardcoded to 0
                                          0, // index
                                          0, // flags
                                          ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                          ZE_COMMAND_QUEUE_PRIORITY_NORMAL};

  ze_command_list_desc_t CmdListDesc = {
      ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, // TODO ordi hardcoded
      ZE_COMMAND_LIST_FLAG_RELAXED_ORDERING |
          ZE_COMMAND_LIST_FLAG_MAXIMIZE_THROUGHPUT};

  ze_result_t Res;
  Res = zeCommandQueueCreate(ContextH, DeviceH, &CmdQueueDesc, &QueueH);
  if (Res == ZE_RESULT_SUCCESS) {
    Res = zeCommandListCreate(ContextH, DeviceH, &CmdListDesc, &ListH);
  }

  return (QueueH != nullptr && ListH != nullptr);
}

Level0BuiltinProgram::~Level0BuiltinProgram() {
  if (QueueH)
    zeCommandQueueDestroy(QueueH);
  if (ListH)
    zeCommandListDestroy(ListH);
}

bool Level0BuiltinProgram::addFinishedBuild(Level0BuildBaseUPtr B) {

  std::lock_guard<std::mutex> LockGuard(Mutex);
  BuildLog.append(B->getBuildLog());
  Level0BuildBase::BuildType BT = B->getBuildType();
  assert(BT == Level0BuildBase::BuildType::BuiltinProgram);

  Level0BuiltinProgramBuild *Build =
      static_cast<Level0BuiltinProgramBuild *>(B.release());
  BuildLog.append(Build->getBuildLog());

  if (!Build->isSuccessful() ||
      !Build->loadBinaries(ContextH, DeviceH, QueueH, ListH)) {
    BuildLog.append("Error: build not successful or "
                    "couldn't load built binary\n");
    return false;
  }

  FinishedBuild.reset(Build);
  return true;
}

// TODO make a copy of graph on createKernel ?
Level0BuiltinKernel *
Level0BuiltinProgram::createKernel(const std::string Name) {
  std::lock_guard<std::mutex> LockGuard(Mutex);
  Level0BuiltinKernelSPtr Kernel = std::make_shared<Level0BuiltinKernel>(Name);
  Kernels.push_back(Kernel);
  return Kernel.get();
}

// TODO release copy of graph on createKernel ?
bool Level0BuiltinProgram::releaseKernel(Level0BuiltinKernel *Kernel) {
  std::lock_guard<std::mutex> LockGuard(Mutex);

  auto Iter = std::find_if(
      Kernels.begin(), Kernels.end(),
      [&Kernel](Level0BuiltinKernelSPtr &K) { return K.get() == Kernel; });

  if (Iter == Kernels.end())
    return false;

  Kernels.erase(Iter);
  return true;
}

bool Level0BuiltinProgram::getBestKernel(Level0BuiltinKernel *BKernel,
                                         ze_graph_handle_t &Ker) {
  std::lock_guard<std::mutex> LockGuard(Mutex);
  assert((bool)FinishedBuild);

  auto It = std::find_if(Kernels.begin(), Kernels.end(),
                         [BKernel](const Level0BuiltinKernelSPtr &V) {
                           Level0BuiltinKernel *JK = V.get();
                           return JK == BKernel;
                         });

  if (It == Kernels.end())
    return false;

  Ker = FinishedBuild->getGraphHandle(BKernel->getName());
  return true;
}

Level0BuiltinKernel::Level0BuiltinKernel(const std::string N) : Name(N) {
  CacheUUID = getStringHash(N);
}

#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

void Level0CompilationJob::signalFinished() {
  std::lock_guard<std::mutex> LockGuard(Mutex);
  Finished = true;
  // adds only successful builds to program
  Successful = Program->addFinishedBuild(std::move(Build));
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
                                 Level0ProgramBase *Prog,
                                 Level0BuildBase *Build) {

  if (Queue.empty()) {
    return Level0CompilationJobSPtr(nullptr);
  }

  ze_device_handle_t Dev = Prog->getDevice();
  auto Iter = std::find_if(Queue.begin(), Queue.end(),
                           [&Prog, &Build, Dev](Level0CompilationJobSPtr &J) {
                             return J->isForProgram(Prog) &&
                                    J->isForBuild(Build) && J->isForDevice(Dev);
                           });

  if (Iter == Queue.end()) {
    return Level0CompilationJobSPtr(nullptr);
  }

  return *Iter;
}

Level0CompilationJobSPtr Level0CompilerJobQueue::findOrCreateWork(
    bool HiPrio, Level0ProgramBaseSPtr ProgS, Level0BuildBaseUPtr BuildU) {

  std::unique_lock<std::mutex> UniqLock(Mutex);
  Level0BuildBase *Build = BuildU.get();
  Level0ProgramBase *Prog = ProgS.get();

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

void Level0CompilerJobQueue::cancelAllJobsForProgram(
    Level0ProgramBase *Program) {
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

  DriverH = H;

#ifdef ENABLE_NPU
  ze_result_t Res;
  Res = zeDriverGetExtensionFunctionAddress(
      DriverH, GRAPH_EXT_NAME, reinterpret_cast<void **>(&GraphDDITable));
  if (Res != ZE_RESULT_SUCCESS)
    GraphDDITable = nullptr;

  Res = zeDriverGetExtensionFunctionAddress(
      DriverH, ZE_PROFILING_DATA_EXT_NAME,
      reinterpret_cast<void **>(&GraphProfDDITable));
  if (Res != ZE_RESULT_SUCCESS)
    GraphProfDDITable = nullptr;

  if (!GraphDDITable || !GraphProfDDITable) {
    POCL_MSG_PRINT_LEVEL0(
        "JobScheduler: Failed to initialize LevelZero Graph Ext\n");
  }
#endif

  JobQueue = std::make_unique<Level0CompilerJobQueue>();
  unsigned NumDevices = DevicesH.size();
  unsigned NumThreads = std::min((NumDevices * 2), std::thread::hardware_concurrency());
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

  Level0ProgramSPtr Program =
      findProgram<Level0Program, Level0ProgramSPtr>(Prog, Programs, true);
  return (bool)Program;
}

template <class P, class SPtr>
SPtr Level0CompilationJobScheduler::findProgram(P *Prog, std::list<SPtr> &List,
                                                bool erase) {
  std::lock_guard<std::mutex> Lock(ProgramsLock);
  SPtr Program;

  auto Iter = std::find_if(List.begin(), List.end(),
                           [&Prog](SPtr &Parg) { return Parg.get() == Prog; });

  if (Iter == List.end())
    return Program;

  if (erase)
    List.erase(Iter);
  else
    Program = *Iter;

  return Program;
}

#ifdef ENABLE_NPU
Level0BuiltinProgramSPtr
Level0CompilationJobScheduler::findProgram(Level0BuiltinProgram *Prog) {
  return findProgram<Level0BuiltinProgram, Level0BuiltinProgramSPtr>(
      Prog, BuiltinPrograms);
}
#endif

Level0ProgramSPtr
Level0CompilationJobScheduler::findProgram(Level0Program *Prog) {
  return findProgram<Level0Program, Level0ProgramSPtr>(Prog, Programs);
}

Level0Kernel *Level0CompilationJobScheduler::createKernel(Level0Program *Prog,
                                                          const char *Name) {
  Level0ProgramSPtr Program = findProgram(Prog);

  if (!Program) {
    POCL_MSG_ERR("cannot find a program %p\n", Prog);
    return nullptr;
  }

  Level0Kernel *K = Program->createKernel(Name);
  if (!K)
    return nullptr;

  // prebuild a 32bit small-WG specialization here
  // this might not be necessary but is useful for timing & catching errors
  // early

  if (Prog->isJITCompiled()) {
    if (pocl_get_bool_option("POCL_LEVEL0_JIT_PREBUILD", 0)) {
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
  } else {
    // prebuild a 32bit any-WG kernel specialization here.
    // This is for local size optimizer
    ze_module_handle_t Mod;
    ze_kernel_handle_t Ker;
    Program->getBestKernel(K, false, true, Mod, Ker);
  }

  return K;
}

bool Level0CompilationJobScheduler::releaseKernel(Level0Program *Prog,
                                                  Level0Kernel *Kernel) {
  Level0ProgramSPtr Program = findProgram(Prog);
  if (!Program) {
    POCL_MSG_ERR("cannot find a program %p\n", Prog);
    return false;
  }
  return Program->releaseKernel(Kernel);
}

bool Level0CompilationJobScheduler::createProgramBuilds(
    Level0ProgramSPtr Program, std::string &BuildLog,
    bool DeviceSupports64bitBuffers, bool Optimize) {
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

bool Level0CompilationJobScheduler::createProgramBuildFullOptions(
    Level0ProgramSPtr Program, std::string &BuildLog, bool WaitForFinish,
    bool Optimize, bool LargeOffsets, bool SmallWG, bool HighPrio) {

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

bool Level0CompilationJobScheduler::createAndWaitKernelJITBuilds(
    Level0ProgramSPtr Program, Level0Kernel *Kernel, bool LargeOffsets,
    bool SmallWG) {

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
  Level0ProgramSPtr Program = findProgram(Prog);
  if (!Program) {
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

#ifdef ENABLE_NPU
Level0BuiltinProgram *Level0CompilationJobScheduler::createBuiltinProgram(
    ze_context_handle_t Ctx, ze_device_handle_t Dev, std::string &BuildLog,
    size_t num_builtin_kernels, char **builtin_kernel_names,
    void *builtin_kernel_ids, void **builtin_kernel_attributes,
    const char *CDir, const std::string &UUID) {

  if (!GraphDDITable || !GraphProfDDITable) {
    BuildLog.append("Failed to initialize LevelZero Graph Ext - "
                    "cannot create program\n");
    return nullptr;
  }

  Level0BuiltinProgramSPtr Prog = std::make_shared<Level0BuiltinProgram>(
      Ctx, Dev, num_builtin_kernels, builtin_kernel_names, builtin_kernel_ids,
      builtin_kernel_attributes, CDir, UUID);
  if (!Prog->init()) {
    BuildLog.append("failed to initialize Level0BuiltinProgram\n");
    return nullptr;
  }

  bool Res = createAndWaitBuiltinProgramBuilds(Prog, BuildLog);
  if (!Res) {
    BuildLog.append("failed to build Level0BuiltinProgram\n");
    return nullptr;
  }

  std::lock_guard<std::mutex> Lock(ProgramsLock);
  BuiltinPrograms.push_back(Prog);
  return Prog.get();
}

bool Level0CompilationJobScheduler::createAndWaitBuiltinProgramBuilds(
    Level0BuiltinProgramSPtr Program, std::string &BuildLog) {
  Level0BuiltinProgramBuildUPtr ProgBuild =
      std::make_unique<Level0BuiltinProgramBuild>(Program.get(), GraphDDITable);

  Level0CompilationJobSPtr ProgBuildJob =
      JobQueue->findOrCreateWork(true, Program, std::move(ProgBuild));
  ProgBuildJob->waitForFinish();
  BuildLog.append(Program->getBuildLog());
  return ProgBuildJob->isSuccessful();
}

bool Level0CompilationJobScheduler::releaseBuiltinProgram(
    Level0BuiltinProgram *Prog) {
  JobQueue->cancelAllJobsForProgram(Prog);
  Level0BuiltinProgramSPtr Program =
      findProgram<Level0BuiltinProgram, Level0BuiltinProgramSPtr>(
          Prog, BuiltinPrograms, true);
  return (bool)Program;
}

Level0BuiltinKernel *
Level0CompilationJobScheduler::createBuiltinKernel(Level0BuiltinProgram *Prog,
                                                   const char *Name) {
  Level0BuiltinProgramSPtr Program = findProgram(Prog);

  if (!Program) {
    POCL_MSG_ERR("Cannot find program %p\n", Prog);
    return nullptr;
  }

  return Program->createKernel(Name);
}

bool Level0CompilationJobScheduler::releaseBuiltinKernel(
    Level0BuiltinProgram *Prog, Level0BuiltinKernel *Kernel) {
  Level0BuiltinProgramSPtr Program = findProgram(Prog);
  if (!Program) {
    POCL_MSG_ERR("Cannot find program %p\n", Prog);
    return false;
  }
  return Program->releaseKernel(Kernel);
}

bool Level0CompilationJobScheduler::getBestBuiltinKernel(
    Level0BuiltinProgram *Prog, Level0BuiltinKernel *Kernel,
    ze_graph_handle_t &Graph) {
  Level0BuiltinProgramSPtr Program = findProgram(Prog);
  if (!Program) {
    POCL_MSG_ERR("Cannot find program %p\n", Prog);
    return false;
  }
  return Program->getBestKernel(Kernel, Graph);
}

#endif
