/* DBDevice.cc - Device based on parsing Almaif database and instantiating
 *               other device types based on what it finds from there

   Copyright (c) 2023 Topi Leppänen / Tampere University

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

#include "DBDevice.hh"

#include "../AlmaifShared.hh"

#ifdef HAVE_XRT
#include "../XilinxXrtDevice.hh"
#include "../XilinxXrtRegion.hh"
#endif

DBDevice::DBDevice(const std::string &DBPath) : DB_(DBPath) {

  POCL_MSG_PRINT_INFO("Starting bitstream database device initialization");

  bool UseAlveoDevice = false;
  bool UseIntelDevice = false;
  if (pocl_is_option_set("XILINX_XRT")) {
    UseAlveoDevice = true;
    UsedDeviceType_ = AlmaIFBitstreamDatabaseManager::DEVICE_TYPE::ALVEOU280;
  }
  if (pocl_is_option_set("INTEL_ACL")) {
    UseIntelDevice = true;
    UsedDeviceType_ = AlmaIFBitstreamDatabaseManager::DEVICE_TYPE::ARRIA10;
  }
  if (UseAlveoDevice && UseIntelDevice) {
    POCL_ABORT("AlmaIF: DBDevice only supports one vendor FPGA at the time\n");
  }

  std::string ExternalMemParams = DB_.externalMemoryParameters(UsedDeviceType_);

  if (UseAlveoDevice) {
    Dev_ = new XilinxXrtDevice(DB_.defaultKernelName(), DB_.defaultBitstream(),
                               ExternalMemParams, 0);
  } else if (UseIntelDevice) {
    POCL_ABORT_UNIMPLEMENTED("AlmaIF intel device not implemented\n");
  } else {
    POCL_ABORT("AlmaIF: DBDevice didn't find any vendor FPGAs\n");
  }

  ControlMemory = Dev_->ControlMemory;
  InstructionMemory = Dev_->InstructionMemory;
  CQMemory = Dev_->CQMemory;
  DataMemory = Dev_->DataMemory;
  RelativeAddressing = Dev_->RelativeAddressing;
  HasHardwareClock = Dev_->HasHardwareClock;
  HwClockFrequency = Dev_->HwClockFrequency;
  PointerSize = Dev_->PointerSize;
  ExternalMemory = Dev_->ExternalMemory;
  AllocRegions = Dev_->AllocRegions;
}

DBDevice::~DBDevice() { delete Dev_; }

void DBDevice::programBIKernelBitstream(cl_dbk_id_exp BikID) {

  const AlmaIFBitstreamDatabaseManager::ProgrammingFiles &BitstreamToProgram =
      DB_.getBitstreamFile(BikID, UsedDeviceType_);
  std::string BitstreamPath = BitstreamToProgram.BitstreamPath;
  std::string KernelName = BitstreamToProgram.KernelName;

  if (BitstreamPath == LoadedBitstreamPath_) {
    return;
  }

  POCL_MSG_PRINT_ALMAIF("Programming built-in kernel %s bitstream from: %s\n",
                        KernelName.c_str(), BitstreamPath.c_str());
  if (UsedDeviceType_ ==
      AlmaIFBitstreamDatabaseManager::DEVICE_TYPE::ALVEOU280) {
    ((XilinxXrtDevice *)Dev_)
        ->programBitstream(KernelName.c_str(), BitstreamPath.c_str(), 0);
  } else if (UsedDeviceType_ ==
             AlmaIFBitstreamDatabaseManager::DEVICE_TYPE::ARRIA10) {
    POCL_ABORT_UNIMPLEMENTED("AlmaIF intel device not implemented\n");
  } else {
    POCL_ABORT("Almaif neither device activated\n");
  }

  LoadedBitstreamPath_ = BitstreamPath;
}

void DBDevice::programBIKernelFirmware(cl_dbk_id_exp BikID) {

  const AlmaIFBitstreamDatabaseManager::ProgrammingFiles &BitstreamToProgram =
      DB_.getFirmwareFile(BikID, UsedDeviceType_);
  std::string FirmwarePath = BitstreamToProgram.FirmwarePath;

  if (FirmwarePath == LoadedFirmwarePath_) {
    return;
  }
  POCL_MSG_PRINT_ALMAIF("Programming built-in kernel firmware from: %s\n",
                        FirmwarePath.c_str());

  ControlMemory->Write32(ALMAIF_CONTROL_REG_COMMAND, ALMAIF_RESET_CMD);

  if (UsedDeviceType_ ==
      AlmaIFBitstreamDatabaseManager::DEVICE_TYPE::ALVEOU280) {
    ((XilinxXrtRegion *)InstructionMemory)->initRegion(FirmwarePath.c_str());
  } else if (UsedDeviceType_ ==
             AlmaIFBitstreamDatabaseManager::DEVICE_TYPE::ARRIA10) {
    POCL_ABORT_UNIMPLEMENTED("AlmaIF intel device not implemented\n");
  } else {
    POCL_ABORT("Neither device activated\n");
  }
  ControlMemory->Write32(ALMAIF_CONTROL_REG_COMMAND, ALMAIF_CONTINUE_CMD);

  POCL_MSG_PRINT_ALMAIF("Programming done");
  LoadedFirmwarePath_ = FirmwarePath;
}

void DBDevice::loadProgramToDevice(almaif_kernel_data_s *KernelData,
                                   cl_kernel Kernel,
                                   _cl_command_node *Command) {
  Dev_->loadProgramToDevice(KernelData, Kernel, Command);
}

void DBDevice::printMemoryDump() { Dev_->printMemoryDump(); }

void DBDevice::writeDataToDevice(pocl_mem_identifier *DstMemId,
                                 const char *__restrict__ const Src,
                                 size_t Size, size_t Offset) {
  Dev_->writeDataToDevice(DstMemId, Src, Size, Offset);
}

void DBDevice::readDataFromDevice(char *__restrict__ const Dst,
                                  pocl_mem_identifier *SrcMemId, size_t Size,
                                  size_t Offset) {
  Dev_->readDataFromDevice(Dst, SrcMemId, Size, Offset);
}

cl_int DBDevice::allocateBuffer(pocl_mem_identifier *P, size_t Size) {
  Dev_->allocateBuffer(P, Size);
}

cl_int DBDevice::allocatePipe(pocl_mem_identifier *P, size_t Size) {
  Dev_->allocatePipe(P, Size);
}

int DBDevice::pipeCount() { return Dev_->pipeCount(); }

void DBDevice::freeBuffer(pocl_mem_identifier *P) { Dev_->freeBuffer(P); }
void DBDevice::freePipe(pocl_mem_identifier *P) { Dev_->freePipe(P); }

size_t DBDevice::pointerDeviceOffset(pocl_mem_identifier *P) {
  Dev_->pointerDeviceOffset(P);
}

void DBDevice::discoverDeviceParameters() { Dev_->discoverDeviceParameters(); }

std::vector<cl_dbk_id_exp> DBDevice::supportedBuiltinKernels() {
  return DB_.supportedBuiltinKernels(UsedDeviceType_);
}
