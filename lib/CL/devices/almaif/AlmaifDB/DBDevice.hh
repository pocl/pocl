/* DBDevice.hh - Device based on parsing Almaif database and instantiating
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

#ifndef POCL_DBDEVICE_H
#define POCL_DBDEVICE_H

#include "../AlmaIFDevice.hh"
#include "AlmaIFBitstreamDatabaseManager.hh"

// A class that acts as an interface between the Almaif-driver
// and the underlying FPGA device.
// This class is FPGA vendor-agnostic AlmaIFDevice.
// It instantiates a separate vendor-specific AlmaIFDevice-class.
// Many of the class methods are simply forwarded as is to the
// underlying vendor-specific AlmaIFDevice stored in the private
// Dev_-variable.
//
// This class uses AlmaIFBitstreamManager-class to parse the
// bitstream database, and to fetch the bitstream and firmware
// filepaths from there.
class DBDevice : public AlmaIFDevice {

public:
  DBDevice(const std::string &DBPath);
  ~DBDevice();

  virtual void loadProgramToDevice(almaif_kernel_data_s *KernelData,
                                   cl_kernel Kernel, _cl_command_node *Command);
  void printMemoryDump();
  void writeDataToDevice(pocl_mem_identifier *DstMemId,
                         const char *__restrict__ const Src, size_t Size,
                         size_t Offset) override;
  void readDataFromDevice(char *__restrict__ const Dst,
                          pocl_mem_identifier *SrcMemId, size_t Size,
                          size_t Offset) override;
  cl_int allocateBuffer(pocl_mem_identifier *P, size_t Size) override;
  cl_int allocatePipe(pocl_mem_identifier *P, size_t Size) override;
  void freeBuffer(pocl_mem_identifier *P) override;
  void freePipe(pocl_mem_identifier *P) override;
  size_t pointerDeviceOffset(pocl_mem_identifier *P) override;

  virtual void programBIKernelFirmware(BuiltinKernelId BikID);
  virtual void programBIKernelBitstream(BuiltinKernelId BikID);

  virtual std::vector<BuiltinKernelId> supportedBuiltinKernels();
  virtual void discoverDeviceParameters();
  int pipeCount() override;

  bool isDBDevice() override { return true; }

protected:
private:
  AlmaIFBitstreamDatabaseManager DB_;
  AlmaIFDevice *Dev_;

  AlmaIFBitstreamDatabaseManager::DEVICE_TYPE UsedDeviceType_;

  std::string LoadedBitstreamPath_ = "";
  std::string LoadedFirmwarePath_ = "";
};

#endif
