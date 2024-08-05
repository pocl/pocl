/* AlmaIFBitstreamDatabaseManager.hh - Parses and responds to queries about
   AlmaifDB

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

#ifndef POCL_ALMAIFBITSTREAMDATABASEMANAGER_H
#define POCL_ALMAIFBITSTREAMDATABASEMANAGER_H

#include <map>
#include <string>
#include <vector>

#include "pocl_builtin_kernels.h"

typedef struct json_s json_t;

// A helper class used by DBDevice to parse the bitstream database.
// This class can be thought to be the interface from C++ to the JSON-based
// database. After parsing, the DBDevice will query this class
// for information about the bitstream database.
//
// Since the AFOCL bitstream database format is still very experimental, there
// is no fixed specification for it. Therefore, this class defines the format
// since it is responsible for parsing the database.
// While having a clear, versioned format would obviously be the best solution,
// maintaining a separate specification at this point would risk it
// going out-of-date. You can see the separate AFOCL-project for examples of
// the current database format.
class AlmaIFBitstreamDatabaseManager {
public:
  AlmaIFBitstreamDatabaseManager(const std::string &DBPath);
  virtual ~AlmaIFBitstreamDatabaseManager();

  enum DEVICE_TYPE { ARRIA10, ALVEOU280 };

  DEVICE_TYPE string2DeviceTypeEnum(const std::string &Str);
  std::string deviceTypeEnum2String(DEVICE_TYPE DeviceType);

  struct ProgrammingFiles {
    int BikID;
    DEVICE_TYPE FpgaType;
    std::string BitstreamPath;
    std::string FirmwarePath;
    std::string KernelName;
  };
  const ProgrammingFiles &getBitstreamFile(BuiltinKernelId BikID,
                                           DEVICE_TYPE UsedDeviceType);
  const ProgrammingFiles &getFirmwareFile(BuiltinKernelId BikID,
                                          DEVICE_TYPE UsedDeviceType);

  std::vector<BuiltinKernelId>
  supportedBuiltinKernels(DEVICE_TYPE UsedDeviceType);
  std::string externalMemoryParameters(DEVICE_TYPE UsedDeviceType);
  std::string defaultBitstream();
  std::string defaultKernelName();

private:
  void parseDB(json_t const *DB, const std::string &DBPath);
  void parseOverlay(json_t const *Overlay, const std::string &DBPath);
  void parseAccelerator(json_t const *Accel,
                        struct ProgrammingFiles &ProgFilesInfo,
                        const std::string &OverlayPath);
  void parseFirmware(json_t const *Firmware,
                     struct ProgrammingFiles &ProgFilesInfo,
                     const std::string &AcceleratorPath);
  void parseBIKernels(json_t const *Bik,
                      struct ProgrammingFiles &ProgFilesInfo);

  const struct {
    DEVICE_TYPE Val;
    const std::string Str;
  } Conversion[2] = {
      {ARRIA10, "arria10"},
      {ALVEOU280, "alveou280"},
  };

  std::map<DEVICE_TYPE, std::vector<ProgrammingFiles>> SupportedBIKernels_;
  std::map<DEVICE_TYPE, std::string> DeviceExternalMemParameters_;
  std::string DefaultFilenamePath_;
  std::string DefaultKernelName_;
};

#endif
