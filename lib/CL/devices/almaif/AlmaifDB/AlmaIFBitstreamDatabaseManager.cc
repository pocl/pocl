/* AlmaIFBitstreamDatabaseManager.cc - Parses and responds to queries about
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

#include "AlmaIFBitstreamDatabaseManager.hh"

#include "../AlmaifShared.hh"

#include "pocl_file_util.h"

#include <dirent.h>
#include <iostream>
#include <set>

#include "tiny-json.h"

void AlmaIFBitstreamDatabaseManager::parseOverlay(json_t const *Overlay,
                                                  const std::string &DBPath) {

  json_t const *OverlayName = json_getProperty(Overlay, "name");
  if (!OverlayName || JSON_TEXT != json_getType(OverlayName)) {
    POCL_ABORT("Error, the overlay name property is not found.");
  }
  std::string OverlayNameStr = json_getValue(OverlayName);
  POCL_MSG_PRINT_ALMAIF("Overlay Name: %s.\n", OverlayNameStr.c_str());

  std::string OverlayPath = DBPath + "/" + OverlayNameStr;

  json_t const *PrDevice = json_getProperty(Overlay, "device");
  if (!PrDevice || JSON_TEXT != json_getType(PrDevice)) {
    POCL_ABORT("Overlay doesn't have associated device\n");
  }
  std::string PrDeviceName = json_getValue(PrDevice);
  POCL_MSG_PRINT_ALMAIF("PR device name: %s\n", PrDeviceName.c_str());
  DEVICE_TYPE PrDeviceEnum = string2DeviceTypeEnum(PrDeviceName);
  POCL_MSG_PRINT_ALMAIF("PR device enum: %d\n", PrDeviceEnum);

  struct ProgrammingFiles ProgFilesInfo = {0, PrDeviceEnum, "", "", ""};

  json_t const *OverlayDefaultFilename = json_getProperty(Overlay, "filename");
  if (!OverlayDefaultFilename ||
      JSON_TEXT != json_getType(OverlayDefaultFilename)) {
    POCL_ABORT("Error, the overlay default filename property is not found.");
  }
  std::string OverlayDefaultFilenameStr = json_getValue(OverlayDefaultFilename);

  std::string OverlayDefaultFilenamePath =
      OverlayPath + "/" + OverlayDefaultFilenameStr;

  POCL_MSG_PRINT_ALMAIF("Overlay default filename path: %s.\n",
                        OverlayDefaultFilenamePath.c_str());
  DefaultFilenamePath_ = OverlayDefaultFilenamePath;

  json_t const *OverlayDefaultKernelName =
      json_getProperty(Overlay, "default-kernel");
  if (!OverlayDefaultKernelName ||
      JSON_TEXT != json_getType(OverlayDefaultKernelName)) {
    POCL_ABORT("Error, the overlay default kernel name property is not found.");
  }
  std::string OverlayDefaultKernelNameStr =
      json_getValue(OverlayDefaultKernelName);
  DefaultKernelName_ = OverlayDefaultKernelNameStr;

  json_t const *OverlayExternalMemory =
      json_getProperty(Overlay, "external-memory");
  if (!OverlayExternalMemory ||
      JSON_TEXT != json_getType(OverlayExternalMemory)) {
    POCL_ABORT("Error, the overlay external-memory property is not found.");
  }
  DeviceExternalMemParameters_[PrDeviceEnum] =
      json_getValue(OverlayExternalMemory);

  json_t const *Accelerators = json_getProperty(Overlay, "accelerators");
  if (!Accelerators || JSON_ARRAY != json_getType(Accelerators)) {
    POCL_ABORT("Error, accelerators list not parsed\n");
  }

  json_t const *Accel;
  for (Accel = json_getChild(Accelerators); Accel != 0;
       Accel = json_getSibling(Accel)) {
    parseAccelerator(Accel, ProgFilesInfo, OverlayPath);
  }
}

void AlmaIFBitstreamDatabaseManager::parseAccelerator(
    json_t const *Accel, struct ProgrammingFiles &ProgFilesInfo,
    const std::string &OverlayPath) {

  json_t const *AccelNameJs = json_getProperty(Accel, "name");
  if (!AccelNameJs || JSON_TEXT != json_getType(AccelNameJs)) {
    POCL_ABORT("Partial bitstream doesn't have a name\n");
  }
  ProgFilesInfo.KernelName = json_getValue(AccelNameJs);
  POCL_MSG_PRINT_ALMAIF("PR device name: %s\n",
                        ProgFilesInfo.KernelName.c_str());

  std::string AcceleratorPath =
      OverlayPath + "/accelerators/" + ProgFilesInfo.KernelName;

  json_t const *PrBitstream = json_getProperty(Accel, "filename");
  if (!PrBitstream || JSON_TEXT != json_getType(PrBitstream)) {
    POCL_ABORT("Partial bitstream filename parsing failed\n");
  }

  std::string PrBitstreamPath =
      AcceleratorPath + "/" + json_getValue(PrBitstream);
  POCL_MSG_PRINT_ALMAIF("Arria device pr file %s\n", PrBitstreamPath.c_str());
  ProgFilesInfo.BitstreamPath = PrBitstreamPath;

  json_t const *Firmwares = json_getProperty(Accel, "firmwares");
  if (!Firmwares || JSON_ARRAY != json_getType(Firmwares)) {
    POCL_ABORT("Error, firmwares not found\n");
  }
  for (json_t const *Firmware = json_getChild(Firmwares); Firmware != 0;
       Firmware = json_getSibling(Firmware)) {
    parseFirmware(Firmware, ProgFilesInfo, AcceleratorPath);
  }
}

void AlmaIFBitstreamDatabaseManager::parseFirmware(
    json_t const *Firmware, struct ProgrammingFiles &ProgFilesInfo,
    const std::string &AcceleratorPath) {
  json_t const *FirmwarePath = json_getProperty(Firmware, "filename");
  if (!FirmwarePath || JSON_TEXT != json_getType(FirmwarePath)) {
    POCL_ABORT("Error, firmware filepath not found from json\n");
  }
  std::string FirmwarePathStr =
      AcceleratorPath + "/firmwares/" + json_getValue(FirmwarePath);
  ProgFilesInfo.FirmwarePath = FirmwarePathStr;

  json_t const *BiKernels = json_getProperty(Firmware, "builtin-kernels");
  if (!BiKernels || JSON_ARRAY != json_getType(BiKernels)) {
    POCL_ABORT("Error, builtin kernels not found\n");
  }
  json_t const *Bik;
  for (Bik = json_getChild(BiKernels); Bik != 0; Bik = json_getSibling(Bik)) {
    parseBIKernels(Bik, ProgFilesInfo);
  }
}

void AlmaIFBitstreamDatabaseManager::parseBIKernels(
    json_t const *Bik, struct ProgrammingFiles &ProgFilesInfo) {
  if (JSON_INTEGER != json_getType(Bik)) {
    POCL_ABORT("Error, Builtin kernel id is wrong type\n");
  }
  int64_t BikIDLong = json_getInteger(Bik);
  assert(BikIDLong < 0xFFFF);

  cl_dbk_id_exp BikID = (cl_dbk_id_exp)BikIDLong;
  SupportedBIKernels_[ProgFilesInfo.FpgaType].push_back(
      {BikID, ProgFilesInfo.FpgaType, ProgFilesInfo.BitstreamPath,
       ProgFilesInfo.FirmwarePath, ProgFilesInfo.KernelName});
  POCL_MSG_PRINT_ALMAIF(
      "Found support for builtin kernel %d with fw path: %s\n", BikID,
      ProgFilesInfo.FirmwarePath.c_str());
}

AlmaIFBitstreamDatabaseManager::AlmaIFBitstreamDatabaseManager(
    const std::string &DBPath) {

  std::string DBFile = DBPath;

  DIR *dp;
  struct dirent *dirp;
  if ((dp = opendir(DBPath.c_str())) == NULL) {
    POCL_ABORT("Failed opening the Almaif db directory\n");
  }
  while ((dirp = readdir(dp)) != NULL) {
    std::string OverlayFolderName = dirp->d_name;
    if (OverlayFolderName.find("overlay") != std::string::npos) {
      POCL_MSG_PRINT_ALMAIF("Found overlay dir %s\n",
                            OverlayFolderName.c_str());
      std::string BitstreamDatabaseIndexPath =
          DBFile + "/" + OverlayFolderName + "/db.json";

      uint64_t Size = 0;
      char *BitstreamDatabaseIndex = NULL;
      pocl_read_file(BitstreamDatabaseIndexPath.c_str(),
                     &BitstreamDatabaseIndex, &Size);
      POCL_MSG_PRINT_ALMAIF("Read file size=%lld\n", Size);

      POCL_MSG_PRINT_ALMAIF("DATABASE FILE %s:\n",
                            BitstreamDatabaseIndexPath.c_str());
      POCL_MSG_PRINT_ALMAIF("%s\n", BitstreamDatabaseIndex);
      POCL_MSG_PRINT_ALMAIF("DATABASE FILE END\n");

      json_t Mem[256];
      json_t const *t =
          json_create(BitstreamDatabaseIndex, Mem, sizeof(Mem) / sizeof(*Mem));
      if (!t) {
        POCL_ABORT("Failed opening AlmaifDB as json object\n");
      }
      parseOverlay(t, DBPath);
    }
  }
}

AlmaIFBitstreamDatabaseManager::~AlmaIFBitstreamDatabaseManager() {}

AlmaIFBitstreamDatabaseManager::DEVICE_TYPE
AlmaIFBitstreamDatabaseManager::string2DeviceTypeEnum(const std::string &Str) {
  unsigned int Len = Str.length();
  std::string StringToConvert = Str;
  for (int i = 0; i < Len; i++) {
    StringToConvert[i] = tolower(Str[i]);
  }

  for (int j = 0; j < sizeof(Conversion) / sizeof(Conversion[0]); ++j)
    if (StringToConvert == Conversion[j].Str) {
      return Conversion[j].Val;
    } else {
      POCL_MSG_PRINT_ALMAIF(
          "String-to-enum. String:%s, comparing:%s, lengths:%d,%d\n",
          Conversion[j].Str.c_str(), StringToConvert.c_str(),
          Conversion[j].Str.length(), StringToConvert.length());
    }
  POCL_ABORT("Almaif DB device string to enum conversion failed. String %s\n",
             Str.c_str());
}

std::string
AlmaIFBitstreamDatabaseManager::deviceTypeEnum2String(DEVICE_TYPE DeviceType) {
  for (int j = 0; j < sizeof(Conversion) / sizeof(Conversion[0]); ++j) {
    if (DeviceType == Conversion[j].Val) {
      return Conversion[j].Str;
    }
  }
  POCL_ABORT("Almaif DB device enum to string conversion failed");
}

const AlmaIFBitstreamDatabaseManager::ProgrammingFiles &
AlmaIFBitstreamDatabaseManager::getBitstreamFile(cl_dbk_id_exp BikID,
                                                 DEVICE_TYPE UsedDeviceType) {

  for (const ProgrammingFiles &Iter : SupportedBIKernels_[UsedDeviceType]) {
    if (Iter.BikID == BikID) {
      return Iter;
    }
  }
  POCL_ABORT("Built in kernel %d bitstream not found\n", BikID);
}

const AlmaIFBitstreamDatabaseManager::ProgrammingFiles &
AlmaIFBitstreamDatabaseManager::getFirmwareFile(cl_dbk_id_exp BikID,
                                                DEVICE_TYPE UsedDeviceType) {

  for (const ProgrammingFiles &Iter : SupportedBIKernels_[UsedDeviceType]) {
    if (Iter.BikID == BikID) {
      return Iter;
    }
  }
  POCL_ABORT("Built in kernel %d firmware not found\n", BikID);
}

std::vector<cl_dbk_id_exp>
AlmaIFBitstreamDatabaseManager::supportedBuiltinKernels(
    DEVICE_TYPE UsedDeviceType) {

  std::vector<cl_dbk_id_exp> Output;
  for (const ProgrammingFiles &Iter : SupportedBIKernels_[UsedDeviceType]) {
    Output.push_back((cl_dbk_id_exp)Iter.BikID);
  }
  return Output;
}

std::string AlmaIFBitstreamDatabaseManager::externalMemoryParameters(
    DEVICE_TYPE UsedDeviceType) {

  return DeviceExternalMemParameters_[UsedDeviceType];
}

std::string AlmaIFBitstreamDatabaseManager::defaultBitstream() {
  return DefaultFilenamePath_;
}

std::string AlmaIFBitstreamDatabaseManager::defaultKernelName() {
  return DefaultKernelName_;
}
