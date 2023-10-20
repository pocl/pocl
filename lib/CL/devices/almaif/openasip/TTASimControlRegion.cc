/* TTASimControlRegion.cc - TTASim device pretending to be mmapped device

   Copyright (c) 2022 Topi Leppänen / Tampere University

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

#include "TTASimControlRegion.hh"
#include "TTASimDevice.hh"

#include "../AlmaifShared.hh"

#include <SimpleSimulatorFrontend.hh>
//#include <SimulatorFrontend.hh>
#include <AddressSpace.hh>
#include <Machine.hh>

#include <math.h>

TTASimControlRegion::TTASimControlRegion(const TTAMachine::Machine &mach,
                                         TTASimDevice *parent) {

  POCL_MSG_PRINT_ALMAIF_MMAP("TTASim: Initializing TTASimControlRegion\n");
  PhysAddress_ = 0;
  Size_ = ALMAIF_DEFAULT_CTRL_SIZE;
  parent_ = parent;
  assert(parent_ != nullptr &&
         "simulator parent handle NULL, is the sim opened properly?");

  setupControlRegisters(mach);
}

uint32_t TTASimControlRegion::Read32(size_t offset) {

  POCL_MSG_PRINT_ALMAIF_MMAP("MMAP: Reading from physical address 0x%zx with "
                             "offset 0x%zx\n",
                             PhysAddress_, offset);
  assert(offset < Size_ && "Attempt to access data outside MMAP'd buffer");
  auto value = ControlRegisters_[offset / sizeof(uint32_t)];
  return value;
}

void TTASimControlRegion::Write32(size_t offset, uint32_t value) {
  POCL_MSG_PRINT_ALMAIF_MMAP("MMAP: Writing to physical address 0x%zx with "
                             "offset 0x%zx\n",
                             PhysAddress_, offset);

  if (offset == ALMAIF_CONTROL_REG_COMMAND) {
    switch (value) {
    case ALMAIF_RESET_CMD:
      parent_->stopProgram();
      break;
    case ALMAIF_CONTINUE_CMD:
      parent_->restartProgram();
      break;
    }
  }
}

void TTASimControlRegion::Write64(size_t offset, uint64_t value) {
  POCL_ABORT("64b writes to ttasimcontrolregion unimplemented\n");
}

void TTASimControlRegion::Write16(size_t offset, uint16_t value) {
  POCL_ABORT("Unimplemented 16bit writes to ttasimcontrolregion\n");
}

uint64_t TTASimControlRegion::Read64(size_t offset) {

  POCL_MSG_PRINT_ALMAIF_MMAP("MMAP: Reading from physical address 0x%zx with "
                             "offset 0x%zx\n",
                             PhysAddress_, offset);
  assert(offset < Size_ && "Attempt to access data outside MMAP'd buffer");
  auto value = reinterpret_cast<uint64_t *>(
      ControlRegisters_)[offset / sizeof(uint64_t)];
  return value;
}

void TTASimControlRegion::CopyToMMAP(size_t destination, const void *source,
                                     size_t bytes) {
  POCL_ABORT("Unimplemented copytommap for ttasimcontrolregion\n");
}

void TTASimControlRegion::CopyFromMMAP(void *destination, size_t source,
                                       size_t bytes) {

  POCL_ABORT("Unimplemented copyfrommmap for ttasimcontrolregion\n");
}

void TTASimControlRegion::CopyInMem(size_t source, size_t destination,
                                    size_t bytes) {

  POCL_ABORT("Unimplemented copyinmem for ttasimcontrolregion\n");
}

void TTASimControlRegion::setupControlRegisters(
    const TTAMachine::Machine &mach) {
  bool hasPrivateMem = false;
  bool sharedDataAndCq = false;
  bool relativeAddressing = true;
  int DmemSize = 0;
  int CQSize = 0;
  int ImemSize = 0;
  const TTAMachine::Machine::AddressSpaceNavigator &nav =
      mach.addressSpaceNavigator();
  for (int i = 0; i < nav.count(); i++) {
    TTAMachine::AddressSpace *as = nav.item(i);
    if (as->hasNumericalId(TTA_ASID_GLOBAL)) {
      if (as->end() == UINT32_MAX) {
        DmemSize = pow(2, 15); // TODO magic number from almaifintegrator.cc
        relativeAddressing = false;
      } else {
        DmemSize = as->end() + 1;
      }
      if (as->hasNumericalId(TTA_ASID_CQ)) {
        sharedDataAndCq = true;
      }
    } else if (as->hasNumericalId(TTA_ASID_CQ)) {
      CQSize = as->end() + 1;
    } else if (as->hasNumericalId(TTA_ASID_PRIVATE)) {
      hasPrivateMem = true;
    } else if (as->name() == "instructions") {

      ImemSize = (as->end() + 1) * as->width();
    }
  }

  int segment_size = DmemSize > ImemSize ? DmemSize : ImemSize;

  int DmemStart, CQStart;
  if (relativeAddressing) {
    DmemStart = 0;
    CQStart = 0;
  } else {
    CQStart = 2 * segment_size;
    DmemStart = 3 * segment_size;
  }

  if (!hasPrivateMem) {
    // No private mem, so the latter half of the dmem is reserved for it
    int fallback_mem_size = pocl_get_int_option("POCL_ALMAIF_PRIVATE_MEM_SIZE",
                                                ALMAIF_DEFAULT_PRIVATE_MEM_SIZE);
    DmemSize -= fallback_mem_size;
    POCL_MSG_PRINT_ALMAIF(
        "Almaif: No separate private mem found. Setting it to %d\n",
        fallback_mem_size);
  }
  if (sharedDataAndCq) {
    // No separate Cq so reserve small slice of dmem for it
    CQSize = 4 * AQL_PACKET_LENGTH;
    DmemSize -= CQSize;
    CQStart = DmemStart + DmemSize;
  }

  int ImemStart = 0;

  if (!relativeAddressing) {
    unsigned default_baseaddress = 0x40000000; // TODO get from env variable
    CQStart += default_baseaddress;
    DmemStart += default_baseaddress;
  }

  memset(ControlRegisters_, 0, ALMAIF_DEFAULT_CTRL_SIZE);

  ControlRegisters_[ALMAIF_INFO_DEV_CLASS / 4] = 0;
  ControlRegisters_[ALMAIF_INFO_DEV_ID / 4] = 0;
  ControlRegisters_[ALMAIF_INFO_IF_TYPE / 4] = 3;
  ControlRegisters_[ALMAIF_INFO_CORE_COUNT / 4] = 1;
  ControlRegisters_[ALMAIF_INFO_CTRL_SIZE / 4] = 1024;
  ControlRegisters_[ALMAIF_INFO_IMEM_SIZE / 4] = ImemSize;
  ControlRegisters_[ALMAIF_INFO_IMEM_START_LOW / 4] = ImemStart;
  ControlRegisters_[ALMAIF_INFO_CQMEM_SIZE_LOW / 4] = CQSize;
  ControlRegisters_[ALMAIF_INFO_CQMEM_START_LOW / 4] = CQStart;
  ControlRegisters_[ALMAIF_INFO_DMEM_SIZE_LOW / 4] = DmemSize;
  ControlRegisters_[ALMAIF_INFO_DMEM_START_LOW / 4] = DmemStart;
  ControlRegisters_[ALMAIF_INFO_FEATURE_FLAGS_LOW / 4] =
      (relativeAddressing) ? 0 : 1;
  ControlRegisters_[ALMAIF_INFO_PTR_SIZE / 4] = 4;
}
