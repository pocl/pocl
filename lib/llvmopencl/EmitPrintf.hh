//===- AMDGPUEmitPrintf.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility function to lower a printf call into a series of device
// library calls on the AMDGPU target.
//
//===----------------------------------------------------------------------===//

#ifndef POCL_AUTOMATIC_LOCALS_H
#define POCL_AUTOMATIC_LOCALS_H

#include "config.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

namespace pocl {

struct PrintfCallFlags {
  unsigned PrintfBufferAS = 0; // AS of the printf buffer
  bool IsBuffered = true;      // Buffered mode - store arguments to a buffer;
                          // Non-buffered mode - call predefined functions with
                          // arguments
  bool DontAlign = true;    // true = do not add extra alignment; false = aligns
                            // all arguments to 8 bytes (in buffered mode)
  bool FlushBuffer = false; // in Buffered mode, call a function to flush the
                            // buffer immediately after storing the arguments
  bool StorePtrInsteadOfMD5 = false; // if the formatstring is Constant, pass
                                     // it as pointer instead of MD5
  bool AlwaysStoreFmtPtr = false; // if true, always store the FmtStr directly
                                  // in the buffer
  bool ArgPromotionCharShort = true; // Clang promotes char/short
  bool ArgPromotionChar2 = false;    // Clang promotes char2 to int
  bool ArgPromotionFloat = true;     // Clang promotes float to double
  bool IsBigEndian = false;          // device is big-endian
  bool Pointers32Bit = false;        // device's pointer width is 32bits
};

llvm::Value *emitPrintfCall(llvm::IRBuilder<> &Builder,
                            llvm::SmallVector<llvm::Value *> &Args,
                            PrintfCallFlags Flags);

} // namespace pocl

#endif // LLVM_TRANSFORMS_UTILS_AMDGPUEMITPRINTF_H
