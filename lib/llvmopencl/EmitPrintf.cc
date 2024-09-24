//===- AMDGPUEmitPrintf.cpp -----------------------------------------------===//
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
// WARNING: This file knows about certain library functions. It recognizes them
// by name, and hardwires knowledge of their semantics.
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/SparseBitVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/DataExtractor.h>
#include <llvm/Support/MD5.h>
#include <llvm/Support/MathExtras.h>

#include <iostream>

#include "EmitPrintf.hh"
#include "printf_buffer.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-emit-printf"

static Value *fitArgInto64Bits(IRBuilder<> &Builder, Value *Arg) {
  auto Int64Ty = Builder.getInt64Ty();
  auto Ty = Arg->getType();

  if (auto IntTy = dyn_cast<IntegerType>(Ty)) {
    switch (IntTy->getBitWidth()) {
    case 32:
      return Builder.CreateZExt(Arg, Int64Ty);
    case 64:
      return Arg;
    }
  }

  if (Ty->getTypeID() == Type::DoubleTyID) {
    return Builder.CreateBitCast(Arg, Int64Ty);
  }

  if (isa<PointerType>(Ty)) {
    return Builder.CreatePtrToInt(Arg, Int64Ty);
  }

  llvm_unreachable("unexpected type");
}

static Value *callPrintfBegin(IRBuilder<> &Builder, Value *Version) {
  auto Int64Ty = Builder.getInt64Ty();
  auto M = Builder.GetInsertBlock()->getModule();
  auto Fn = M->getOrInsertFunction("__ockl_printf_begin", Int64Ty, Int64Ty);
  return Builder.CreateCall(Fn, Version);
}

static Value *callAppendArgs(IRBuilder<> &Builder, Value *Desc, int NumArgs,
                             Value *Arg0, Value *Arg1, Value *Arg2, Value *Arg3,
                             Value *Arg4, Value *Arg5, Value *Arg6,
                             bool IsLast) {
  auto Int64Ty = Builder.getInt64Ty();
  auto Int32Ty = Builder.getInt32Ty();
  auto M = Builder.GetInsertBlock()->getModule();
  auto Fn = M->getOrInsertFunction("__ockl_printf_append_args", Int64Ty,
                                   Int64Ty, Int32Ty, Int64Ty, Int64Ty, Int64Ty,
                                   Int64Ty, Int64Ty, Int64Ty, Int64Ty, Int32Ty);
  auto IsLastValue = Builder.getInt32(IsLast);
  auto NumArgsValue = Builder.getInt32(NumArgs);
  return Builder.CreateCall(Fn, {Desc, NumArgsValue, Arg0, Arg1, Arg2, Arg3,
                                 Arg4, Arg5, Arg6, IsLastValue});
}

static Value *appendArg(IRBuilder<> &Builder, Value *Desc, Value *Arg,
                        bool IsLast) {
  auto Arg0 = fitArgInto64Bits(Builder, Arg);
  auto Zero = Builder.getInt64(0);
  return callAppendArgs(Builder, Desc, 1, Arg0, Zero, Zero, Zero, Zero, Zero,
                        Zero, IsLast);
}

// The device library does not provide strlen, so we build our own loop
// here. While we are at it, we also include the terminating null in the length.
static Value *getStrlenWithNull(IRBuilder<> &Builder, Value *Str) {
  auto *Prev = Builder.GetInsertBlock();
  Module *M = Prev->getModule();

  auto CharZero = Builder.getInt8(0);
  auto One = Builder.getInt64(1);
  auto Zero = Builder.getInt64(0);
  auto Int64Ty = Builder.getInt64Ty();

  // The length is either zero for a null pointer, or the computed value for an
  // actual string. We need a join block for a phi that represents the final
  // value.
  //
  //  Strictly speaking, the zero does not matter since
  // __ockl_printf_append_string_n ignores the length if the pointer is null.
  BasicBlock *Join = nullptr;
  if (Prev->getTerminator()) {
    Join = Prev->splitBasicBlock(Builder.GetInsertPoint(), "strlen.join");
    Prev->getTerminator()->eraseFromParent();
  } else {
    Join =
        BasicBlock::Create(M->getContext(), "strlen.join", Prev->getParent());
  }
  BasicBlock *While = BasicBlock::Create(M->getContext(), "strlen.while",
                                         Prev->getParent(), Join);
  BasicBlock *WhileDone = BasicBlock::Create(
      M->getContext(), "strlen.while.done", Prev->getParent(), Join);

  ConstantPointerNull *NullPtr =
      ConstantPointerNull::get(cast<PointerType>(Str->getType()));

  // Emit an early return for when the pointer is null.
  Builder.SetInsertPoint(Prev);
  auto CmpNull = Builder.CreateICmpEQ(Str, NullPtr);
  Builder.CreateCondBr(CmpNull, Join, While);

  // Entry to the while loop.
  Builder.SetInsertPoint(While);

  auto PtrPhi = Builder.CreatePHI(Str->getType(), 2);
  PtrPhi->addIncoming(Str, Prev);
  auto PtrNext = Builder.CreateGEP(Builder.getInt8Ty(), PtrPhi, One);
  PtrPhi->addIncoming(PtrNext, While);

  // Condition for the while loop.
  auto Data = Builder.CreateLoad(Builder.getInt8Ty(), PtrPhi);
  auto Cmp = Builder.CreateICmpEQ(Data, CharZero);
  Builder.CreateCondBr(Cmp, WhileDone, While);

  // Add one to the computed length.
  Builder.SetInsertPoint(WhileDone, WhileDone->begin());
  auto Begin = Builder.CreatePtrToInt(Str, Int64Ty);
  auto End = Builder.CreatePtrToInt(PtrPhi, Int64Ty);
  auto Len = Builder.CreateSub(End, Begin);
  Len = Builder.CreateAdd(Len, One);

  // Final join.
  BranchInst::Create(Join, WhileDone);
  Builder.SetInsertPoint(Join, Join->begin());
  auto LenPhi = Builder.CreatePHI(Len->getType(), 2);
  LenPhi->addIncoming(Len, WhileDone);
  LenPhi->addIncoming(Zero, Prev);

  return LenPhi;
}

static Value *callAppendStringN(IRBuilder<> &Builder, Value *Desc, Value *Str,
                                Value *Length, bool isLast) {
  auto Int64Ty = Builder.getInt64Ty();
  auto IsLastInt32 = Builder.getInt32(isLast);
  auto M = Builder.GetInsertBlock()->getModule();
  auto Fn = M->getOrInsertFunction("__ockl_printf_append_string_n", Int64Ty,
                                   Desc->getType(), Str->getType(),
                                   Length->getType(), IsLastInt32->getType());
  return Builder.CreateCall(Fn, {Desc, Str, Length, IsLastInt32});
}

static Value *appendString(IRBuilder<> &Builder, Value *Desc, Value *Arg,
                           bool IsLast) {
  auto Length = getStrlenWithNull(Builder, Arg);
  return callAppendStringN(Builder, Desc, Arg, Length, IsLast);
}

static Value *processArg(IRBuilder<> &Builder, Value *Desc, Value *Arg,
                         bool SpecIsCString, bool IsLast) {
  if (SpecIsCString && isa<PointerType>(Arg->getType())) {
    return appendString(Builder, Desc, Arg, IsLast);
  }
  // If the format specifies a string but the argument is not, the frontend will
  // have printed a warning. We just rely on undefined behaviour and send the
  // argument anyway.
  return appendArg(Builder, Desc, Arg, IsLast);
}

// Scan the format string to locate all specifiers, and mark the ones that
// specify a string, i.e, the "%s" specifier with optional '*' characters.
static void locateCStrings(SparseBitVector<8> &BV, StringRef Str) {
  static const char ConvSpecifiers[] = "diouxXfFeEgGaAcspn";
  size_t SpecPos = 0;
  // Skip the first argument, the format string.
  unsigned ArgIdx = 1;

  while ((SpecPos = Str.find_first_of('%', SpecPos)) != StringRef::npos) {
    if (Str[SpecPos + 1] == '%') {
      SpecPos += 2;
      continue;
    }
    auto SpecEnd = Str.find_first_of(ConvSpecifiers, SpecPos);
    if (SpecEnd == StringRef::npos)
      return;
    auto Spec = Str.slice(SpecPos, SpecEnd + 1);
    ArgIdx += Spec.count('*');
    if (Str[SpecEnd] == 's') {
      BV.set(ArgIdx);
    }
    SpecPos = SpecEnd + 1;
    ++ArgIdx;
  }
}

// helper struct to package the string related data
struct StringData {
  StringRef Str;
  Value *RealSize = nullptr;
  Value *AlignedSize = nullptr;
  bool IsConst = true;

  StringData(StringRef ST, Value *RS, Value *AS, bool IC)
      : Str(ST), RealSize(RS), AlignedSize(AS), IsConst(IC) {}
};

// Calculates frame size required for current printf expansion and allocates
// space on printf buffer. Printf frame includes following contents
// [ ControlDWord , format string/Hash , Arguments (each aligned to 8 byte) ]
static Value *
callBufferedPrintfStart(IRBuilder<> &Builder, SmallVector<llvm::Value *> &Args,
                        Value *Fmt, unsigned PrintfBufferAS, bool SkipFmtStr,
                        bool DontAlign, SparseBitVector<8> &SpecIsCString,
                        SmallVectorImpl<StringData> &StringContents,
                        Value *&ArgSize, Value *&FmtStrLen) {
  Module *M = Builder.GetInsertBlock()->getModule();
  Value *NonConstStrLen = nullptr;
  Value *LenWithNull = nullptr;
  Value *LenWithNullAligned = nullptr;
  Value *TempAdd = nullptr;
  auto DL = M->getDataLayout();

  // First 4 bytes to be reserved for control dword
  size_t BufSize = 4;
  if (SkipFmtStr)
    // First 8 bytes of MD5 hash
    BufSize += 8;
  else {
    LenWithNull = getStrlenWithNull(Builder, Fmt);

    if (DontAlign) {
      NonConstStrLen = LenWithNull;
    } else {
      // Align the computed length to next 8 byte boundary
      TempAdd = Builder.CreateAdd(LenWithNull,
                                  ConstantInt::get(LenWithNull->getType(), 7U));
      NonConstStrLen = Builder.CreateAnd(
          TempAdd, ConstantInt::get(LenWithNull->getType(), ~7U));
    }

    StringContents.push_back(
        StringData(StringRef(), LenWithNull, NonConstStrLen, false));
    FmtStrLen = Builder.CreateTrunc(NonConstStrLen, Builder.getInt32Ty());
  }

  for (size_t i = 1; i < Args.size(); i++) {
    if (SpecIsCString.test(i)) {
      StringRef ArgStr;
      // handle a nullptr value given to %s argument by replacing with "(null)"
      // string
      if (Args[i]->getType()->isPointerTy() && isa<Constant>(Args[i])) {
        PointerType *PT = dyn_cast<PointerType>(Args[i]->getType());
        Constant *C = dyn_cast<Constant>(Args[i]);
        assert(PT);
        assert(C);
        if (C->isNullValue()) {
          GlobalVariable *Replace = M->getNamedGlobal("__pocl_nullptr_string");
          if (Replace == nullptr) {
            Constant *C =
                ConstantDataArray::getString(M->getContext(), "(null)", true);
            Replace = new GlobalVariable(*M, C->getType(), true,
                                         GlobalValue::PrivateLinkage, C,
                                         "__pocl_nullptr_string");
          }
          Args[i] = Replace;
        }
      }

      if (getConstantStringInfo(Args[i], ArgStr)) {
        uint64_t bufStrLen;
        if (DontAlign)
          bufStrLen = ArgStr.size() + 1;
        else
          bufStrLen = alignTo(ArgStr.size() + 1, 8);
        StringContents.push_back(
            StringData(ArgStr,
                       /* RealSize */ Builder.getInt64(ArgStr.size() + 1),
                       /*AlignedSize*/ Builder.getInt64(bufStrLen),
                       /*IsConst*/ true));
        BufSize += bufStrLen;
      } else {
        LenWithNull = getStrlenWithNull(Builder, Args[i]);

        if (DontAlign) {
          LenWithNullAligned = LenWithNull;
        } else {
          // Align the computed length to next 8 byte boundary
          TempAdd = Builder.CreateAdd(
              LenWithNull, ConstantInt::get(LenWithNull->getType(), 7U));
          LenWithNullAligned = Builder.CreateAnd(
              TempAdd, ConstantInt::get(LenWithNull->getType(), ~7U));
        }

        if (NonConstStrLen) {
          auto Val = Builder.CreateAdd(LenWithNullAligned, NonConstStrLen,
                                       "cumulativeAdd");
          NonConstStrLen = Val;
        } else
          NonConstStrLen = LenWithNullAligned;

        StringContents.push_back(
            StringData(StringRef(), LenWithNull, LenWithNullAligned, false));
      }
    } else {
      TypeSize AllocSize = DL.getTypeAllocSize(Args[i]->getType());
      llvm::AllocaInst *AI = dyn_cast<AllocaInst>(Args[i]);
      if (AI) {
        // byval arguments are passed as pointers/allocas.
        // get the actual size from the alloca
        // TBD: is this handling of allocas enough
#if LLVM_MAJOR > 15
        auto SSize = AI->getAllocationSize(DL)->getFixedValue();
#else
        auto SSize = AI->getAllocationSizeInBits(DL)->getFixedValue() / 8;
#endif
        AllocSize = TypeSize::get(SSize, false);
      }

      if (!DontAlign)
        // We end up expanding non string arguments to 8 bytes
        // (args smaller than 8 bytes)
        AllocSize = alignTo(AllocSize, 8);

      BufSize += AllocSize.getFixedValue();
    }
  }

  // calculate final size value to be passed to printf_alloc
  Value *SizeToReserve = ConstantInt::get(Builder.getInt64Ty(), BufSize, false);
  SmallVector<Value *, 1> Alloc_args;
  if (NonConstStrLen)
    SizeToReserve = Builder.CreateAdd(NonConstStrLen, SizeToReserve);

  ArgSize = Builder.CreateTrunc(SizeToReserve, Builder.getInt32Ty());
  Alloc_args.push_back(ArgSize);

  // call the printf_alloc function
  AttributeList Attr = AttributeList::get(
      Builder.getContext(), AttributeList::FunctionIndex, Attribute::NoUnwind);

  Type *Tys_alloc[1] = {Builder.getInt32Ty()};
  Type *PtrTy =
#ifdef LLVM_OPAQUE_POINTERS
      Builder.getPtrTy(PrintfBufferAS);
#else
      Builder.getInt8PtrTy(PrintfBufferAS);
#endif
  FunctionType *FTy_alloc = FunctionType::get(PtrTy, Tys_alloc, false);

  auto PrintfAllocFn =
      M->getOrInsertFunction(StringRef("__printf_alloc"), FTy_alloc, Attr);
  return Builder.CreateCall(PrintfAllocFn, Alloc_args, "printf_alloc_fn");
}

// Prepare constant string argument to push onto the buffer
static void processConstantStringArg(StringData *SD, IRBuilder<> &Builder,
                                     bool DontAlign,
                                     SmallVectorImpl<Value *> &WhatToStore) {
  std::string Str(SD->Str.str() + '\0');

  DataExtractor Extractor(Str, /*IsLittleEndian=*/true, 8);
  DataExtractor::Cursor Offset(0);
  while (Offset && Offset.tell() < Str.size()) {
    const uint64_t ReadSize = 4;
    uint64_t ReadNow = std::min(ReadSize, Str.size() - Offset.tell());
    uint64_t ReadBytes = 0;
    switch (ReadNow) {
    default:
      llvm_unreachable("min(4, X) > 4?");
    case 1:
      ReadBytes = Extractor.getU8(Offset);
      break;
    case 2:
      ReadBytes = Extractor.getU16(Offset);
      break;
    case 3:
      ReadBytes = Extractor.getU24(Offset);
      break;
    case 4:
      ReadBytes = Extractor.getU32(Offset);
      break;
    }
    cantFail(Offset.takeError(), "failed to read bytes from constant array");

    APInt IntVal(8 * ReadNow, ReadBytes);

    // TODO: Should not bother aligning up.
    if (!DontAlign && ReadNow < ReadSize)
      IntVal = IntVal.zext(8 * ReadSize);

    Type *IntTy = Type::getIntNTy(Builder.getContext(), IntVal.getBitWidth());
    WhatToStore.push_back(ConstantInt::get(IntTy, IntVal));
  }
  if (!DontAlign) {
    // Additional padding for 8 byte alignment
    int Rem = (Str.size() % 8);
    if (Rem > 0 && Rem <= 4)
      WhatToStore.push_back(ConstantInt::get(Builder.getInt32Ty(), 0));
  }
}

static Value *processNonStringArg(Value *Arg, IRBuilder<> &Builder) {
  const DataLayout &DL = Builder.GetInsertBlock()->getModule()->getDataLayout();
  auto Ty = Arg->getType();

  if (auto IntTy = dyn_cast<IntegerType>(Ty)) {
    if (IntTy->getBitWidth() < 64) {
      return Builder.CreateZExt(Arg, Builder.getInt64Ty());
    }
  }

  if (Ty->isFloatingPointTy()) {
    if (DL.getTypeAllocSize(Ty) < 8) {
      return Builder.CreateFPExt(Arg, Builder.getDoubleTy());
    }
  }

  return Arg;
}

static StoreInst *createAlignedStore(IRBuilder<> &Builder, Value *Val,
                                     Value *Ptr, bool DontAlign) {
#ifndef LLVM_OPAQUE_POINTERS
  PointerType *PtrTy = cast<PointerType>(Ptr->getType());
  Ptr = Builder.CreateBitCast(
      Ptr, PointerType::get(Val->getType(), PtrTy->getAddressSpace()));
#endif
  if (DontAlign)
    return Builder.CreateAlignedStore(Val, Ptr, Align(1));
  else
    return Builder.CreateStore(Val, Ptr);
}

static void
callBufferedPrintfArgPush(IRBuilder<> &Builder,
                          SmallVector<llvm::Value *> &Args, Value *PtrToStore,
                          SparseBitVector<8> &SpecIsCString,
                          SmallVectorImpl<StringData> &StringContents,
                          bool SkipFmtStr, bool DontAlign) {
  Module *M = Builder.GetInsertBlock()->getModule();
  const DataLayout &DL = M->getDataLayout();
  auto StrIt = StringContents.begin();
  size_t i = SkipFmtStr ? 1 : 0;
  for (; i < Args.size(); i++) {
    SmallVector<Value *, 32> WhatToStore;
    if ((i == 0) || SpecIsCString.test(i)) {
      if (StrIt->IsConst && !DontAlign) {
        processConstantStringArg(StrIt, Builder, DontAlign, WhatToStore);
        StrIt++;
      } else {
        // This copies the contents of the string, however the next offset
        // is at aligned length, the extra space that might be created due
        // to alignment padding is not populated with any specific value
        // here. This would be safe as long as runtime is sync with
        // the offsets.
        Value *MemcpyLen =
            Builder.CreateTrunc(StrIt->RealSize, Builder.getInt32Ty());
        Builder.CreateMemCpy(PtrToStore, /*DstAlign*/ Align(1), Args[i],
                             /*SrcAlign*/ Args[i]->getPointerAlignment(DL),
                             MemcpyLen);

        Value *StoreSize = DontAlign ? StrIt->RealSize : StrIt->AlignedSize;
        PtrToStore = Builder.CreateInBoundsGEP(Builder.getInt8Ty(), PtrToStore,
                                               {StoreSize}, "PrintBuffNextPtr");
        LLVM_DEBUG(dbgs() << "inserting gep to the printf buffer:"
                          << *PtrToStore << '\n');

        // done with current argument, move to next
        StrIt++;
        continue;
      }
    } else {
      if (DontAlign)
        WhatToStore.push_back(Args[i]);
      else
        WhatToStore.push_back(processNonStringArg(Args[i], Builder));
    }

    for (Value *toStore : WhatToStore) {
      TypeSize AllocSize =
          M->getDataLayout().getTypeAllocSize(toStore->getType());
      StoreInst *StBuff = nullptr;
      llvm::AllocaInst *AI = dyn_cast<AllocaInst>(toStore);
      if (AI) {
        // byval arguments are passed as pointers/allocas. Creating a simple
        // store stores the pointer instead of the actual value; therefore we
        // need to use a memcpy. TBD: is this handling of allocas enough
#if LLVM_MAJOR > 15
        auto SSize = AI->getAllocationSize(DL)->getFixedValue();
#else
        auto SSize = AI->getAllocationSizeInBits(DL)->getFixedValue() / 8;
#endif
        AllocSize = TypeSize::get(SSize, false);
        Value *MemcpyLen = ConstantInt::get(Builder.getInt32Ty(), SSize);
        Builder.CreateMemCpy(PtrToStore, Align(1), AI, Align(1), MemcpyLen);
        StBuff = nullptr;
      } else {
        StBuff = createAlignedStore(Builder, toStore, PtrToStore, DontAlign);
        LLVM_DEBUG(dbgs() << "inserting store to printf buffer:" << *StBuff
                          << '\n');
        (void)StBuff;
      }
      PtrToStore = Builder.CreateConstInBoundsGEP1_32(
          Builder.getInt8Ty(), PtrToStore, AllocSize, "PrintBuffNextPtr");
      LLVM_DEBUG(dbgs() << "inserting gep to the printf buffer:" << *PtrToStore
                        << '\n');
    }
  }
}

Value *pocl::emitPrintfCall(IRBuilder<> &Builder,
                            SmallVector<llvm::Value *> &Args,
                            PrintfCallFlags Flags) {
  auto NumOps = Args.size();
  assert(NumOps >= 1);

  auto Fmt = Args[0];
  SparseBitVector<8> SpecIsCString;
  StringRef FmtStr;

  if (getConstantStringInfo(Fmt, FmtStr))
    locateCStrings(SpecIsCString, FmtStr);

  if (Flags.IsBuffered) {
    SmallVector<StringData, 8> StringContents;
    Module *M = Builder.GetInsertBlock()->getModule();
    LLVMContext &Ctx = Builder.getContext();
    auto Int8Ty = Builder.getInt8Ty();
    auto Int32Ty = Builder.getInt32Ty();
    bool SkipFmtStr = !FmtStr.empty() && !Flags.AlwaysStoreFmtPtr;

    Value *ArgSize = nullptr;
    Value *FmtStrLen = nullptr;
    // this truncates ArgSize to int32
    Value *StartPtr = callBufferedPrintfStart(
        Builder, Args, Fmt, Flags.PrintfBufferAS, SkipFmtStr, Flags.DontAlign,
        SpecIsCString, StringContents, ArgSize, FmtStrLen);

    // The buffered version still follows OpenCL printf standards for
    // printf return value, i.e 0 on success, -1 on failure.
    ConstantPointerNull *zeroIntPtr =
        ConstantPointerNull::get(cast<PointerType>(StartPtr->getType()));

    auto *Cmp = cast<ICmpInst>(Builder.CreateICmpNE(StartPtr, zeroIntPtr, ""));

    // split the remaining code (after the printf call) at this point
    // into its own "end" BB
    auto LastBB = Builder.GetInsertBlock();
    BasicBlock *End =
        LastBB->splitBasicBlock(Builder.GetInsertPoint(), "end.block");
    // splitBasicBlock adds an unconditional branch; remove it here, we will
    // add our own branch later.
    if (LastBB->getTerminator())
      LastBB->getTerminator()->eraseFromParent();

    BasicBlock *ArgPush = BasicBlock::Create(
        Ctx, "argpush.block", Builder.GetInsertBlock()->getParent());

    // if the allocation returned null, jump to end BB
    BranchInst::Create(ArgPush, End, Cmp, LastBB);

    Builder.SetInsertPoint(ArgPush);

    // Create controlDWord and store as the first entry, format as follows
    // Bit 0 (LSB) -> stream (1 if stderr, 0 if stdout,
    // Bit 1 -> constant format string (1 if constant)
    // Bit 2 -> char & short promotion to int
    // Bit 3 -> char2 promotion to int
    // Bit 4 -> float promotion to double
    // Bit 5 -> 1 = big-endian, 0 = little-endian
    // Bits 6-31 -> size
    // of printf data frame

    auto ControlDWord = Builder.CreateShl(
        ArgSize, Builder.getInt32(PRINTF_BUFFER_CTWORD_FLAG_BITS));
    uint32_t FlagsOrValue = 0;
    if (SkipFmtStr)
      FlagsOrValue |= PRINTF_BUFFER_CTWORD_SKIP_FMT_STR;
    if (Flags.ArgPromotionCharShort)
      FlagsOrValue |= PRINTF_BUFFER_CTWORD_CHAR_SHORT_PR;
    if (Flags.ArgPromotionChar2)
      FlagsOrValue |= PRINTF_BUFFER_CTWORD_CHAR2_PR;
    if (Flags.ArgPromotionFloat)
      FlagsOrValue |= PRINTF_BUFFER_CTWORD_FLOAT_PR;
    if (Flags.IsBigEndian)
      FlagsOrValue |= PRINTF_BUFFER_CTWORD_BIG_ENDIAN;

    if (FlagsOrValue)
      ControlDWord =
          Builder.CreateOr(ControlDWord, Builder.getInt32(FlagsOrValue));
    createAlignedStore(Builder, ControlDWord, StartPtr, Flags.DontAlign);

    Value *Ptr = Builder.CreateConstInBoundsGEP1_32(Int8Ty, StartPtr, 4);

    // Create MD5 hash for costant format string, push low 64 bits of the
    // same onto buffer and metadata.
    NamedMDNode *metaD = M->getOrInsertNamedMetadata("llvm.printf.fmts");
    if (SkipFmtStr) {
      if (Flags.StorePtrInsteadOfMD5) {
        createAlignedStore(Builder, Fmt, Ptr, Flags.DontAlign);
        Ptr = Builder.CreateConstInBoundsGEP1_32(Int8Ty, Ptr, 8);
      } else {
        MD5 Hasher;
        MD5::MD5Result Hash;
        Hasher.update(FmtStr);
        Hasher.final(Hash);

        // Try sticking to llvm.printf.fmts format, although we are not going to
        // use the ID and argument size fields while printing,
        std::string MetadataStr =
            "0:0:" + llvm::utohexstr(Hash.low(), /*LowerCase=*/true) + "," +
            FmtStr.str();
        MDString *fmtStrArray = MDString::get(Ctx, MetadataStr);
        MDNode *myMD = MDNode::get(Ctx, fmtStrArray);
        metaD->addOperand(myMD);

        createAlignedStore(Builder, Builder.getInt64(Hash.low()), Ptr,
                           Flags.DontAlign);
        Ptr = Builder.CreateConstInBoundsGEP1_32(Int8Ty, Ptr, 8);
      }
    } else {
      // Include a dummy metadata instance in case of only non constant
      // format string usage, This might be an absurd usecase but needs to
      // be done for completeness
      if (metaD->getNumOperands() == 0) {
        MDString *fmtStrArray =
            MDString::get(Ctx, "0:0:ffffffff,\"Non const format string\"");
        MDNode *myMD = MDNode::get(Ctx, fmtStrArray);
        metaD->addOperand(myMD);
      }
    }

    // Push The printf arguments onto buffer
    callBufferedPrintfArgPush(Builder, Args, Ptr, SpecIsCString, StringContents,
                              SkipFmtStr, Flags.DontAlign);

    // flush the buffer if requested by calling
    // void __printf_flush_buffer(void* buffer, uint32_t bytes);
    if (Flags.FlushBuffer) {
      SmallVector<Value *, 2> Alloc_args;
      Alloc_args.push_back(StartPtr);
      Alloc_args.push_back(ArgSize);

      AttributeList Attr =
          AttributeList::get(Builder.getContext(), AttributeList::FunctionIndex,
                             Attribute::NoUnwind);

      Type *PtrTy =
#ifdef LLVM_OPAQUE_POINTERS
          Builder.getPtrTy(Flags.PrintfBufferAS);
#else
          Builder.getInt8PtrTy(Flags.PrintfBufferAS);
#endif

      Type *Tys_alloc[2] = {PtrTy, Builder.getInt32Ty()};
      FunctionType *FTy_alloc =
          FunctionType::get(Builder.getVoidTy(), Tys_alloc, false);
      auto PrintfAllocFn = M->getOrInsertFunction(
          StringRef("__printf_flush_buffer"), FTy_alloc, Attr);
      Builder.CreateCall(PrintfAllocFn, Alloc_args);
    }

    // End block, returns -1 on failure
    BranchInst::Create(End, ArgPush);
#if LLVM_MAJOR > 17
    Builder.SetInsertPoint(End->getFirstInsertionPt());
#else
    Builder.SetInsertPoint(&*End->getFirstInsertionPt());
#endif
    return Builder.CreateSExt(Builder.CreateNot(Cmp), Int32Ty, "printf_result");
  }

  auto Desc = callPrintfBegin(Builder, Builder.getIntN(64, 0));
  Desc = appendString(Builder, Desc, Fmt, NumOps == 1);

  // FIXME: This invokes hostcall once for each argument. We can pack up to
  // seven scalar printf arguments in a single hostcall. See the signature of
  // callAppendArgs().
  for (unsigned int i = 1; i != NumOps; ++i) {
    bool IsLast = i == NumOps - 1;
    bool IsCString = SpecIsCString.test(i);
    Desc = processArg(Builder, Desc, Args[i], IsCString, IsLast);
  }

  return Builder.CreateTrunc(Desc, Builder.getInt32Ty());
}
