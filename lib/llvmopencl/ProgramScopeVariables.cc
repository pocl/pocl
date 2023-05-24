// takes care of program scope variables:
//  * turns all references to them to references of a special extern buffer
//    _pocl_gvar_buffer + offset into that
//  * that extern buffer variable is replaced by actual storage in Workgroup.cc
//  * creates an initializer kernel, called "pocl.gvar.init"
//  * adds program.bc module metadata "program.scope.var.size" - this is
//    required for later loads of poclbinary
//
// Copyright (c) 2022 Michal Babej / Intel Finland Oy
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

const char *PoclGVarPrefix = "_pocl_gvar_";
const char *PoclGVarBufferName = "_pocl_gvar_buffer";
const char *PoclGVarMDName = "program.scope.var.size";

#include "pocl.h"

#ifndef LLVM_OLDER_THAN_14_0

#include <iostream>
#include <map>
#include <random>
#include <set>
#include <thread>

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wno-maybe-uninitialized"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#pragma GCC diagnostic pop

#define DEBUG_TYPE "pocl-program-scope-vars"

// #define POCL_DEBUG_PROGVARS

#include "LLVMUtils.h"
#include "ProgramScopeVariables.h"

#include "pocl_llvm_api.h"
#include "pocl_spir.h"

using namespace llvm;

namespace pocl {

using GVarSetT = std::set<GlobalVariable *>;
using Const2InstMapT = std::map<Constant *, Instruction *>;
using GVarUlongMapT = std::map<GlobalVariable *, uint64_t>;

static void findInstructionUsesImpl(Use &U, std::vector<Use *> &Uses,
                                    std::set<Use *> &Visited) {
  if (Visited.count(&U))
    return;
  Visited.insert(&U);

  assert(isa<Constant>(*U));
  if (isa<Instruction>(U.getUser())) {
    Uses.push_back(&U);
    return;
  }
  if (isa<Constant>(U.getUser())) {
    for (auto &U : U.getUser()->uses())
      findInstructionUsesImpl(U, Uses, Visited);
    return;
  }

  // Catch other user kinds - we may need to process them (somewhere but not
  // here).
  llvm_unreachable("Unexpected user kind.");
}

// Return list of non-constant leaf use edges whose users are instructions.
static std::vector<Use *> findInstructionUses(GlobalVariable *GVar) {
  std::vector<Use *> Uses;
  std::set<Use *> Visited;
  for (auto &U : GVar->uses())
    findInstructionUsesImpl(U, Uses, Visited);
  return Uses;
}

// Returns a constant expression rewritten as instructions if needed.
// Global variable references found in the GVarMap are replaced with a load from
// the mapped pointer value.  New instructions will be added at Builder's
// current insertion point.
// TODO this likely doesn't handle all cases
static Value *expandConstant(Constant *C, IRBuilder<> &Builder,
                             Value *GVarBuffer, Type *GVarBufferTy,
                             GVarSetT &GVarSet, // for replacements
                             GVarUlongMapT &GVarOffsets,
                             Const2InstMapT &InsnCache) {
  if (InsnCache.count(C))
    return InsnCache[C];

  if (isa<ConstantData>(C))
    return C;

  if (isa<ConstantAggregate>(C))
    return C;

  if (GlobalVariable *GVar = dyn_cast<GlobalVariable>(C)) {
    StringRef GVarName = (GVar->hasName() ? GVar->getName() : "_unknown");
    if (GVarSet.count(GVar)) {
      LLVM_DEBUG(dbgs() << "expanding GVAR: " << GVarName);
      // Replace with pointer load. All constant expressions depending
      // on this will be rewritten as instructions.
      uint64_t GVOffset = GVarOffsets[GVar];
      Type *I64Ty = Type::getInt64Ty(C->getContext());
      SmallVector<Value *, 2> Indices{ConstantInt::get(I64Ty, 0),
                                      ConstantInt::get(I64Ty, GVOffset)};
      Value *GVarPtrWithOffset =
          Builder.CreateGEP(GVarBufferTy, GVarBuffer, Indices,
                            Twine{"_pocl_gvar_with_offset_", GVarName});
      // TODO addrspacecast
      GVarPtrWithOffset =
          Builder.CreateBitCast(GVarPtrWithOffset, GVar->getType());
      Instruction *I = cast<Instruction>(GVarPtrWithOffset);
      InsnCache[GVar] = I;
      return I;
    } else {
      LLVM_DEBUG(dbgs() << "NOT expanding GVAR: " << GVarName);
      return GVar;
    }
  }

  if (auto *CE = dyn_cast<ConstantExpr>(C)) {
    SmallVector<Value *, 4> Ops; // Collect potentially expanded operands.
    bool AnyOpExpanded = false;
    for (Value *Op : CE->operand_values()) {
      Value *V = expandConstant(cast<Constant>(Op), Builder, GVarBuffer,
                                GVarBufferTy, GVarSet, GVarOffsets, InsnCache);
      Ops.push_back(V);
      AnyOpExpanded |= !isa<Constant>(V);
    }

    if (!AnyOpExpanded)
      return CE;

    auto *AsInsn = Builder.Insert(CE->getAsInstruction());
    // Replace constant operands with expanded ones.
    for (auto &U : AsInsn->operands())
      U.set(Ops[U.getOperandNo()]);
    InsnCache[CE] = AsInsn;
    return AsInsn;
  }

  llvm_unreachable("Unexpected constant kind.");
}

// creates a [GEP+store initializer] for the hidden initializer kernel
static void addGlobalVarInitInstr(GlobalVariable *OriginalGVarDef,
                                  LLVMContext &Ctx, IRBuilder<> &Builder,
                                  Value *GVarBuffer,  // ptr %_pocl_gvar_buffer_load
                                  Type *GVarBufferTy, // [N x i8]
                                  unsigned DeviceGlobalAS,
                                  size_t GVarBufferOffset,
                                  GVarSetT &GVarSet, // for replacements
                                  GVarUlongMapT &GVarOffsets,
                                  Const2InstMapT &Cache) {

  //    <GVar> = <GVarBufferPtr + offset>
  //    *<GVar> = <initializer>;
  assert(OriginalGVarDef->hasInitializer());

#ifdef POCL_DEBUG_PROGVARS
  std::cerr << "@@@ addGlobalVarInitInstr FOR: "
            << OriginalGVarDef->getName().str() << "\n";
  std::cerr << "@@@ addGlobalVarInitInstr INITIALIZER: ";
  OriginalGVarDef->getInitializer()->dump();
  std::cerr << "\n";
#endif

  // %_pocl_gvar_with_offset_GVAR = getelementptr [N x i8], ptr %_pocl_gvar_buffer_load, i64 0, i64 GVAR_OFFSET
  Type *I64Ty = Type::getInt64Ty(Ctx);
  SmallVector<Value *, 2> Indices{ConstantInt::get(I64Ty, 0),
                                  ConstantInt::get(I64Ty, GVarBufferOffset)};
  Value *GVarPtrWithOffset = Builder.CreateGEP(
      GVarBufferTy, GVarBuffer, Indices,
      Twine{"_pocl_gvar_with_offset_", OriginalGVarDef->getName()});

  // gvar is alvays pointer
  PointerType *OrigGVarPTy = OriginalGVarDef->getType();

  // AScast from DeviceGlobalAS to use's AS
  // %0 = addrspacecast ptr %_pocl_gvar_with_offset_GVAR to ptr addrspace(1)
  if (OrigGVarPTy->getAddressSpace() != DeviceGlobalAS)
    GVarPtrWithOffset = Builder.CreateAddrSpaceCast(
        GVarPtrWithOffset,
        PointerType::get(GVarBufferTy, OrigGVarPTy->getAddressSpace()));
// bitcast to final pointer type if needed
#ifndef LLVM_OPAQUE_POINTERS
  GVarPtrWithOffset = Builder.CreateBitCast(GVarPtrWithOffset, OrigGVarPTy);
#endif

  // Initializers are constant expressions. If they have references to a global
  // variables we are going to replace with load instructions we need to rewrite
  // the constant expression as instructions.
  Value *Init =
      expandConstant(OriginalGVarDef->getInitializer(), Builder, GVarBuffer,
                     GVarBufferTy, GVarSet, GVarOffsets, Cache);

  // store [Z x i8] initializer, ptr addrspace(1) %0, align 1
  Builder.CreateStore(Init, GVarPtrWithOffset);
}



// walks through a Module's global variables,
// determines which ones are OpenCL program-scope variables
// and checks all of those have definitions
static bool areAllGvarsDefined(Module *Program, std::string &log,
                               GVarSetT &GVarSet, unsigned DeviceLocalAS) {

  bool FoundAllReferences = true;

  for (GlobalVariable &GVar : Program->globals()) {

    if (isProgramScopeVariable(GVar, DeviceLocalAS)) {

      assert(GVar.hasName());
      // adding GV declarations to the module also changes
      // the global iteration to include them
      if (GVarSet.count(&GVar) != 0)
        continue;

      if (GVar.isDeclaration()) {
        log.append("Undefined reference for program scope variable: ");
        log.append(GVar.getName().data());
        log.append("\n");
        FoundAllReferences = false;
      } else {
        GVarSet.insert(&GVar);
        // std::cerr << "**************************\n";
        // GVar.dump();
        // std::cerr << "**************************\n";
      }
    }
  }

  return FoundAllReferences;
}

// Emit a kernel named "pocl.gvar.init" for initialing global variables
static void emitInitializeKernel(Module *Program, LLVMContext &Ctx,
                                 GlobalVariable *GVarBufferPtr,
                                 PointerType *GVarBufferTy,
                                 unsigned DeviceGlobalAS,
                                 Type *GVarBufferArrayTy, GVarSetT &GVarSet,
                                 GVarUlongMapT &GVarOffsets,
                                 std::string &Log) {
  Function *GVarInitF = cast<Function>(
      Program
          ->getOrInsertFunction(
              POCL_GVAR_INIT_KERNEL_NAME,
              FunctionType::get(Type::getVoidTy(Ctx), {} /* ArgTypes */, false))
          .getCallee());
  GVarInitF->setCallingConv(CallingConv::SPIR_KERNEL);
  GVarInitF->setDSOLocal(true);
  // TODO copy attributes

  assert(GVarInitF->empty() && "Function name clash?");

  Const2InstMapT Cache;
  IRBuilder<> IrBuilder(BasicBlock::Create(Ctx, "entry", GVarInitF));
  Value *GVarBuffer = IrBuilder.CreateLoad(GVarBufferTy, GVarBufferPtr,
                                           "_pocl_gvar_buffer_load");

  for (GlobalVariable *GVar : GVarSet) {
    addGlobalVarInitInstr(GVar, Ctx, IrBuilder,
                          GVarBuffer,        // GVarBuffer Ptr
                          GVarBufferArrayTy, // GVarBuffer type
                          DeviceGlobalAS,
                          GVarOffsets[GVar], // GVarBuffer Offset
                          GVarSet,           // for replacements
                          GVarOffsets,
                          Cache);
  }

  // add return
  IrBuilder.CreateRetVoid();

  // add OpenCL metadata. Empty because 0 arguments
  MDTuple *EmptyMD = MDNode::get(Ctx, {});
  GVarInitF->setMetadata("kernel_arg_addr_space", EmptyMD);
  GVarInitF->setMetadata("kernel_arg_access_qual", EmptyMD);
  GVarInitF->setMetadata("kernel_arg_type", EmptyMD);
  GVarInitF->setMetadata("kernel_arg_base_type", EmptyMD);
  GVarInitF->setMetadata("kernel_arg_type_qual", EmptyMD);
  GVarInitF->setMetadata("kernel_arg_name", EmptyMD);

  // at this point, the initializers have been copied into the kernel;
  // remove the initializers from GVars. Doing this prevents the later
  // replaceGVarUses getting confused if a GVar references other GVar
  // in its initializer value
  for (GlobalVariable *GVar : GVarSet) {
    GVar->setInitializer(UndefValue::get(GVar->getValueType()));
  }
}

// for a set of program scope variables,
// calculate their offsets & sizes for later replacement with
// indexing into a single large buffer
// @returns the total size of all variables
static size_t calculateOffsetsSizes(GVarUlongMapT &GVarOffsets,
                                    const DataLayout &DL, GVarSetT &GVarSet) {
  GVarUlongMapT GVarSizes;

  // offset into the storage buffer for all of this program's global variables
  size_t CurrentOffset = 0;

  for (GlobalVariable *GVar : GVarSet) {
    assert(GVar->hasInitializer());

    // if the current offset into the buffer is not aligned enough, fix it
#ifdef LLVM_OLDER_THAN_15_0
    uint64_t GVarAlign = GVar->getAlignment();
#else
    Align GVarA = GVar->getAlign().valueOrOne();
    uint64_t GVarAlign = GVarA.value();
#endif

    if (GVarAlign > 0 && CurrentOffset % GVarAlign) {
      CurrentOffset |= (GVarAlign - 1);
      ++CurrentOffset;
    }
    GVarOffsets[GVar] = CurrentOffset;

    // add to the offset the required amount of storage for the global variable
    TypeSize GVSize = DL.getTypeAllocSize(GVar->getValueType());
    assert(GVSize.isScalable() == false);
    GVarSizes[GVar] = GVSize.getFixedValue();
    CurrentOffset += GVarSizes[GVar];

#ifdef POCL_DEBUG_PROGVARS
    std::cerr << "@@@ GlobalVar: " << GVar->getName().str()
              << "\n   OFFSET: " << GVarOffsets[GVar]
              << "\n   SIZE: " << GVarSizes[GVar] << "\n";
#endif
  }

  size_t TotalSize = CurrentOffset;
  return TotalSize;
}

// emit the GEP+casts for a replacement of GVar with [GVarBuffer+Offset]
static Value *loadGVarFromBuffer(Instruction *GVarBuffer,
                                 Type *GVarBufferArrayTy,
                                 unsigned DeviceGlobalAS,
                                 GlobalVariable *GVar, IRBuilder<> &Builder,
                                 uint64_t Offset) {

  Type *I64Ty = Type::getInt64Ty(GVarBuffer->getContext());
  // gvar is alvays pointer
  PointerType *GVarPTy = GVar->getType();
  Value *V;

  if (Offset == 0) {
    V = GVarBuffer;

    // AScast from DeviceGlobalAS to use's AS
    if (GVarPTy->getAddressSpace() != DeviceGlobalAS)
      V = Builder.CreateAddrSpaceCast(
          V, PointerType::get(GVar->getType(), GVarPTy->getAddressSpace()));
#ifndef LLVM_OPAQUE_POINTERS
    V = Builder.CreateBitCast(V, GVar->getType());
#endif
  } else {
    SmallVector<Value *, 2> Indices{ConstantInt::get(I64Ty, 0),
                                    ConstantInt::get(I64Ty, Offset)};
    V = Builder.CreateGEP(GVarBufferArrayTy, GVarBuffer, Indices);

    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
      Instruction *VI = CE->getAsInstruction();
      VI->insertBefore(&*Builder.GetInsertPoint());
      V = VI;
    }

    // AScast from DeviceGlobalAS to use's AS
    if (GVarPTy->getAddressSpace() != DeviceGlobalAS)
      V = Builder.CreateAddrSpaceCast(
          V, PointerType::get(GVar->getType(), GVarPTy->getAddressSpace()));
#ifndef LLVM_OPAQUE_POINTERS
    V = Builder.CreateBitCast(V, GVar->getType());
#endif
  }

#ifdef POCL_DEBUG_PROGVARS
  std::cerr << "@@@ LOAD of GlobalVar: " << GVar->getName().str()
            << " REPLACED WITH: \n";
  V->dump();
#endif

  return V;
}

static void getInstUsers(ConstantExpr *CE,
                         SmallVector<Instruction *, 4> &Users) {
  for (Value *U : CE->users()) {
    if (Instruction *I = dyn_cast<Instruction>(U)) {
      Users.push_back(I);
    }
    if (ConstantExpr *SubCE = dyn_cast<ConstantExpr>(U)) {
      getInstUsers(SubCE, Users);
    }
  }
}

static void breakConstantExprs(const GVarSetT &GVars) {
  for (GlobalVariable *GV : GVars) {
    for (Value *U : GV->users()) {
      ConstantExpr *CE = dyn_cast<ConstantExpr>(U);
      if (!CE)
        continue;
      SmallVector<Instruction *, 4> IUsers;
      getInstUsers(CE, IUsers);
      for (Instruction *I : IUsers) {
        convertConstantExprsToInstructions(I, CE);
      }
    }
  }
}

// replaces program scope variables with [GVarBuffer+Offset] combos.
static void replaceGlobalVariableUses(GVarSetT &GVarSet,
                                      GVarUlongMapT &GVarOffsets,
                                      GlobalVariable *GVarBufferPtr,
                                      PointerType *GVarBufferTy,
                                      Type *GVarBufferArrayTy,
                                      unsigned DeviceGlobalAS) {
  using KeyT = std::pair<Function *, GlobalVariable *>;
  std::map<KeyT, Instruction *> GVar2InsnCache;
  std::map<Function *, Instruction *> GVarBufferCache;
  std::map<Function *, std::unique_ptr<IRBuilder<>>> Fn2Builder;

  auto getBuilder = [&Fn2Builder](Function *F) -> IRBuilder<> & {
    auto &BuilderPtr = Fn2Builder[F];
    if (!BuilderPtr) {
      auto &E = F->getEntryBlock();
      auto InsPt = E.getFirstInsertionPt();
      // Put insertion point after allocas. SPIRV-LLVM translator panics (or at
      // least used to) if all allocas are not put in the entry block as the
      // first instructions.
      while (isa<AllocaInst>(*InsPt))
        InsPt = std::next(InsPt);
      BuilderPtr = std::make_unique<IRBuilder<>>(&E, InsPt);
    }
    return *BuilderPtr;
  };

  for (GlobalVariable *GVar : GVarSet) {
    LLVM_DEBUG(dbgs() << "Replacing: " << GVar->getName() << "\n";
               dbgs() << "   with: load from" << GVarBufferPtr->getName()
                      << "\n";);
    assert(GVarOffsets.find(GVar) != GVarOffsets.end());
    uint64_t Offset = GVarOffsets[GVar];

    for (auto *U : findInstructionUses(GVar)) {

      auto *IUser = cast<Instruction>(U->getUser());
      Function *FnUser = IUser->getParent()->getParent();
      IRBuilder<> &Builder = getBuilder(FnUser);
      LLVM_DEBUG(dbgs() << "in user: "; IUser->print(dbgs()); dbgs() << "\n";);

#ifdef POCL_DEBUG_PROGVARS
      std::cerr << " REPLACING GVAR USE: " << GVar->getName().str() << "\n";
      IUser->dump();
#endif

      Instruction *GVarBuffer = GVarBufferCache[FnUser];
      if (!GVarBuffer) {
        GVarBuffer = Builder.CreateLoad(GVarBufferTy, GVarBufferPtr,
                                        "_pocl_gvar_buffer_load");
        GVarBufferCache[FnUser] = GVarBuffer;
      }
      assert(GVarBuffer);

      auto Key = std::make_pair(FnUser, GVar);
      Value *GVarLoad = GVar2InsnCache[Key];
      if (!GVarLoad) {
        GVarLoad = loadGVarFromBuffer(GVarBuffer, GVarBufferArrayTy,
                                      DeviceGlobalAS, GVar, Builder, Offset);
        Instruction *I = dyn_cast<Instruction>(GVarLoad);
        if (I)
          GVar2InsnCache[Key] = I;
      }
      U->set(GVarLoad);
    }
  }
}

// erases program scope variables after they've been replaced
static void eraseMappedGlobalVariables(GVarSetT &GVarSet) {
  for (GlobalVariable *GVar : GVarSet) {
    if (GVar->hasNUses(0) ||
        // There might still be constantExpr users but no instructions should
        // depend on them.
        findInstructionUses(GVar).size() == 0) {
      GVar->replaceAllUsesWith(UndefValue::get(GVar->getType()));
      GVar->eraseFromParent();
    } else
      // A non-instruction and non-constantExpr user?
      llvm_unreachable("Original variable still has uses!");
  }
}

} // namespace pocl

using namespace pocl;

int runProgramScopeVariablesPass(
    Module *Program,
    unsigned DeviceGlobalAS, // the Target Global AS, not SPIR AS
    unsigned DeviceLocalAS,  // the Target Local AS, not SPIR AS
    size_t &TotalGVarSize, std::string &Log) {

  assert(Program);
  GVarSetT GVarSet;
  GVarUlongMapT GVarOffsets;
  LLVMContext &Ctx = Program->getContext();
  const DataLayout &DL = Program->getDataLayout();

  if (!areAllGvarsDefined(Program, Log, GVarSet, DeviceLocalAS))
    return -1;

  if (GVarSet.empty()) {
    return 0;
  }

  breakConstantExprs(GVarSet);

  TotalGVarSize = calculateOffsetsSizes(GVarOffsets, DL, GVarSet);

  Type *GVarBufferArrayTy = ArrayType::get(Type::getInt8Ty(Ctx), TotalGVarSize);
  PointerType *GVarBufferTy =
      PointerType::get(GVarBufferArrayTy, DeviceGlobalAS);
  Program->getOrInsertGlobal(PoclGVarBufferName, GVarBufferTy);

  GlobalVariable *GVarBuffer = Program->getNamedGlobal(PoclGVarBufferName);
  assert(GVarBuffer);

  setModuleIntMetadata(Program, PoclGVarMDName, TotalGVarSize);

  emitInitializeKernel(Program, Ctx, GVarBuffer, GVarBufferTy, DeviceGlobalAS,
                       GVarBufferArrayTy, GVarSet, GVarOffsets, Log);
  replaceGlobalVariableUses(GVarSet, GVarOffsets, GVarBuffer, GVarBufferTy,
                            GVarBufferArrayTy, DeviceGlobalAS);

  eraseMappedGlobalVariables(GVarSet);

  raw_string_ostream OS(Log);
  bool BrokenDebug = false;
  if (verifyModule(*Program, &OS, &BrokenDebug))
    POCL_MSG_ERR("LLVM Module verifier returned errors:\n");
  OS.flush();

  return 0;
}

#endif
