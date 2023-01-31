/* spirv_parser.cc - a light parser for SPIR-V binaries. Only parses enough to
 * get kernel function signatures and their argument metadata (types, sizes,
 * address spaces..)
 *
 * Copyright (c) 2021-22 CHIP-SPV developers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "spirv.hh"
#include "spirv_parser.hh"

#include "pocl_debug.h"

#define logWarn(...) POCL_MSG_WARN(__VA_ARGS__);
#define logError(...) POCL_MSG_ERR(__VA_ARGS__);
#define logTrace(...) POCL_MSG_PRINT_INFO(__VA_ARGS__);

typedef std::map<int32_t, std::shared_ptr<OCLFuncInfo>> OCLFuncInfoMap;

const std::string OpenCLStd{"OpenCL.std"};

class SPIRVtype {
protected:
  int32_t Id_;
  size_t Size_;

public:
  SPIRVtype(int32_t Id, size_t Size) : Id_(Id), Size_(Size) {}
  virtual ~SPIRVtype(){};
  virtual size_t size() { return Size_; }
  int32_t id() { return Id_; }
  virtual OCLType ocltype() = 0;
  virtual OCLSpace getAS() { return OCLSpace::Private; }
  virtual spv::AccessQualifier getImgAccess() {
    return spv::AccessQualifier::Max;
  }
};

typedef std::map<int32_t, SPIRVtype *> SPIRTypeMap;
typedef std::map<int32_t, std::string> ID2NameMap;
typedef std::map<int32_t, size_t_3> ID2Size3Map;
typedef std::set<spv::Decoration> DecorSet;
typedef std::map<int32_t, int32_t> ID2IDMap;

class SPIRVtypePOD : public SPIRVtype {
public:
  SPIRVtypePOD(int32_t Id, size_t Size) : SPIRVtype(Id, Size) {}
  virtual ~SPIRVtypePOD(){};
  virtual OCLType ocltype() override { return OCLType::POD; }
};

class SPIRVtypePODStruct : public SPIRVtype {
  size_t PackedSize_;
  bool IsPacked_;
public:
  SPIRVtypePODStruct(int32_t Id, size_t Size, size_t PSize)
      : SPIRVtype(Id, Size), PackedSize_(PSize), IsPacked_(false) {}
  virtual size_t size() override { return IsPacked_ ? PackedSize_ : Size_; }
  void setPacked(bool Val) { IsPacked_ = Val; }
  virtual ~SPIRVtypePODStruct(){};
  virtual OCLType ocltype() override { return OCLType::POD; }
};


class SPIRVtypeOpaque : public SPIRVtype {
  std::string Name;

public:
  SPIRVtypeOpaque(int32_t Id, std::string &&N)
      : SPIRVtype(Id, 0), Name(std::move(N)) {} // Opaque types are unsized.
  virtual ~SPIRVtypeOpaque(){};
  virtual OCLType ocltype() override { return OCLType::Opaque; }
};

class SPIRVtypeImage : public SPIRVtype {
  spv::AccessQualifier AQ;

public:
  SPIRVtypeImage(int32_t Id, int32_t AccessQual) : SPIRVtype(Id, 0) {
    if (AccessQual == (int32_t)spv::AccessQualifier::ReadOnly) {
      AQ = spv::AccessQualifier::ReadOnly;
    }
    if (AccessQual == (int32_t)spv::AccessQualifier::WriteOnly) {
      AQ = spv::AccessQualifier::WriteOnly;
    }
    if (AccessQual == (int32_t)spv::AccessQualifier::ReadWrite) {
      AQ = spv::AccessQualifier::ReadWrite;
    }
  }
  virtual ~SPIRVtypeImage(){};
  virtual OCLType ocltype() override { return OCLType::Image; }
  virtual OCLSpace getAS() override { return OCLSpace::Global; }
  virtual spv::AccessQualifier getImgAccess() override { return AQ; }
};

class SPIRVtypeSampler : public SPIRVtype {
public:
  SPIRVtypeSampler(int32_t Id) : SPIRVtype(Id, 0) {}
  virtual ~SPIRVtypeSampler(){};
  virtual OCLType ocltype() override { return OCLType::Sampler; }
  virtual OCLSpace getAS() override { return OCLSpace::Constant; }
};

class SPIRVtypePointer : public SPIRVtype {
  OCLSpace ASpace_;

public:
  SPIRVtypePointer(int32_t Id, int32_t StorClass, size_t PointerSize)
      : SPIRVtype(Id, PointerSize) {
    switch (StorClass) {
    case (int32_t)spv::StorageClass::CrossWorkgroup:
      ASpace_ = OCLSpace::Global;
      break;

    case (int32_t)spv::StorageClass::Workgroup:
      ASpace_ = OCLSpace::Local;
      break;

    case (int32_t)spv::StorageClass::UniformConstant:
      ASpace_ = OCLSpace::Constant;
      break;

    case (int32_t)spv::StorageClass::Function:
      assert(0 && "should have been handled elsewhere!");
      break;

    default:
      ASpace_ = OCLSpace::Unknown;
    }
  }
  virtual ~SPIRVtypePointer(){};
  virtual OCLType ocltype() override { return OCLType::Pointer; }
  OCLSpace getAS() override { return ASpace_; }
};

// Parses and checks SPIR-V header. Sets word buffer pointer to poin
// past the header and updates NumWords count to exclude header words.
// Return false if there is an error in the header. Otherwise, return
// true.
static bool parseHeader(const int32_t *&WordBuffer, size_t &NumWords) {
  if (*WordBuffer != spv::MagicNumber) {
    logError("Incorrect SPIR-V magic number.");
    return false;
  }
  ++WordBuffer;

  if (*WordBuffer < spv::Version10 || *WordBuffer > spv::Version15) {
    logError("Unsupported SPIR-V version.");
    return false;
  }
  ++WordBuffer;

  // GENERATOR
  ++WordBuffer;

  // BOUND
  // int32_t Bound = *WordBuffer;
  ++WordBuffer;

  // RESERVED
  if (*WordBuffer != 0) {
    logError("Invalid SPIR-V: Reserved word is not 0.");
    return false;
  }
  ++WordBuffer;

  NumWords -= 5;
  return true;
}

class SPIRVinst {
  spv::Op Opcode_;
  size_t WordCount_;
  // 9 required to fully decode images
  int32_t Word1_;
  int32_t Word2_;
  int32_t Word3_;
  int32_t Word4_;
  int32_t Word5_;
  int32_t Word6_;
  int32_t Word7_;
  int32_t Word8_;
  int32_t Word9_;
  std::string Extra_;
  const int32_t *OrigStream_;

public:
  SPIRVinst(const int32_t *Stream) {
    OrigStream_ = Stream;
    int32_t Word0 = Stream[0];
    WordCount_ = (unsigned)Word0 >> 16;
    Opcode_ = (spv::Op)(Word0 & 0xFFFF);

    if (WordCount_ > 1)
      Word1_ = Stream[1];

    if (WordCount_ > 2)
      Word2_ = Stream[2];

    if (WordCount_ > 3)
      Word3_ = Stream[3];

    if (WordCount_ > 4)
      Word4_ = Stream[4];

    if (WordCount_ > 5)
      Word5_ = Stream[5];

    if (WordCount_ > 6)
      Word6_ = Stream[6];

    if (WordCount_ > 7)
      Word7_ = Stream[7];

    if (WordCount_ > 8)
      Word8_ = Stream[8];

    if (WordCount_ > 9)
      Word9_ = Stream[9];

    if (Opcode_ == spv::Op::OpEntryPoint) {
      const char *Pp = (const char *)(Stream + 3);
      Extra_ = Pp;
    }

    if (Opcode_ == spv::Op::OpExtInstImport) {
      const char *Pp = (const char *)(Stream + 2);
      Extra_ = Pp;
    }

    if (Opcode_ == spv::Op::OpTypeOpaque) {
      const char *Pp = (const char *)(Stream + 2);
      Extra_ = Pp;
    }

    if (Opcode_ == spv::Op::OpName) {
      const char *Pp = (const char *)(Stream + 2);
      Extra_ = Pp;
    }
  }

  bool isKernelCapab() const {
    return (Opcode_ == spv::Op::OpCapability) &&
           (Word1_ == (int32_t)spv::Capability::Kernel);
  }
  bool isExtIntOpenCL() const { return Extra_ == OpenCLStd; }
  bool isMemModelOpenCL() const {
    return (Opcode_ == spv::Op::OpMemoryModel) &&
           (Word2_ == (int32_t)spv::MemoryModel::OpenCL);
  }
  bool isExecutionMode() const { return (Opcode_ == spv::Op::OpExecutionMode); }
  bool isLangOpenCL() const {
    return (Opcode_ == spv::Op::OpSource) &&
           ((Word1_ == (int32_t)spv::SourceLanguage::OpenCL_C) ||
            (Word1_ == (int32_t)spv::SourceLanguage::OpenCL_CPP));
  }
  bool isEntryPoint() {
    return (Opcode_ == spv::Op::OpEntryPoint) &&
           (Word1_ == (int32_t)spv::ExecutionModel::Kernel);
  }
  bool isFunctionType() const { return (Opcode_ == spv::Op::OpTypeFunction); }
  bool isFunction() const { return (Opcode_ == spv::Op::OpFunction); }
  bool isFunctionEnd() const { return (Opcode_ == spv::Op::OpFunctionEnd); }
  bool isFunctionParam() const {
    return (Opcode_ == spv::Op::OpFunctionParameter);
  }
  bool isName() const { return Opcode_ == spv::Op::OpName; }
  bool isDecoration() const { return Opcode_ == spv::Op::OpDecorate; }
  bool isType() const {
    return ((int32_t)Opcode_ >= (int32_t)spv::Op::OpTypeVoid) &&
           ((int32_t)Opcode_ <= (int32_t)spv::Op::OpTypeForwardPointer);
  }

  std::string &&getName() { return std::move(Extra_); }
  int32_t nameID() { return Word1_; }
  size_t getPointerSize() const {
    if (Opcode_ != spv::Op::OpMemoryModel)
      return 0;
    return (Word1_ == (int32_t)spv::AddressingModel::Physical64) ? 8 : 4;
  }

  size_t size() const { return WordCount_; }
  spv::Op getOpcode() const { return Opcode_; }

  int32_t entryPointID() { return Word2_; }
  int32_t getFunctionID() const { return Word2_; }
  int32_t getFunctionTypeID() const { return OrigStream_[4]; }
  int32_t getFunctionRetType() const { return Word1_; }
  int32_t getDecorationID() const { return Word1_; }
  int32_t getTypeID() const {
    assert(isType());
    return Word1_;
  }

  int32_t getFunctionParamID() const { return Word2_; }
  int32_t getFunctionParamType() const { return Word1_; }

  int32_t getExecutionModeEntryPoint() const { return Word1_; }
  bool isExecutionModeLocal() const {
    return Word2_ == (int32_t)spv::ExecutionMode::LocalSize;
  }
  bool isExecutionModeLocalHint() const {
    return Word2_ == (int32_t)spv::ExecutionMode::LocalSizeHint;
  }
  bool isExecutionModeVecTypeHint() const {
    return Word2_ == (int32_t)spv::ExecutionMode::VecTypeHint;
  }
  size_t_3 getExecutionModeSize() const {
    if (Opcode_ == spv::Op::OpExecutionMode)
      return size_t_3{(size_t)Word3_, (size_t)Word4_, (size_t)Word5_};
    else
      return size_t_3{0, 0, 0};
  }

  spv::Decoration getDecorationType() const { return (spv::Decoration)Word2_; }
  int32_t getDecorationExtraOper() const { return Word3_; }

  SPIRVtype *decodeType(SPIRTypeMap &TypeMap, size_t PointerSize) {
    if (Opcode_ == spv::Op::OpTypeVoid) {
      return new SPIRVtypePOD(Word1_, 0);
    }

    if (Opcode_ == spv::Op::OpTypeBool) {
      return new SPIRVtypePOD(Word1_, 1);
    }

    if (Opcode_ == spv::Op::OpTypeInt) {
      return new SPIRVtypePOD(Word1_, ((size_t)Word2_ / 8));
    }

    if (Opcode_ == spv::Op::OpTypeFloat) {
      return new SPIRVtypePOD(Word1_, ((size_t)Word2_ / 8));
    }

    if (Opcode_ == spv::Op::OpTypeVector) {
      auto Type = TypeMap[Word2_];
      if (!Type) {
        logWarn("SPIR-V Parser: Word2_ %i not found in type map", Word2_);
        return nullptr;
      }
      size_t TypeSize = Type->size();
      return new SPIRVtypePOD(Word1_, TypeSize * OrigStream_[3]);
    }

    if (Opcode_ == spv::Op::OpTypeArray) {
      auto Type = TypeMap[Word2_];
      if (!Type) {
        logWarn("SPIR-V Parser: Word2_ %i not found in type map", Word2_);
        return nullptr;
      }
      size_t TypeSize = Type->size();
      return new SPIRVtypePOD(Word1_, TypeSize * Word3_);
    }

    if (Opcode_ == spv::Op::OpTypeStruct) {
      size_t TotalSize = 0;
      size_t TotalPackedSize = 0;
      for (size_t i = 2; i < WordCount_; ++i) {
        int32_t MemberId = OrigStream_[i];

        auto Type = TypeMap[MemberId];
        if (!Type) {
          logWarn("SPIR-V Parser: MemberId %i not found in type map", MemberId);
          continue;
        }

        size_t TypeSize = Type->size();
        TotalPackedSize += TypeSize;
        if (TotalSize % TypeSize != 0) {
          size_t Count = TotalSize / TypeSize;
          TotalSize = (Count + 1) * TypeSize;
        }

        TotalSize += TypeSize;
      }
      // logTrace("TOTAL STRUCT SIZE: %zu\n", TotalSize);
      return new SPIRVtypePODStruct(Word1_, TotalSize, TotalPackedSize);
    }

    if (Opcode_ == spv::Op::OpTypeOpaque) {
      return new SPIRVtypeOpaque(Word1_, std::move(Extra_));
    }

    if (Opcode_ == spv::Op::OpTypeImage) {
      return new SPIRVtypeImage(Word1_, Word9_);
    }

    if (Opcode_ == spv::Op::OpTypeSampler) {
      return new SPIRVtypeSampler(Word1_);
    }

    if (Opcode_ == spv::Op::OpTypePointer) {
      // structs or vectors passed by value are represented in LLVM IR / SPIRV
      // by a pointer with "byval" keyword; handle them here
      if (Word2_ == (int32_t)spv::StorageClass::Function) {
        int32_t Pointee = Word3_;
        auto Type = TypeMap[Pointee];
        if (!Type) {
          logError("SPIR-V Parser: Failed to find size for type id %i",
                   Pointee);
          return nullptr;
        }

        size_t PointeeSize = Type->size();
        return new SPIRVtypePOD(Word1_, PointeeSize);

      } else
        return new SPIRVtypePointer(Word1_, Word2_, PointerSize);
    }

    return nullptr;
  }

  // doesn't result in full decoding because some attrs (eg names)
  // are attached to function parameters, not their types
  OCLFuncInfo *decodeFunctionType(SPIRTypeMap &TypeMap,
                                  ID2Size3Map ReqLocalMap_,
                                  ID2Size3Map LocalHintMap_,
                                  ID2Size3Map VecTypeHintMap_,
                                  size_t PointerSize) {
    assert(Opcode_ == spv::Op::OpTypeFunction);

    OCLFuncInfo *Fi = new OCLFuncInfo;

    int32_t RetId = Word2_;
    auto It = TypeMap.find(RetId);
    assert(It != TypeMap.end());
    Fi->RetTypeInfo.Type = It->second->ocltype();
    Fi->RetTypeInfo.Size = It->second->size();
    Fi->RetTypeInfo.Space = It->second->getAS();

    size_t NumArgs = WordCount_ - 3;
    if (NumArgs > 0) {
      Fi->ArgTypeInfo.resize(NumArgs);
      for (size_t i = 0; i < NumArgs; ++i) {
        int32_t TypeId = OrigStream_[i + 3];
        auto It = TypeMap.find(TypeId);
        assert(It != TypeMap.end());
        Fi->ArgTypeInfo[i].TypeID = TypeId;
        Fi->ArgTypeInfo[i].Type = It->second->ocltype();
        Fi->ArgTypeInfo[i].Size = It->second->size();
        Fi->ArgTypeInfo[i].Space = It->second->getAS();
        switch (It->second->getImgAccess()) {
        case spv::AccessQualifier::ReadOnly:
          Fi->ArgTypeInfo[i].Attrs.ReadableImg = 1;
          Fi->ArgTypeInfo[i].Attrs.WriteableImg = 0;
          break;
        case spv::AccessQualifier::ReadWrite:
          Fi->ArgTypeInfo[i].Attrs.ReadableImg = 1;
          Fi->ArgTypeInfo[i].Attrs.WriteableImg = 1;
          break;
        case spv::AccessQualifier::WriteOnly:
          Fi->ArgTypeInfo[i].Attrs.ReadableImg = 0;
          Fi->ArgTypeInfo[i].Attrs.WriteableImg = 1;
          break;
        default:
          Fi->ArgTypeInfo[i].Attrs.ReadableImg = 0;
          Fi->ArgTypeInfo[i].Attrs.WriteableImg = 0;
        }
      }
    }

    int32_t FuncID = getTypeID();
    if (ReqLocalMap_.find(FuncID) != ReqLocalMap_.end()) {
      Fi->ReqLocalSize = ReqLocalMap_.at(FuncID);
    }
    if (LocalHintMap_.find(FuncID) != LocalHintMap_.end()) {
      Fi->LocalSizeHint = LocalHintMap_.at(FuncID);
    }
    if (VecTypeHintMap_.find(FuncID) != VecTypeHintMap_.end()) {
      Fi->VecTypeHint = VecTypeHintMap_.at(FuncID);
    }
    return Fi;
  }
};

class SPIRVmodule {
  ID2NameMap EntryPointMap_;
  ID2NameMap NameMap_;
  SPIRTypeMap TypeMap_;
  ID2Size3Map ReqLocalMap_;
  ID2Size3Map LocalHintMap_;
  ID2Size3Map VecTypeHintMap_;
  OCLFuncInfoMap FunctionTypeMap_;
  std::map<int32_t, DecorSet> DecorationMap_;
  ID2IDMap EntryToFunctionTypeIDMap_;
  ID2IDMap AlignmentMap_;

  bool MemModelCL_;
  bool KernelCapab_;
  bool ExtIntOpenCL_;
  bool HeaderOK_;
  bool ParseOK_;

public:
  ~SPIRVmodule() {
    for (auto I : TypeMap_) {
      delete I.second;
    }
  }

  bool valid() {
    bool AllOk = true;
    auto Check = [&](bool Cond, const char *ErrMsg) {
      if (!Cond)
        logError("%s", ErrMsg);
      AllOk &= Cond;
    };

    Check(HeaderOK_, "Invalid SPIR-V header.");
    // TODO: Temporary. With these check disabled the simple_kernel
    //       runs successfully on OpenCL backend at least. Note that we are
    //       passing invalid SPIR-V binary.
    // Check(KernelCapab_, "Kernel capability missing.");
    // Check(ExtIntOpenCL_, "Missing extended OpenCL instructions.");
    Check(MemModelCL_, "Incorrect memory model.");
    Check(ParseOK_, "An error encountered during parsing.");
    return AllOk;
  }

  bool parseSPIRV(const int32_t *Stream, size_t NumWords) {
    KernelCapab_ = false;
    ExtIntOpenCL_ = false;
    HeaderOK_ = false;
    MemModelCL_ = false;
    ParseOK_ = false;
    HeaderOK_ = parseHeader(Stream, NumWords);
    if (!HeaderOK_)
      return false;

    // INSTRUCTION STREAM
    ParseOK_ = parseInstructionStream(Stream, NumWords);
    return valid();
  }

  bool fillModuleInfo(OpenCLFunctionInfoMap &ModuleMap) {
    if (!valid())
      return false;

    for (auto i : EntryPointMap_) {
      int32_t EntryPointID = i.first;
      auto Ft = EntryToFunctionTypeIDMap_.find(EntryPointID);
      assert(Ft != EntryToFunctionTypeIDMap_.end());
      auto Fi = FunctionTypeMap_.find(Ft->second);
      assert(Fi != FunctionTypeMap_.end());
      ModuleMap.emplace(std::make_pair(i.second, Fi->second));
    }
    FunctionTypeMap_.clear();

    return true;
  }

private:
  bool parseInstructionStream(const int32_t *Stream, size_t NumWords) {
    const int32_t *StreamIntPtr = Stream;
    size_t PointerSize = 0;
    int32_t CurrentKernelID = 0;
    int32_t CurrentKernelParam = 0;
    while (NumWords > 0) {
      SPIRVinst Inst(StreamIntPtr);

      if (Inst.isKernelCapab())
        KernelCapab_ = true;

      if (Inst.isExtIntOpenCL())
        ExtIntOpenCL_ = true;

      if (Inst.isMemModelOpenCL()) {
        MemModelCL_ = true;
        PointerSize = Inst.getPointerSize();
        assert(PointerSize > 0);
      }

      if (Inst.isExecutionMode()) {
        int32_t ID = Inst.getExecutionModeEntryPoint();
        if (Inst.isExecutionModeLocal()) {
          ReqLocalMap_.emplace(std::make_pair(ID, Inst.getExecutionModeSize()));
        }
        if (Inst.isExecutionModeLocalHint()) {
          LocalHintMap_.emplace(
              std::make_pair(ID, Inst.getExecutionModeSize()));
        }
        if (Inst.isExecutionModeVecTypeHint()) {
          VecTypeHintMap_.emplace(
              std::make_pair(ID, Inst.getExecutionModeSize()));
        }
      }

      if (Inst.isEntryPoint()) {
        EntryPointMap_.emplace(
            std::make_pair(Inst.entryPointID(), Inst.getName()));
      }

      if (Inst.isName()) {
        NameMap_.emplace(std::make_pair(Inst.nameID(), Inst.getName()));
      }

      if (Inst.isDecoration()) {
        int32_t ID = Inst.getDecorationID();
        spv::Decoration Type = Inst.getDecorationType();
        DecorationMap_[ID].insert(Type);
        if (Type == spv::Decoration::Alignment) {
          AlignmentMap_[ID] = Inst.getDecorationExtraOper();
        }
      }

      if (Inst.isType()) {
        if (Inst.isFunctionType())
          FunctionTypeMap_.emplace(std::make_pair(
              Inst.getTypeID(),
              Inst.decodeFunctionType(TypeMap_, ReqLocalMap_, LocalHintMap_,
                                      VecTypeHintMap_, PointerSize)));
        else
          TypeMap_.emplace(std::make_pair(
              Inst.getTypeID(), Inst.decodeType(TypeMap_, PointerSize)));
      }

      if (Inst.isFunction() &&
          (EntryPointMap_.find(Inst.getFunctionID()) != EntryPointMap_.end())) {
        // ret type must be void, to be a kernel
        auto Retty = TypeMap_.find(Inst.getFunctionRetType());
        assert(Retty != TypeMap_.end());
        assert(TypeMap_[Inst.getFunctionRetType()]->size() == 0);
        assert(CurrentKernelID == 0);
        CurrentKernelID = Inst.getFunctionID();

        EntryToFunctionTypeIDMap_.emplace(
            std::make_pair(Inst.getFunctionID(), Inst.getFunctionTypeID()));
      }

      if (Inst.isFunctionParam() && (CurrentKernelID != 0)) {
        assert(EntryPointMap_.find(CurrentKernelID) != EntryPointMap_.end());
        int32_t KernelTypeID = EntryToFunctionTypeIDMap_[CurrentKernelID];
        assert(FunctionTypeMap_.find(KernelTypeID) != FunctionTypeMap_.end());
        OCLFuncInfo *FI = FunctionTypeMap_[KernelTypeID].get();
        OCLArgTypeInfo &AI = FI->ArgTypeInfo[CurrentKernelParam];
        int32_t ParamID = Inst.getFunctionParamID();
        int32_t ParamType = Inst.getFunctionParamType();
        if (NameMap_.find(ParamID) != NameMap_.end()) {
          AI.Name = NameMap_[ParamID];
        } else {
          AI.Name = "unknown";
        }
        AI.Attrs.CPacked = 0;
        AI.Attrs.Constant = 0;
        AI.Attrs.Restrict = 0;
        AI.Attrs.Volatile = 0;
        if (DecorationMap_.find(ParamID) != DecorationMap_.end()) {
          DecorSet &DS = DecorationMap_[ParamID];
          for (spv::Decoration D : DS) {
            switch (D) {
            // case spv::Decoration::SpecId: break; // TODO
            case spv::Decoration::CPacked:
              AI.Attrs.CPacked = 1;
              break;
            case spv::Decoration::Restrict:
              AI.Attrs.Restrict = 1;
              break;
            case spv::Decoration::Volatile:
              AI.Attrs.Volatile = 1;
              break;
            case spv::Decoration::Constant:
              AI.Attrs.Constant = 1;
              break;
            case spv::Decoration::Alignment: {
              if (AlignmentMap_.find(ParamID) != AlignmentMap_.end())
                AI.Alignment = AlignmentMap_[ParamID];
            }
            default:
              break;
            }
          }
        }

        if (AI.Attrs.CPacked) {
          auto It = TypeMap_.find(AI.TypeID);
          assert(It != TypeMap_.end());
          SPIRVtypePODStruct *Str = static_cast<SPIRVtypePODStruct *>(It->second);
          Str->setPacked(true);
          AI.Size = Str->size();
        }

        ++CurrentKernelParam;
      }

      if (Inst.isFunctionEnd()) {
        CurrentKernelID = 0;
        CurrentKernelParam = 0;
      }

      size_t InsnSize = Inst.size();
      assert(InsnSize && "Invalid instruction size, will loop forever!");

      NumWords -= Inst.size();
      StreamIntPtr += Inst.size();
    }

    return true;
  }
};

bool poclParseSPIRV(int32_t *Stream, size_t NumWords,
               OpenCLFunctionInfoMap &Output) {
  SPIRVmodule Mod;
  if (!Mod.parseSPIRV(Stream, NumWords))
    return false;
  return Mod.fillModuleInfo(Output);
}
