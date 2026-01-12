/* StripMemSpaces.cpp

   Copyright (c) 2026 Topi Leppänen / Tampere University

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Pass/Pass.h>

#include "pocl/Transforms/Passes.hh"

namespace {
#define GEN_PASS_DEF_STRIPMEMSPACES
#include "pocl/Transforms/Passes.h.inc"
} // namespace

using namespace mlir;

class StripMemorySpaceConverter final : public TypeConverter {
public:
  StripMemorySpaceConverter() {
    addConversion([](BaseMemRefType memRefType) -> std::optional<Type> {
      if (auto memSpaceAttr = memRefType.getMemorySpace()) {
        if (auto rankedType = dyn_cast<MemRefType>(memRefType)) {
          return MemRefType::get(memRefType.getShape(),
                                 memRefType.getElementType(),
                                 rankedType.getLayout(), mlir::Attribute());
        }
        return UnrankedMemRefType::get(memRefType.getElementType(),
                                       mlir::Attribute());
      }
      return memRefType;
    });
  }
};

namespace {
struct StripMemSpaces : public impl::StripMemSpacesBase<StripMemSpaces> {
  void runOnOperation() override {
    auto converter = StripMemorySpaceConverter();
    AttrTypeReplacer replacer;
    replacer.addReplacement(
        [&converter](BaseMemRefType origType) -> std::optional<BaseMemRefType> {
          return converter.convertType<BaseMemRefType>(origType);
        });
    replacer.recursivelyReplaceElementsIn(getOperation(),
                                          /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::pocl::createStripMemSpacesPass() {
  return std::make_unique<StripMemSpaces>();
}
