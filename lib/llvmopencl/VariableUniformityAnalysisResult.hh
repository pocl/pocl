// Implementation for VariableUniformityAnalysis function pass.
//
// Copyright (c) 2023 Michal Babej / Intel Finland Oy
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

namespace llvm {
class Function;
class LoopInfo;
class PostDominatorTree;
class BasicBlock;
class Value;
class Module;
class Loop;

class PreservedAnalyses;
template <typename, typename...> class AnalysisManager;
} // namespace llvm

#include <map>

namespace pocl {
class VariableUniformityAnalysisResult {

public:
  bool runOnFunction(llvm::Function &F, llvm::LoopInfo &LI,
                     llvm::PostDominatorTree &PDT);
  bool isUniform(llvm::Function *f, llvm::Value *v);
  void setUniform(llvm::Function *f, llvm::Value *v, bool isUniform = true);
  void analyzeBBDivergence(llvm::Function *f, llvm::BasicBlock *bb,
                           llvm::BasicBlock *previousUniformBB,
                           llvm::PostDominatorTree &PDT);

  bool shouldBePrivatized(llvm::Function *f, llvm::Value *val);
  bool doFinalization(llvm::Module &M);
  void markInductionVariables(llvm::Function &F, llvm::Loop &L);
  ~VariableUniformityAnalysisResult() { uniformityCache_.clear(); }

  // TODO this could be wrong
#if LLVM_MAJOR >= MIN_LLVM_NEW_PASSMANAGER
  bool invalidate(llvm::Function &F, const llvm::PreservedAnalyses PA,
                  llvm::AnalysisManager<llvm::Function>::Invalidator &Inv);
#endif

private:
  bool isUniformityAnalyzed(llvm::Function *f, llvm::Value *val) const;

  using UniformityIndex = std::map<llvm::Value *, bool>;
  using UniformityCache = std::map<llvm::Function *, UniformityIndex>;
  mutable UniformityCache uniformityCache_;
};

} // namespace pocl
