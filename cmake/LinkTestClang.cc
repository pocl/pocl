/* PoCL CMake build system: LinkTestClang.cc

   Copyright (c) 2023 Michal Babej / Intel Finland Oy

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

#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/LangOptions.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <stdio.h>

using namespace clang;
using namespace llvm;

int main(int argc, char *argv[]) {
  if (argc < 2)
    exit(2);

  CompilerInstance CI;
  CompilerInvocation &pocl_build = CI.getInvocation();
#if (LLVM_MAJOR < 18)
  LangOptions *la = pocl_build.getLangOpts();
#else
  LangOptions L = pocl_build.getLangOpts();
  LangOptions *la = &L;
#endif
  PreprocessorOptions &po = pocl_build.getPreprocessorOpts();
  po.Includes.push_back("/usr/include/test/path.h");

  la->OpenCLVersion = 300;
  la->FakeAddressSpaceMap = false;
  la->Blocks = true;     //-fblocks
  la->MathErrno = false; // -fno-math-errno
  la->NoBuiltin = true;  // -fno-builtin
  la->AsmBlocks = true;  // -fasm (?)

  la->setStackProtector(LangOptions::StackProtectorMode::SSPOff);

  la->PICLevel = PICLevel::BigPIC;
  la->PIE = 0;

  clang::TargetOptions &ta = pocl_build.getTargetOpts();
  ta.Triple = "x86_64-pc-linux-gnu";
  ta.CPU = "haswell";

  FrontendOptions &fe = pocl_build.getFrontendOpts();
  fe.Inputs.clear();

  fe.Inputs.push_back(
      FrontendInputFile(argv[1], clang::InputKind(clang::Language::OpenCL)));

  CodeGenOptions &cg = pocl_build.getCodeGenOpts();
  cg.EmitOpenCLArgMetadata = true;
  cg.StackRealignment = true;
  cg.VerifyModule = true;

  bool success = true;
  clang::PrintPreprocessedAction Preprocess;
  success = CI.ExecuteAction(Preprocess);

  return (success ? 0 : 11);
}
