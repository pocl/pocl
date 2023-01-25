// Header for WorkitemHandlerChooser function pass.
// 
// Copyright (c) 2012 Pekka Jääskeläinen / TUT
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

#ifndef _POCL_WORKITEM_HANDLER_CHOOSER_H
#define _POCL_WORKITEM_HANDLER_CHOOSER_H

#include "WorkitemHandler.h"

namespace pocl {
  class Workgroup;

  class WorkitemHandlerChooser : public pocl::WorkitemHandler {
  public:
    static char ID;
    
    enum WorkitemHandlerType {
      POCL_WIH_FULL_REPLICATION,
      POCL_WIH_LOOPS,
      POCL_WIH_CBS
    };

  WorkitemHandlerChooser() : pocl::WorkitemHandler(ID), 
      chosenHandler_(POCL_WIH_LOOPS) {}

    virtual void getAnalysisUsage(llvm::AnalysisUsage &AU) const;
    virtual bool runOnFunction(llvm::Function &F);
    
    WorkitemHandlerType chosenHandler() { return chosenHandler_; }
  private:
    WorkitemHandlerType chosenHandler_;
  };
}

#endif
