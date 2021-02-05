/* pocl_llvm_api.cc: internally used header for pocl's LLVM API sources.

   Copyright (c) 2013 Kalle Raiskila
                 2013-2017 Pekka Jääskeläinen

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

#include "pocl_llvm.h"

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING ("-Wunused-parameter")

#include <llvm/IR/Module.h>
#include <map>
#include <string>

typedef struct _cl_device_id *cl_device_id;

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

/* The LLVM API interface functions are not thread safe at the moment;
 * Pocl needs to ensure only one thread is using this layer at the time.
 *
 * Pocl used a llvm::sys::Mutex class variable before, unfortunately,
 * using llvm::sys::Mutex is not safe. Reason:
 *
 * if pocl is dlopened from a C++ program, pocl's C++ object destructors
 * are called before the program's dtors. This causes the Mutex to be destroyed,
 * and if the program's dtors call clReleaseProgram()
 * -> pocl_free_llvm_irs() -> llvm::PoclMutexGuard guard_variable(Mutex)
 * ... the program will freeze/segfault.
 *
 * This happens with many ViennaCL examples.
 *
 * This class is a replacement that uses a simple pthread lock
 */

class PoclCompilerMutexGuard {
  PoclCompilerMutexGuard(const PoclCompilerMutexGuard &) = delete;
  void operator=(const PoclCompilerMutexGuard &) = delete;

public:
  // an unused argument is required, otherwise compiler optimizes out the object
  PoclCompilerMutexGuard(void *unused);
  ~PoclCompilerMutexGuard();
};


POCL_EXPORT
extern cl_device_id currentPoclDevice;

void InitializeLLVM();
llvm::LLVMContext &GlobalContext();
extern long numberOfIRs;

llvm::Module *parseModuleIR(const char *path);
void writeModuleIR(const llvm::Module *mod, std::string &str);
llvm::Module *parseModuleIRMem(const char *input_stream, size_t size);
int getModuleTriple(const char *input_stream, size_t size, std::string &triple);
std::string getDiagString();

void clearKernelPasses();
void clearTargetMachines();
void cleanKernelLibrary();

extern std::string currentWgMethod;

#ifdef __GNUC__
#pragma GCC visibility pop
#endif
