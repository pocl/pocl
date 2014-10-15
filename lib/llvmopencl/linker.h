#ifndef POCL_LINKER_H
#define POCL_LINKER_H

#include "config.h"
#ifdef LLVM_3_2
#include "llvm/Module.h"
#else
#include "llvm/IR/Module.h"
#endif

/**
 * Link in module lib to krn.
 * This function searches for each undefined symbol 
 * in krn from lib, cloning as needed. For big modules,
 * this is faster than calling llvm::Linker and then
 * running DCE.
 */
void link(llvm::Module *krn, const llvm::Module *lib);

#endif
