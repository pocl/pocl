#ifndef POCL_LINKER_H
#define POCL_LINKER_H

#include "config.h"

#include "llvm/IR/Module.h"
#include "llvm/ADT/Triple.h"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

/**
 * Link in module lib to krn.
 * This function searches for each undefined symbol 
 * in krn from lib, cloning as needed. For big modules,
 * this is faster than calling llvm::Linker and then
 * running DCE.
 *
 * log is used to report errors if we run into undefined symbols
 */
int link(llvm::Module *krn, const llvm::Module *lib, std::string &log);

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif
