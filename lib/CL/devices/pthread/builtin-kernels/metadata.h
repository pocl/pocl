//
// Created by rabijl on 28.11.2023.
// This file contains metadata on builtin kernels that can be loaded as an so.
//

#ifndef POCL_METADATA_H
#define POCL_METADATA_H

#define NUM_PTHREAD_BUILTIN_HOST_KERNELS 1
static char *const kernel_names[NUM_PTHREAD_BUILTIN_HOST_KERNELS] = {
  "pocl.add.i8",
};

// Make sure LD_LIBRARY_PATH is set to contain the .so files
static char *const dylib_names[NUM_PTHREAD_BUILTIN_HOST_KERNELS] = {
  "libpocl_pthread_add_i8.so",
};

static const char *const init_fn_names[NUM_PTHREAD_BUILTIN_HOST_KERNELS] = {
  "init_pocl_add_i8",
};

static const char *const free_fn_names[NUM_PTHREAD_BUILTIN_HOST_KERNELS] = {
  "free_pocl_add_i8",
};

#endif // POCL_METADATA_H
