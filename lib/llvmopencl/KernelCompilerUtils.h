// Misc. helpers for kernel compilation.
//
// Copyright (c) 2024 Pekka Jääskeläinen / Intel Finland Oy
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

#ifndef POCL_KERNEL_COMPILER_UTILS_H
#define POCL_KERNEL_COMPILER_UTILS_H

#include "config.h"

// Generates the name for the global magic variable for the local id.
#define LID_G_NAME(DIM) (std::string("_local_id_") + (char)('x' + DIM))
// Generates the name for the global magic variable for the global id iterator.
#define GID_G_NAME(DIM) (std::string("_global_id_") + (char)('x' + DIM))
// Generates the name for the global magic variable for the group id.
#define GROUP_ID_G_NAME(DIM) (std::string("_group_id_") + (char)('x' + DIM))
// Generates the name for the global magic variable for the local size.
#define LS_G_NAME(DIM) (std::string("_local_size_") + (char)('x' + DIM))

#ifdef KERNEL_TRIPLE_TARGETS_MSVC_TOOLCHAIN
// Used 'clang -target x86_64-pc-windows-msvc -cl-std=cl3.0 -S -emit-llvm
// -o - some.cl' to generate MSVC-mangled symbols.

// The name of the mangled get_global_id() builtin function.
#define GID_BUILTIN_NAME "?get_global_id@@$$J0YAKI@Z"
// The name of the mangled get_global_size() builtin function.
#define GS_BUILTIN_NAME "?get_local_size@@$$J0YAKI@Z"
// The name of the mangled get_group_id() builtin function.
#define GROUP_ID_BUILTIN_NAME "?get_group_id@@$$J0YAKI@Z"
// The name of the mangled get_local_id() builtin function.
#define LID_BUILTIN_NAME "?get_local_id@@$$J0YAKI@Z"
// The name of the mangled get_local_size() builtin function.
#define LS_BUILTIN_NAME "?get_local_size@@$$J0YAKI@Z"
// The name of the mangled get_local_size() builtin function.
#define GOFF_BUILTIN_NAME "?get_global_offset@@$$J0YAKI@Z"
// The name of the mangled get_enqueued_local_size() builtin function.
#define ENQUEUE_LS_BUILTIN_NAME "?get_enqueued_local_size@@$$J0YAKI@Z"
// The name of the mangled get_num_groups() builtin function.
#define NGROUPS_BUILTIN_NAME "?get_num_groups@@$$J0YAKI@Z"
// The name of the mangled get_work_dim() builtin function.
#define WDIM_BUILTIN_NAME "?get_work_dim@@$$J0YAIXZ"
// The name of the mangled get_global_linear_id() builtin function.
#define GLID_BUILTIN_NAME "?get_global_linear_id@@$$J0YAKXZ"
// The name of the mangled get_local_linear_id() builtin function.
#define LLID_BUILTIN_NAME "?get_local_linear_id@@$$J0YAKXZ"
// The name of the mangled barrier() function.
#define BARRIER_BUILTIN_NAME "?barrier@@$$J0YAXI@Z"

#else
// Assuming Itanium mangling.
// Used 'clang -target x86_64-pc-linux-gnu -cl-std=cl3.0 -S -emit-llvm
// -o - some.cl' to generate Itanium-mangled symbols.

// The name of the mangled get_global_id() builtin function.
#define GID_BUILTIN_NAME "_Z13get_global_idj"
// The name of the mangled get_global_size() builtin function.
#define GS_BUILTIN_NAME "_Z15get_global_sizej"
// The name of the mangled get_group_id() builtin function.
#define GROUP_ID_BUILTIN_NAME "_Z12get_group_idj"
// The name of the mangled get_local_id() builtin function.
#define LID_BUILTIN_NAME "_Z12get_local_idj"
// The name of the mangled get_local_size() builtin function.
#define LS_BUILTIN_NAME "_Z14get_local_sizej"
// The name of the mangled get_local_size() builtin function.
#define GOFF_BUILTIN_NAME "_Z17get_global_offsetj"
// The name of the mangled get_enqueued_local_size() builtin function.
#define ENQUEUE_LS_BUILTIN_NAME "_Z23get_enqueued_local_sizej"
// The name of the mangled get_num_groups() builtin function.
#define NGROUPS_BUILTIN_NAME "_Z14get_num_groupsj"
// The name of the mangled get_work_dim() builtin function.
#define WDIM_BUILTIN_NAME "_Z12get_work_dimv"
// The name of the mangled get_global_linear_id() builtin function.
#define GLID_BUILTIN_NAME "_Z20get_global_linear_idv"
// The name of the mangled get_local_linear_id() builtin function.
#define LLID_BUILTIN_NAME "_Z19get_local_linear_idv"
// The name of the mangled barrier() function.
#define BARRIER_BUILTIN_NAME "_Z7barrierj"
#endif

#endif
