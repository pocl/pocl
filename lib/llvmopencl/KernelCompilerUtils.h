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

// Generates the name for the global magic variable for the local id.
#define LID_G_NAME(DIM) (std::string("_local_id_") + (char)('x' + DIM))
// Generates the name for the global magic variable for the global id iterator.
#define GID_G_NAME(DIM) (std::string("_global_id_") + (char)('x' + DIM))
// The name of the mangled get_global_id builtin function.
#define GID_BUILTIN_NAME "_Z13get_global_idj"

#endif
