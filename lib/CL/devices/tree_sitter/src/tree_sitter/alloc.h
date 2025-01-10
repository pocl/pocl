/*
The MIT License (MIT)

Copyright (c) 2014 Max Brunsfeld (C version), 2023 Peter Lef (OpenCL C
modifications).

Generated from https://github.com/lefp/tree-sitter-opencl commit: 8e1d24a57066

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef TREE_SITTER_ALLOC_H_
#define TREE_SITTER_ALLOC_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// Allow clients to override allocation functions
#ifdef TREE_SITTER_REUSE_ALLOCATOR

extern void *(*ts_current_malloc)(size_t size);
extern void *(*ts_current_calloc)(size_t count, size_t size);
extern void *(*ts_current_realloc)(void *ptr, size_t size);
extern void (*ts_current_free)(void *ptr);

#ifndef ts_malloc
#define ts_malloc  ts_current_malloc
#endif
#ifndef ts_calloc
#define ts_calloc  ts_current_calloc
#endif
#ifndef ts_realloc
#define ts_realloc ts_current_realloc
#endif
#ifndef ts_free
#define ts_free    ts_current_free
#endif

#else

#ifndef ts_malloc
#define ts_malloc  malloc
#endif
#ifndef ts_calloc
#define ts_calloc  calloc
#endif
#ifndef ts_realloc
#define ts_realloc realloc
#endif
#ifndef ts_free
#define ts_free    free
#endif

#endif

#ifdef __cplusplus
}
#endif

#endif // TREE_SITTER_ALLOC_H_
