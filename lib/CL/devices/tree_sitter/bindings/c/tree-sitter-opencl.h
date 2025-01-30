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

#ifndef TREE_SITTER_OPENCL_H_
#define TREE_SITTER_OPENCL_H_

typedef struct TSLanguage TSLanguage;

#ifdef __cplusplus
extern "C"
{
#endif

  const TSLanguage *tree_sitter_opencl (void);

#ifdef __cplusplus
}
#endif

#endif // TREE_SITTER_OPENCL_H_
