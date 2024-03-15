/* OpenCL built-in library: internal work group memory allocation functionality

   Copyright (c) 2022-2023 Pekka Jääskeläinen / Intel Finland Oy

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

/**
 * \brief Internal pseudo function which allocates space from the work-group
 * thread's stack (basically local memory) for each work-item.
 *
 * It's expanded in WorkitemLoops.cc to an alloca().
 *
 * @param element_size The size of an element to allocate (for all WIs in the
 * WG).
 * @param align The alignment of the start of chunk.
 * @param extra_bytes extra bytes to add to the allocation, some functions need
 * extra space
 * @return pointer to the allocated stack space (freed at unwind).
 */
void *__pocl_work_group_alloca (size_t element_size, size_t align,
                                size_t extra_bytes);

/**
 * \brief Internal pseudo function which allocates space from the work-group
 * thread's stack (basically local memory).
 *
 * It's expanded in WorkitemLoops.cc to an alloca().
 *
 * @param bytes The size of data to allocate in bytes.
 * @param align The alignment of the start of chunk.
 * @return pointer to the allocated stack space (freed at unwind).
 */
void *__pocl_local_mem_alloca (size_t bytes, size_t align);
