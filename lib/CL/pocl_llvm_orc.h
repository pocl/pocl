/* OpenCL runtime library: in-process JIT linking and loading of kernel object
   files via LLVM ORC + JITLink.

   Copyright (c) 2026 PoCL developers

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

/* These functions replace the "link kernel object into a shared library via
   the Clang driver, then dlopen() it" pipeline for CPU host devices with an
   in-process ORC LLJIT whose object-linking layer is JITLink. JITLink acts as
   both the linker and the loader: it resolves relocations, maps the kernel
   code into executable memory, and registers EH frames, all without spawning
   an external linker and without producing an on-disk shared object. */

#ifndef POCL_LLVM_ORC_H
#define POCL_LLVM_ORC_H

#ifdef __cplusplus
extern "C"
{
#endif

  /* Lazily create the process-global LLJIT used to load kernel objects.
     Idempotent and thread-safe; the triple/CPU determine the JIT's data
     layout (and thus symbol mangling). Returns 0 on success. */
  int pocl_jit_initialize (const char *triple, const char *cpu);

  /* Read a relocatable kernel object file from 'path', JIT-link it into the
     process inside a fresh isolated namespace, and return an opaque handle
     (NULL on failure). 'uniq_name' is used only for diagnostics and need not
     be unique. */
  void *pocl_jit_load_object (const char *path, const char *uniq_name);

  /* Look up an (unmangled) symbol in a previously loaded object. Returns the
     executable address, or NULL if not found. */
  void *pocl_jit_lookup (void *handle, const char *symbol_name);

  /* Unload a previously loaded object, reclaiming its code and memory.
     Returns nonzero on success. */
  int pocl_jit_unload (void *handle);

#ifdef __cplusplus
}
#endif

#endif
