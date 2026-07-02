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

/* In-process JIT linking and loading of CPU host-device kernel objects via an
   ORC LLJIT whose object-linking layer is JITLink. JITLink is both the linker
   and the loader: it resolves relocations, maps the kernel code into
   executable memory, and registers EH frames, without spawning an external
   linker and without producing an on-disk shared object. */

#ifndef POCL_LLVM_ORC_H
#define POCL_LLVM_ORC_H

#ifdef __cplusplus
extern "C"
{
#endif

/* Create the process-global LLJIT that loads kernel objects. Called from
   CPU device init; idempotent and thread-safe. The triple determines the
   JIT's data layout (and thus symbol mangling). Returns 0 on success. A
   failure to create the JIT is latched internally and every later call
   fails fast; device init uses the result to permanently route the device
   through the linker path (see pocl_cpu_device_uses_jit()). */
int pocl_jit_initialize (const char *triple,
                         const char *cpu);

/* Read a relocatable kernel object file from 'path' and stage it for
   in-process JIT-linking in a fresh isolated JITDylib, returning an opaque
   handle (NULL on failure). Relocation and linking are deferred until the
   first pocl_jit_lookup() on the handle. 'uniq_name' is used only for
   diagnostics and need not be unique. */
void *pocl_jit_load_object (const char *path, const char *uniq_name,
                            const char *kernellib_bc_path);

/* Look up an (unmangled) symbol in a previously loaded object. Returns the
   executable address, or NULL on failure; the failure text is then available
   from pocl_jit_last_error(). */
void *pocl_jit_lookup (void *handle, const char *symbol_name);

/* Text of the most recent pocl_jit_lookup() failure (a dlerror() analogue),
   or NULL if no lookup has failed yet. With JITLink the first lookup is where
   linking happens, so this is the diagnostic that names unresolved symbols or
   relocation problems. The returned pointer is valid until the next failing
   lookup; use (or copy) it immediately. */
const char *pocl_jit_last_error (void);

/* Unload a previously loaded object, reclaiming its code and memory.
   Returns nonzero on success. */
int pocl_jit_unload (void *handle);

#ifdef __cplusplus
}
#endif

#endif
