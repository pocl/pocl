/* pocl_llvm.h: interface to call LLVM and Clang.

   Copyright (c) 2013 Kalle Raiskila and
                 2013-2019 Pekka Jääskeläinen

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

#include "pocl_cl.h"

#ifndef POCL_LLVM_H
#define POCL_LLVM_H

#ifdef __cplusplus
extern "C" {
#endif


  void InitializeLLVM ();
  void UnInitializeLLVM ();

  /* Returns the cpu name as reported by LLVM. */
  POCL_EXPORT
  char *pocl_get_llvm_cpu_name ();

  /* For distro builds, return the kernellib variant based on the host CPU */
  POCL_EXPORT
  const char *pocl_get_distro_kernellib_variant ();

  /* For distro builds, return the target cpu name for a kernellib name */
  POCL_EXPORT
  const char *pocl_get_distro_cpu_name (const char *kernellib_name);

  /* Returns if the cpu supports FMA instruction (uses LLVM). */
  int cpu_has_fma ();

  POCL_EXPORT
  int pocl_bitcode_is_triple (const char *bitcode, size_t size, const char *triple);

  /* Sets up the native/preferred vector widths at runtime (using LLVM). */
  void cpu_setup_vector_widths (cl_device_id dev);

  /* Compiles an .cl file into LLVM IR.
   */
  POCL_EXPORT
  int pocl_llvm_build_program (cl_program program, unsigned device_i,
                               cl_uint num_input_headers,
                               const cl_program *input_headers,
                               const char **header_include_names,
                               int linking_program);

  /* Retrieve metadata of the given kernel in the program to populate the
   * cl_kernel object.
   */
  int pocl_llvm_get_kernels_metadata (cl_program program, unsigned device_i);

  /* This function links the input kernel LLVM bitcode and the OpenCL kernel
   * runtime library into one LLVM module, then runs pocl's kernel compiler
   * passes on that module to produce a function that executes all work-items
   * in a work-group.
   *
   * Output is a LLVM bitcode file that contains a work-group function and its
   * associated launchers. If @param Specialize is set to true, generates a
   * WG function that might be specialized according to the properties of
   * the given Command.
   */
  POCL_EXPORT
  int pocl_llvm_generate_workgroup_function (unsigned DeviceI,
                                             cl_device_id Device,
                                             cl_kernel Kernel,
                                             _cl_command_node *Command,
                                             int Specialize);

  int pocl_llvm_generate_workgroup_function_nowrite (
      unsigned DeviceI, cl_device_id Device, cl_kernel Kernel,
      _cl_command_node *Command, void **output, int Specialize);

  POCL_EXPORT
  int pocl_llvm_run_passes_on_program (cl_program Program, unsigned DeviceI);
  /**
   * Free the LLVM IR of a program for a given device
   */
  POCL_EXPORT
  void pocl_llvm_free_llvm_irs (cl_program program, unsigned device_i);

  POCL_EXPORT
  void pocl_destroy_llvm_module (void *modp, cl_context ctx);

  int pocl_llvm_remove_file_on_signal (const char *file);

  void pocl_llvm_create_context (cl_context ctx);
  void pocl_llvm_release_context (cl_context ctx);

  /**
   * \brief Creates instance of a class that holds llvm::Module of input IR
   * (program.bc), plus its own LLVM contexts, so it can be safely used in
   * multithreaded env. Returns an output SPIR-V with Program-scope variables
   * (and optionally, non-kernel functions). This SPIR-V needs to be turned
   * into a native (device) module and linked with each JIT-compiled kernel
   * module, otherwise Program-scope variables will not work properly.
   *
   * \param [in] ProgramBcBytes in-memory IR of program.bc
   * \param [in] ProgramBcSize size of program.bc
   * \param [out] LinkinSpirvContent output SPIRV with prog-scope vars
   * \param [out] LinkinSpirvSize size of output LinkinSpirvContent
   * \returns opaque handle to instance holding the data, or NULL on error
   *
   */
  POCL_EXPORT
  void *pocl_llvm_create_context_for_program (const char *ProgramBcBytes,
                                              size_t ProgramBcSize,
                                              char **LinkinSpirvContent,
                                              uint64_t *LinkinSpirvSize);

  /**
  * \brief extracts SPIR-V of a single Kernel (plus all functions it uses)
  * from the program IR, converts it to SPIRV & returns it.
  *
  * \param [in] ProgCtx the handle from pocl_llvm_create_context_for_program
  * \param [in] KernelName name of the kernel to extract
  * \param [out] BuildLogStr handle (std::string *) of log with errors/warnings
  * \param [out] SpirvContent output SPIRV with the kernel
  * \param [out] SpirvSize size of output SpirvContent
  * \returns 0 on success
  *
  */
  POCL_EXPORT
  int pocl_llvm_extract_kernel_spirv(void* ProgCtx,
                                     const char* KernelName,
                                     void* BuildLogStr,
                                     char **SpirvContent,
                                     uint64_t *SpirvSize);

  /**
  * \brief destroys the instance of hidden class used to extract kernel SPIR-V
  *
  * \param [in] ProgCtx the  handle from pocl_llvm_create_context_for_program
  */
  POCL_EXPORT
  void pocl_llvm_release_context_for_program (void *ProgCtx);

  /**
   * Count the number of "__kernel" functions in 'program'.
   *
   * Results are valid as long as program binary is not modified.
   */
  unsigned pocl_llvm_get_kernel_count (cl_program program, unsigned device_i);

  /** Compile the kernel in infile from LLVM bitcode to native object file for
   * device, into outfile.
   */
  int pocl_llvm_codegen (cl_device_id device, cl_program program, void *modp,
                         char **output, uint64_t *output_size);

  /* Parse program file and populate program's llvm_irs */
  int pocl_llvm_read_program_llvm_irs (cl_program program, unsigned device_i,
                                       const char *path);

  int pocl_llvm_link_program (cl_program program, unsigned device_i,
                              cl_uint num_input_programs,
                              unsigned char **cur_device_binaries,
                              size_t *cur_device_binary_sizes,
                              void **cur_llvm_irs,
                              int link_device_builtin_library,
                              int linking_into_new_cl_program);

  int pocl_invoke_clang (cl_device_id Device, const char **Args);

#ifdef __cplusplus
}
#endif

#endif
