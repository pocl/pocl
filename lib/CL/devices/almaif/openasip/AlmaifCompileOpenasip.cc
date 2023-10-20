/* AlmaifCompileOpenasip.cc - compiler support for custom devices

   Copyright (c) 2022 Topi Leppänen / Tampere University

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

#include "stdint.h"
#include "unistd.h"

#include "common.h"
#include "pocl_cache.h"
#include "pocl_file_util.h"

#include "pocl_llvm.h"

#include <fstream>
#include <iostream>
#include <string>

#include <AddressSpace.hh>
#include <Environment.hh>
#include <Machine.hh>
#include <Procedure.hh>
#include <Program.hh>
#include <SimpleSimulatorFrontend.hh>
/*#include <CodeLabel.hh>
#include <DataLabel.hh>

#include <GlobalScope.hh>
#include <Program.hh>
*/

#include "../AlmaifCompile.hh"
#include "../AlmaifShared.hh"
#include "AlmaifCompileOpenasip.hh"

#include "TTASimDevice.hh"

int pocl_almaif_openasip_initialize(cl_device_id device,
                                    const std::string &parameters) {
  AlmaifData *d = (AlmaifData *)(device->data);

  openasip_backend_data_t *bd = new openasip_backend_data_t();
  if (bd == NULL) {
    POCL_MSG_WARN("couldn't allocate openasip_backend_data\n");
    return CL_OUT_OF_HOST_MEMORY;
  }

  POCL_INIT_LOCK(bd->openasip_compile_lock);

  if (1) // pocl_offline_compile
  {
    assert(parameters != "");
    /* Convert the filename from env variable to absolute filename.
     * This is required, since generatebits must be run in
     * destination (output) directory with ADF argument */
    char *tmp_path = realpath(parameters.c_str(), NULL);
    if ((tmp_path == NULL) || (!pocl_exists(tmp_path)))
      POCL_ABORT("Can't find ADF file: %s\n", tmp_path);
    bd->machine_file.assign(tmp_path);
    free(tmp_path);

    // snprintf (dev_name, 1024, "ALMAIF openasip: %s", bd->machine_file);

    /* grep the ADF file for endiannes flag */
    char *content = NULL;
    uint64_t size = 0;
    pocl_read_file(bd->machine_file.c_str(), &content, &size);
    if ((size == 0) || (content == NULL))
      POCL_ABORT("Can't read ADF file: %s\n", bd->machine_file);

    device->endian_little = (strstr(content, "<little-endian") != NULL);
    unsigned cores = 0;
    if (sscanf(content, "<adf core-count=\"%u\"", &cores)) {
      assert(cores > 0);
      bd->core_count = cores;
      device->max_compute_units = cores;
    } else
      bd->core_count = 1;
    POCL_MSG_PRINT_ALMAIF("Multicore: %u Cores: %u \n", bd->core_count > 1,
                         bd->core_count);
    POCL_MEM_FREE(content);
  } else {
    bd->machine_file = "";
    device->max_compute_units =
        d->Dev->ControlMemory->Read32(ALMAIF_INFO_CORE_COUNT);
  }

  device->long_name = device->short_name = "ALMAIF OPENASIP";
  device->vendor = "pocl";
  device->extensions = TCE_DEVICE_EXTENSIONS;
  if (device->endian_little) {
    device->llvm_target_triplet = "tcele-tut-llvm";
    device->kernellib_name = "kernel-tcele-tut-llvm";
  } else {
    device->llvm_target_triplet = "tce-tut-llvm";
    device->kernellib_name = "kernel-tce-tut-llvm";
  }
  device->kernellib_fallback_name = NULL;
  device->kernellib_subdir = "tce";
  device->llvm_cpu = NULL;
  d->compilationData->backend_data = (void *)bd;
  device->builtins_sources_path = "tce_builtins.cl";

  return 0;
}

int pocl_almaif_openasip_cleanup(cl_device_id device) {
  void *data = device->data;
  AlmaifData *d = (AlmaifData *)data;

  if (device->device_side_printf) {
    pocl_free_chunk((chunk_info_t *)d->PrintfBuffer);
    pocl_free_chunk((chunk_info_t *)d->PrintfPosition);
  }

  openasip_backend_data_t *bd =
      (openasip_backend_data_t *)d->compilationData->backend_data;

  POCL_DESTROY_LOCK(bd->openasip_compile_lock);

  delete bd;

  return 0;
}

#define SUBST(x) "  -DKERNEL_EXE_CMD_OFFSET=" #x
#define OFFSET_ARG(c) SUBST(c)
#define MAX_CMDLINE_LEN (32 * POCL_MAX_PATHNAME_LENGTH)

std::string oaccCommandLine(_cl_command_run *run_cmd, AlmaifData *D,
                            const std::string &tempDir,
                            const std::string &inputSrc,
                            const std::string &outputTpef,
                            const std::string &machine_file, int is_multicore,
                            int little_endian, const std::string &extraParams,
                            bool standalone_mode) {
  std::string mainC;
  if (is_multicore)
    mainC = "tta_device_main_dthread.c";
  else
    mainC = "tta_device_main.c";

  std::string deviceMainSrc;
  std::string poclIncludePathSwitch;
  if (pocl_get_bool_option("POCL_BUILDING", 0)) {
    deviceMainSrc =
        std::string(SRCDIR) + "/lib/CL/devices/almaif/openasip/" + mainC;
    assert(access(deviceMainSrc.c_str(), R_OK) == 0);
    poclIncludePathSwitch = " -I " + std::string(SRCDIR) + "/include" + " -I " +
                            std::string(SRCDIR) +
                            "/lib/CL/devices/almaif/openasip";
  } else {
    deviceMainSrc = std::string(POCL_INSTALL_PRIVATE_DATADIR) + "/" + mainC;
    assert(access(deviceMainSrc.c_str(), R_OK) == 0);
    poclIncludePathSwitch =
        " -I " + std::string(POCL_INSTALL_PRIVATE_DATADIR) + "/include";
  }

  std::string extraFlags;
  std::string multicoreFlags = "";
  if (is_multicore)
    multicoreFlags = " -ldthread -lsync-lu -llockunit";

  std::string preprocessor_directives =
      set_preprocessor_directives(D, machine_file, standalone_mode);

  const std::string userFlags =
      pocl_get_string_option("POCL_TCECC_EXTRA_FLAGS", "");
  const std::string endianFlags = little_endian ? "--little-endian" : "";
  extraFlags = extraParams + " " + multicoreFlags + " " + userFlags + " " +
               endianFlags + " " + preprocessor_directives +
               " -k dummy_argbuffer";

  std::string kernelObjSrc = tempDir + "/../descriptor.so.kernel_obj.c";

  std::string kernelMdSymbolName =
      "_" + std::string(run_cmd->kernel->name) + "_md";

  std::string programBcFile = tempDir + "/program.bc";

  /* Compile in steps to save the program.bc for automated exploration
     use case when producing the kernel capture scripts. */

  std::string commandline =
      "oacc -llwpr " + poclIncludePathSwitch + " " + deviceMainSrc + " " +
      kernelObjSrc + " " + inputSrc + " -k " + kernelMdSymbolName +
      " -g -O3 --emit-llvm" + " -o " + programBcFile + " " + extraFlags + ";" +
      "oacc -a " + machine_file + " " + programBcFile + " -O3 -o " +
      outputTpef + " " + extraFlags + "\n";
  return commandline;
}

void pocl_openasip_write_kernel_descriptor(char *content, size_t content_size,
                                           _cl_command_node *command,
                                           cl_kernel kernel,
                                           cl_device_id device,
                                           int specialize) {
  // Generate the kernel_obj.c file. This should be optional
  // and generated only for the heterogeneous standalone devices which
  // need the definitions to accompany the kernels, for the launcher
  // code.
  // TODO: the scripts use a generated kernel.h header file that
  // gets added to this file. No checks seem to fail if that file
  // is missing though, so it is left out from there for now

  char *orig_content = content;

  pocl_kernel_metadata_t *meta = kernel->meta;

  snprintf(content, content_size,
           "\n#include <pocl_device.h>\n"
           "void _pocl_kernel_%s"
           "_workgroup(uint8_t* args, uint8_t*, "
           "uint32_t, uint32_t, uint32_t);\n"
           "void _pocl_kernel_%s"
           "_workgroup_fast(uint8_t* args, uint8_t*, "
           "uint32_t, uint32_t, uint32_t);\n"

           "void %s"
           "_workgroup_argbuffer("
           "uint8_t "
           "__attribute__((address_space(%u)))"
           "* args, "
           "uint8_t "
           "__attribute__((address_space(%u)))"
           "* ctx, "
           "uint32_t, uint32_t, uint32_t);\n",
           meta->name, meta->name, meta->name, device->global_as_id,
           device->global_as_id);
  size_t content_len = strlen(content);
  assert(content_len < content_size);
  content += content_len;
  content_size -= content_len;
  snprintf(content, content_size,
           "void* dummy_argbuffer = %s_workgroup_argbuffer;\n", meta->name);

  content_len = strlen(orig_content);
  pocl_cache_write_descriptor(command, kernel, specialize, orig_content,
                              content_len);
}

void pocl_almaif_openasip_compile(_cl_command_node *cmd, cl_kernel kernel,
                                  cl_device_id device, int specialize) {

  if (cmd->type != CL_COMMAND_NDRANGE_KERNEL) {
    POCL_ABORT("Almaif: trying to compile non-ndrange command\n");
  }

  void *data = cmd->device->data;
  AlmaifData *d = (AlmaifData *)data;
  openasip_backend_data_t *bd =
      (openasip_backend_data_t *)d->compilationData->backend_data;

  if (!kernel)
    kernel = cmd->command.run.kernel;
  if (!device)
    device = cmd->device;
  assert(kernel);
  assert(device);
  POCL_MSG_PRINT_ALMAIF("COMPILATION BEFORE WG FUNC\n");
  POCL_LOCK(bd->openasip_compile_lock);
  int error = pocl_llvm_generate_workgroup_function(
      cmd->program_device_i, device, kernel, cmd, specialize);

  POCL_MSG_PRINT_ALMAIF("COMPILATION AFTER WG FUNC\n");
  if (error) {
    POCL_UNLOCK(bd->openasip_compile_lock);
    POCL_ABORT("TCE: pocl_llvm_generate_workgroup_function()"
               " failed for kernel %s\n",
               kernel->name);
  }

  // 12 == strlen (POCL_PARALLEL_BC_FILENAME)
  char inputBytecode[POCL_MAX_PATHNAME_LENGTH + 13];

  assert(d != NULL);
  assert(cmd->command.run.kernel);

  char cachedir[POCL_MAX_PATHNAME_LENGTH];
  pocl_cache_kernel_cachedir_path(cachedir, kernel->program,
                                  cmd->program_device_i, kernel, "", cmd,
                                  specialize);
  cmd->command.run.device_data = strdup(cachedir);

  // output TPEF
  char assemblyFileName[POCL_MAX_PATHNAME_LENGTH];
  snprintf(assemblyFileName, POCL_MAX_PATHNAME_LENGTH, "%s%s", cachedir,
           "/parallel.tpef");

  char tempDir[POCL_MAX_PATHNAME_LENGTH];
  strncpy(tempDir, cachedir, POCL_MAX_PATHNAME_LENGTH);

  if (!pocl_exists(assemblyFileName)) {
    char descriptor_content[64 * 1024];
    pocl_openasip_write_kernel_descriptor(descriptor_content, (64 * 1024), cmd,
                                          kernel, device, specialize);

    error = snprintf(inputBytecode, POCL_MAX_PATHNAME_LENGTH, "%s%s", cachedir,
                     POCL_PARALLEL_BC_FILENAME);

    std::string commandLine =
        oaccCommandLine(&cmd->command.run, d, tempDir,
                        inputBytecode, // inputSrc
                        assemblyFileName, bd->machine_file, bd->core_count > 1,
                        device->endian_little, "", false);

    POCL_MSG_PRINT_ALMAIF("build command: \n%s", commandLine.c_str());

    error = system(commandLine.c_str());
    if (error != 0)
      POCL_ABORT("Error while running oacc.\n");

    // Dump disassembled tpef for debugging
    char OpenasipDisAsmCmd[MAX_CMDLINE_LEN];
    snprintf(OpenasipDisAsmCmd, MAX_CMDLINE_LEN, "tcedisasm -n %s %s",
             bd->machine_file.c_str(), assemblyFileName);
    error = system(OpenasipDisAsmCmd);
    if (error != 0)
        POCL_MSG_WARN("Error while running tcedisasm.\n");
  }

  char md_path[POCL_MAX_PATHNAME_LENGTH];
  snprintf(md_path, POCL_MAX_PATHNAME_LENGTH, "%s/kernel_address.txt",
           cachedir);

  if (!pocl_exists(md_path)) {
    TTAMachine::Machine *mach = NULL;
    try {
        mach = TTAMachine::Machine::loadFromADF(bd->machine_file);
    } catch (Exception &e) {
        POCL_MSG_WARN("Error: %s\n", e.errorMessage().c_str());
        POCL_ABORT("Couldn't open mach\n");
    }

    TTAProgram::Program *prog = NULL;
    try {
        prog = TTAProgram::Program::loadFromTPEF(assemblyFileName, *mach);
    } catch (Exception &e) {
        delete mach;
        POCL_MSG_WARN("Error: %s\n", e.errorMessage().c_str());
        POCL_ABORT("Couldn't open tpef %s after compilation\n",
                   assemblyFileName);
    }

    std::string wg_func_name =
        std::string(cmd->command.run.kernel->name) + "_workgroup_argbuffer";
    if (prog->hasProcedure(wg_func_name)) {
        const TTAProgram::Procedure &proc = prog->procedure(wg_func_name);
        int kernel_address = proc.startAddress().location();

        std::string md_path = std::string(cachedir) + "/kernel_address.txt";
        std::string content =
            "kernel address = " + std::to_string(kernel_address);
        pocl_write_file(md_path.c_str(), content.c_str(), content.length(), 0,
                        0);
    } else {
        POCL_ABORT("Couldn't find wg_function procedure %s from the program\n",
                   wg_func_name);
    }
    delete prog;
    delete mach;
  }

  char imem_file[POCL_MAX_PATHNAME_LENGTH];
  snprintf(imem_file, POCL_MAX_PATHNAME_LENGTH, "%s%s", cachedir,
           "/parallel.img");

  if (!pocl_exists(imem_file)) {
    std::string genbits_command =
        "SAVEDIR=$PWD; cd " + std::string(cachedir) +
        "; generatebits --dmemwidthinmaus 4 " +
        "--piformat=bin2n --diformat=bin2n --program " + "parallel.tpef " +
        bd->machine_file + "; cd $SAVEDIR";
    POCL_MSG_PRINT_ALMAIF("running genbits: \n %s \n", genbits_command.c_str());
    error = system(genbits_command.c_str());
    if (error != 0)
      POCL_ABORT("Error while running generatebits.\n");
  }

  error = pocl_exists(imem_file);
  assert(error != 0 && "parallel.img does not exist!");

  POCL_UNLOCK(bd->openasip_compile_lock);
}

/* This is a version number that is supposed to increase when there is
 * a change in TCE or ALMAIF drivers that makes previous pocl-binaries
 * incompatible (e.g. a change in generated device image file names, etc) */
#define POCL_TCE_ALMAIF_BINARY_VERSION "2"

int pocl_almaif_openasip_device_hash(const char *adf_file,
                                     const char *llvm_triplet, char *output) {

  SHA1_CTX ctx;
  uint8_t bin_dig[SHA1_DIGEST_SIZE];

  char *content;
  uint64_t size;
  int err = pocl_read_file(adf_file, &content, &size);
  if (err || (content == NULL) || (size == 0))
    POCL_ABORT("Can't find ADF file %s \n", adf_file);

  pocl_SHA1_Init(&ctx);
  pocl_SHA1_Update(&ctx, (const uint8_t *)POCL_TCE_ALMAIF_BINARY_VERSION, 1);
  pocl_SHA1_Update(&ctx, (const uint8_t *)llvm_triplet, strlen(llvm_triplet));
  pocl_SHA1_Update(&ctx, (uint8_t *)content, size);

  if (pocl_is_option_set("POCL_TCECC_EXTRA_FLAGS")) {
    const char *extra_flags =
        pocl_get_string_option("POCL_TCECC_EXTRA_FLAGS", "");
    pocl_SHA1_Update(&ctx, (uint8_t *)extra_flags, strlen(extra_flags));
  }

  pocl_SHA1_Final(&ctx, bin_dig);

  unsigned i;
  for (i = 0; i < SHA1_DIGEST_SIZE; i++) {
    *output++ = (bin_dig[i] & 0x0F) + 65;
    *output++ = ((bin_dig[i] & 0xF0) >> 4) + 65;
  }
  *output = 0;
  return 0;
}

char *pocl_almaif_openasip_init_build(void *data) {
  AlmaifData *D = (AlmaifData *)data;
  openasip_backend_data_t *bd =
      (openasip_backend_data_t *)D->compilationData->backend_data;
  assert(bd);

  TCEString mach_tmpdir = Environment::llvmtceCachePath();

  // $HOME/.tce/tcecc/cache may not exist yet, create it here.
  pocl_mkdir_p(mach_tmpdir.c_str());

  TCEString mach_header_base =
      mach_tmpdir + "/" +
      ((TTASimDevice *)(D->Dev))->simulator_->machine().hash();

  int error = 0;

  std::string devextHeaderFn =
      std::string(mach_header_base) + std::string("_opencl_devext.h");

  /* Generate the vendor extensions header to provide explicit
     access to the (custom) hardware operations. */
  // to avoid threading issues, generate to tempfile then rename
  if (!pocl_exists(devextHeaderFn.c_str())) {
    char tempfile[POCL_MAX_PATHNAME_LENGTH];
    pocl_mk_tempname(tempfile, mach_tmpdir.c_str(), ".devext", NULL);

    std::string OpenasipOpgenCmd = std::string("tceopgen > ") + tempfile;

    POCL_MSG_PRINT_TCE("Running: %s \n", OpenasipOpgenCmd.c_str());

    error = system(OpenasipOpgenCmd.c_str());
    if (error == -1)
      return NULL;

    std::string extgenCmd = std::string("tceoclextgen ") + bd->machine_file +
                            std::string(" >> ") + tempfile;

    POCL_MSG_PRINT_TCE("Running: %s \n", extgenCmd.c_str());

    error = system(extgenCmd.c_str());
    if (error == -1)
      return NULL;

    pocl_rename(tempfile, devextHeaderFn.c_str());
  }

  // gnu-keywords needed to support the inline asm blocks
  // -fasm doesn't work in the frontend
  std::string includeSwitch =
      std::string("-fgnu-keywords -Dasm=__asm__ -include ") + devextHeaderFn;

  char *include_switch = strdup(includeSwitch.c_str());

  return include_switch;
}

void pocl_almaif_openasip_produce_standalone_program(AlmaifData *D,
                                                     _cl_command_node *cmd,
                                                     pocl_context32 *pc,
                                                     size_t arg_size,
                                                     void *arguments) {
  _cl_command_run *run_cmd = &cmd->command.run;

  static int runCounter = 0;
  TCEString tempDir((const char *)run_cmd->device_data);
  TCEString baseFname = tempDir + "/";
  baseFname << "standalone_" << runCounter;

  TCEString buildScriptFname = "standalone_";
  buildScriptFname << runCounter << "_build";
  TCEString fname = baseFname + ".c";
  TCEString parallel_bc = tempDir + "/parallel.bc";

  openasip_backend_data_t *bd =
      (openasip_backend_data_t *)D->compilationData->backend_data;

  std::ofstream out(fname.c_str());

  out << "#include <pocl_device.h>" << std::endl;
  out << "#include \"almaif-tce-device-defs.h\"" << std::endl << std::endl;

  out << "#undef ALIGN4" << std::endl;
  out << "#define ALIGN4 __attribute__ ((aligned (4)))" << std::endl;

  /* The standalone binary shall have the same input data as in the original
     kernel host-device kernel launch command. The data is put into
     initialized global arrays to easily exclude the initialization time from
     the execution time. Otherwise, the same command data is used for
     reproducing the execution. For example, the local memory allocations
     (addresses) are the same as in the original one. */

  /* Create the global buffers along with their initialization data. */
  cl_kernel kernel = run_cmd->kernel;
  pocl_kernel_metadata_t *meta = kernel->meta;
  uint32_t gmem_count = 0;
  /* store addresses used for buffer names later*/
  uint32_t gmem_ptr_offsets[1024];
  uint32_t gmem_startaddrs[1024];
  uint32_t gmem_sizes[1024];
  uint32_t write_pos = 0;

  for (size_t i = 0; i < meta->num_args; ++i) {
    struct pocl_argument *al = &(run_cmd->arguments[i]);
    if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) {
      if (al->value == NULL)
        continue;

      cl_mem m = *(cl_mem *)(al->value);
      void *p = m->device_ptrs[cmd->device->global_mem_id].mem_ptr;
      chunk_info_t *ci = (chunk_info_t *)p;
      unsigned start_addr = ci->start_address + al->offset;
      unsigned size = ci->size;

      out << "__global__ ALIGN4 char buffer_" << std::hex << start_addr
          << "[] = {\n";

      gmem_startaddrs[gmem_count] = start_addr;
      gmem_sizes[gmem_count] = size;
      gmem_ptr_offsets[gmem_count++] = write_pos;
      write_pos += 4;
      unsigned char *tmp_buffer = (unsigned char *)malloc(size);
      assert(tmp_buffer);
      D->Dev->DataMemory->CopyFromMMAP(tmp_buffer, start_addr, size);
      for (std::size_t c = 0; c < size; ++c) {
        if (c % 32 == 0)
          out << std::endl << "\t";
        out << "0x" << std::hex << (unsigned int)tmp_buffer[c];
        if (c + 1 < size)
          out << ", ";
      }
      out << std::endl << "};" << std::endl << std::endl;
    } else if (ARG_IS_LOCAL(meta->arg_info[i])) {
      write_pos += 4;
    } else {
      write_pos += al->size;
    }
  }

  /* Scalars (+addresses of global/local buffers) are stored to a single
   * global buffer. */
  {
    out << "__global__ ALIGN4 char arg_buffer[] = {" << std::endl << "\t";

    for (std::size_t c = 0; c < arg_size; ++c) {
      unsigned char val = ((unsigned char *)arguments)[c];
      out << "0x" << std::hex << (unsigned int)val;
      if (c + 1 < arg_size)
        out << ", ";
      if (c % 32 == 31)
        out << std::endl << "\t";
    }
    out << std::endl << "};" << std::endl << std::endl;
  }

  /* pocl_context is stored in a global buffer too; copy it. */
  {
    char *pc_char_ptr = (char *)pc;
    unsigned size = sizeof(pocl_context32);

    out << "__global__ ALIGN4 char ctx_buffer[] = {" << std::endl << "\t";

    for (std::size_t c = 0; c < size; ++c) {
      unsigned char val = pc_char_ptr[c];
      out << "0x" << std::hex << (unsigned int)val;
      if (c + 1 < size)
        out << ", ";
      if (c % 32 == 31)
        out << std::endl << "\t";
    }
    out << std::endl << "};" << std::endl << std::endl;
  }

  /* Setup the kernel command initialization values, pointing to the
     global buffers for the buffer arguments, and using the original values
     for the rest. */

  out << "__global__ int __completion_signal = 0;" << std::endl;

  out << "void " << meta->name << "_workgroup_argbuffer("
      << "uint8_t "
      << "__attribute__((address_space(" << TTA_ASID_GLOBAL << ")))"
      << "* args, "
      << "uint8_t "
      << "__attribute__((address_space(" << TTA_ASID_GLOBAL << ")))"
      << "* ctx, "
      << "uint32_t, uint32_t, uint32_t);" << std::endl;

  out << "__cq__ ALIGN4 struct AQLDispatchPacket standalone_packet;"
      << std::endl;

  out << std::endl;
  out << "__attribute__((noinline))" << std::endl;
  out << "void initialize_kernel_launch() {" << std::endl;

  // update the args pointers with actual addresses
  out << "\t__global__ uint32_t* global_buffer_addr = 0;\n";
  for (size_t i = 0; i < gmem_count; ++i) {
    out << "\tglobal_buffer_addr = (__global__ uint32_t*)(arg_buffer + "
        << std::dec << gmem_ptr_offsets[i] << ");\n";
    out << "\t*global_buffer_addr = (uint32_t)buffer_" << std::hex
        << gmem_startaddrs[i] << ";" << std::endl;
  }
  out << "\t__cq__ uint32_t* aql_read_iter = (__cq__ uint32_t*) (QUEUE_START + "
         "0x"
      << std::hex << ALMAIF_CQ_READ << ");" << std::endl
      << "\t*aql_read_iter = 0;" << std::endl;

  out << "\tstandalone_packet.header = " << std::hex << "(uint32_t)0x"
      << (1 << AQL_PACKET_KERNEL_DISPATCH) << ";\n"
      << "\tstandalone_packet.dimensions = " << std::hex << "(uint32_t)0x"
      << run_cmd->pc.work_dim << ";\n"
      << "\tstandalone_packet.workgroup_size_x = " << std::hex << "(uint32_t)0x"
      << run_cmd->pc.local_size[0] << ";\n"
      << "\tstandalone_packet.workgroup_size_y = " << std::hex << "(uint32_t)0x"
      << run_cmd->pc.local_size[1] << ";\n"
      << "\tstandalone_packet.workgroup_size_z = " << std::hex << "(uint32_t)0x"
      << run_cmd->pc.local_size[2] << ";\n"

      << "\tstandalone_packet.reserved1 = (uint32_t)ctx_buffer"
      << ";\n"
      << "\tstandalone_packet.kernarg_address_low = (uint32_t)arg_buffer"
      << ";\n"

      << "\tstandalone_packet.kernel_object_low = (uint32_t)&"
      << run_cmd->kernel->meta->name << "_workgroup_argbuffer;\n"

      << "\tstandalone_packet.cmd_metadata_low = (uint32_t)&__completion_signal"
      << ";\n";

  out << "}" << std::endl;
  out.close();

  // Create the build script.
  TCEString inputFiles = fname + " " + parallel_bc;
  std::ofstream scriptout(buildScriptFname.c_str());

  std::string commandLine = oaccCommandLine(
      run_cmd, D, tempDir.c_str(), inputFiles.c_str(), "standalone.tpef",
      bd->machine_file, bd->core_count > 1, 1, " -D_STANDALONE_MODE=1", true);
  scriptout << commandLine;
  scriptout.close();

  TCEString simScriptFname = "standalone_";
  simScriptFname << runCounter << "_ttasim";
  std::ofstream simScript(simScriptFname.c_str());
  simScript << "mach " << bd->machine_file << ";" << std::endl;
  simScript << "prog standalone.tpef;" << std::endl;
  simScript << "run;" << std::endl;

  for (size_t i = 0; i < gmem_count; ++i) {
    simScript << "x /u w /n " << std::dec << gmem_sizes[i] / 4 << " buffer_"
              << std::hex << gmem_startaddrs[i] << ";" << std::endl;
  }
  simScript.close();

  ++runCounter;
}

std::string set_preprocessor_directives(AlmaifData *d, const std::string &adf,
                                        bool standalone_mode) {
  TTAMachine::Machine *mach = NULL;
  std::string output = "";
  try {
    mach = TTAMachine::Machine::loadFromADF(adf);
  } catch (Exception &e) {
    POCL_MSG_WARN("Error: %s\n", e.errorMessage().c_str());
    POCL_ABORT("Couldn't open mach\n");
  }
  char private_as_name[MAX_CMDLINE_LEN] = "";
  char global_as_name[MAX_CMDLINE_LEN] = "";

  bool separatePrivateMem = true;
  bool separateCQMem = true;
  const TTAMachine::Machine::AddressSpaceNavigator &nav =
      mach->addressSpaceNavigator();
  for (int i = 0; i < nav.count(); i++) {
    TTAMachine::AddressSpace *as = nav.item(i);
    if (as->hasNumericalId(TTA_ASID_GLOBAL)) {
      if (as->hasNumericalId(TTA_ASID_CQ)) {
        separateCQMem = false;
      }
      if (as->hasNumericalId(TTA_ASID_PRIVATE)) {
        separatePrivateMem = false;
      }
      strcpy(global_as_name, as->name().c_str());
    }
    if (as->hasNumericalId(TTA_ASID_PRIVATE)) {
      strcpy(private_as_name, as->name().c_str());
    }
  }
  if (private_as_name == "") {
    POCL_ABORT("Couldn't find the private address space from machine\n");
  }
  if (global_as_name == "") {
    POCL_ABORT("Couldn't find the global address space from machine\n");
  }

  int AQL_queue_length = d->Dev->CQMemory->Size() / AQL_PACKET_LENGTH - 1;
  unsigned DmemSize = d->Dev->DataMemory->Size();
  unsigned CQSize = d->Dev->CQMemory->Size();

  bool relativeAddressing = d->Dev->RelativeAddressing;
  int i = 0;
  output += "-DQUEUE_LENGTH=" + std::to_string(AQL_queue_length) + " ";
  if (!separatePrivateMem) {
    unsigned initsp = DmemSize;
    unsigned private_mem_start = 0;
    if (!standalone_mode) {
      // The standalone mode, cannot separate the automatic allocation of
      // OpenCL buffers from the other use of the stack and heap.
      // Therefore, only if we are not using the standalone mode
      // we can separate the private_mem to an independent region of the data
      // memory. Another alternative for this would be to change openasip to
      // support multiple address spaces in a single LSU, and automatically
      // connect the address spaces with a hardware decoder.
      int private_mem_size = pocl_get_int_option(
          "POCL_ALMAIF_PRIVATE_MEM_SIZE", ALMAIF_DEFAULT_PRIVATE_MEM_SIZE);
      initsp += private_mem_size;
      private_mem_start += DmemSize;
      if (!separateCQMem) {
        initsp += CQSize;
        private_mem_start += CQSize;
      }
    }

    if (!relativeAddressing) {
      initsp += d->Dev->DataMemory->PhysAddress();
      private_mem_start += d->Dev->DataMemory->PhysAddress();
    }
    output += "--init-sp=";
    output += std::to_string(initsp);
    output += " --data-start=";
    output += private_as_name;
    output += ",";
    output += std::to_string(private_mem_start);
  }
  if (!relativeAddressing && standalone_mode) {
    // Appends to the data-start option
    std::string data_start_option_string;
    if (!separatePrivateMem) {
      data_start_option_string = ",";
    } else {
      data_start_option_string = " --data-start=";
    }
    output += data_start_option_string + global_as_name +
              ",${STANDALONE_GLOBAL_AS_OFFSET}";
  }

  if (!separateCQMem) {
    unsigned queue_start = d->Dev->CQMemory->PhysAddress();
    if (relativeAddressing) {
      queue_start -= d->Dev->DataMemory->PhysAddress();
    }
    output += " -DQUEUE_START=";
    output += std::to_string(queue_start);
    output += " ";
  }

  delete mach;

  return output;
}
