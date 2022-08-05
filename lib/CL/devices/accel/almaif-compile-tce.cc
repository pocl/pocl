#include "stdint.h"
#include "unistd.h"

#include "common.h"
#include "pocl_cache.h"
#include "pocl_file_util.h"

#include "pocl_llvm.h"

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

#include "accel-shared.h"
#include "almaif-compile-tce.h"
#include "almaif-compile.h"

#include "TTASimDevice.h"

int pocl_almaif_tce_initialize(cl_device_id device, const char *parameters) {
  AccelData *d = (AccelData *)(device->data);

  tce_backend_data_t *bd = (tce_backend_data_t *)pocl_aligned_malloc(
      HOST_CPU_CACHELINE_SIZE, sizeof(tce_backend_data_t));
  if (bd == NULL) {
    POCL_MSG_WARN("couldn't allocate tce_backend_data\n");
    return CL_OUT_OF_HOST_MEMORY;
  }

  POCL_INIT_LOCK(bd->tce_compile_lock);

  if (1) // pocl_offline_compile
  {
    assert(parameters);
    /* Convert the filename from env variable to absolute filename.
     * This is required, since generatebits must be run in
     * destination (output) directory with ADF argument */
    bd->machine_file = realpath(parameters, NULL);
    if ((bd->machine_file == NULL) || (!pocl_exists(bd->machine_file)))
      POCL_ABORT("Can't find ADF file: %s\n", bd->machine_file);

    size_t len = strlen(bd->machine_file);
    assert(len > 0);
    // char* dev_name = malloc (len+20);
    // snprintf (dev_name, 1024, "ALMAIF TCE: %s", bd->machine_file);

    /* grep the ADF file for endiannes flag */
    char *content = NULL;
    uint64_t size = 0;
    pocl_read_file(bd->machine_file, &content, &size);
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
    POCL_MSG_PRINT_INFO("Multicore: %u Cores: %u \n", bd->core_count > 1,
                        bd->core_count);
    POCL_MEM_FREE(content);
  } else {
    bd->machine_file = NULL;
    device->max_compute_units = d->Dev->ControlMemory->Read32(ACCEL_INFO_CORE_COUNT);
  }

  device->long_name = device->short_name = "ALMAIF TCE";
  device->vendor = "pocl";
  device->extensions = "";
  if (device->endian_little)
    device->llvm_target_triplet = "tcele-tut-llvm";
  else
    device->llvm_target_triplet = "tce-tut-llvm";
  device->llvm_cpu = NULL;
  d->compilationData->backend_data = (void *)bd;
  device->builtins_sources_path = "tce_builtins.cl";

  return 0;
}

int pocl_almaif_tce_cleanup(cl_device_id device) {
  void *data = device->data;
  AccelData *d = (AccelData *)data;

  tce_backend_data_t *bd =
      (tce_backend_data_t *)d->compilationData->backend_data;

  POCL_DESTROY_LOCK(bd->tce_compile_lock);

  POCL_MEM_FREE(bd->machine_file);

  pocl_aligned_free(bd);

  return 0;
}

#define SUBST(x) "  -DKERNEL_EXE_CMD_OFFSET=" #x
#define OFFSET_ARG(c) SUBST(c)

void tceccCommandLine(char *commandline, size_t max_cmdline_len,
                      _cl_command_run *run_cmd, const char *tempDir,
                      const char *inputSrc, const char *outputTpef,
                      const char *machine_file, int is_multicore,
                      int little_endian, const char *extraParams) {

  const char *mainC;
  if (is_multicore)
    mainC = "tta_device_main_dthread.c";
  else
    mainC = "tta_device_main.c";

  char deviceMainSrc[POCL_FILENAME_LENGTH];
  const char *poclIncludePathSwitch;
  if (pocl_get_bool_option("POCL_BUILDING", 0)) {
    snprintf(deviceMainSrc, POCL_FILENAME_LENGTH, "%s%s%s", SRCDIR,
             "/lib/CL/devices/tce/", mainC);
    assert(access(deviceMainSrc, R_OK) == 0);
    poclIncludePathSwitch = " -I " SRCDIR "/include";
  } else {
    snprintf(deviceMainSrc, POCL_FILENAME_LENGTH, "%s%s%s",
             POCL_INSTALL_PRIVATE_DATADIR, "/", mainC);
    assert(access(deviceMainSrc, R_OK) == 0);
    poclIncludePathSwitch = " -I " POCL_INSTALL_PRIVATE_DATADIR "/include";
  }

  char extraFlags[POCL_FILENAME_LENGTH * 8];
  const char *multicoreFlags = "";
  if (is_multicore)
    multicoreFlags = " -ldthread -lsync-lu -llockunit";



  const char *userFlags = pocl_get_string_option("POCL_TCECC_EXTRA_FLAGS", "");
  const char *endianFlags = little_endian ? "--little-endian" : "";
  snprintf(extraFlags, (POCL_FILENAME_LENGTH * 8),
           "%s %s %s %s -k dummy_argbuffer",
           extraParams, multicoreFlags, userFlags, endianFlags);


  char kernelObjSrc[POCL_FILENAME_LENGTH];
  snprintf(kernelObjSrc, POCL_FILENAME_LENGTH, "%s%s", tempDir,
           "/../descriptor.so.kernel_obj.c");

  char kernelMdSymbolName[POCL_FILENAME_LENGTH];
  snprintf(kernelMdSymbolName, POCL_FILENAME_LENGTH, "_%s_md",
           run_cmd->kernel->name);

  char programBcFile[POCL_FILENAME_LENGTH];
  snprintf(programBcFile, POCL_FILENAME_LENGTH, "%s%s", tempDir, "/program.bc");

  /* Compile in steps to save the program.bc for automated exploration
     use case when producing the kernel capture scripts. */

  snprintf(commandline, max_cmdline_len,
           "tcecc -llwpr %s %s %s %s -k %s -g -O3 --emit-llvm -o %s %s;"
           "tcecc -a %s %s -O3 -o %s %s\n",
           poclIncludePathSwitch, deviceMainSrc, kernelObjSrc, inputSrc,
           kernelMdSymbolName, programBcFile, extraFlags,

           machine_file, programBcFile, outputTpef, extraFlags);
}

void pocl_tce_write_kernel_descriptor(char *content, size_t content_size,
                                      _cl_command_node *command,
                                      cl_kernel kernel, cl_device_id device,
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
  /*
    size_t content_len = strlen(content);
    assert (content_len < content_size);
    content += content_len;
    content_size -= content_len;

    if (device->global_as_id != 0)
      snprintf(content, content_size, "__attribute__((address_space(%u)))\n",
      device->global_as_id);

    content_len = strlen(content);
    assert (content_len < content_size);
    content += content_len;
    content_size -= content_len;

    snprintf(content, content_size,
             "__kernel_metadata _%s_md = {\n"
            "     \"%s\",\n"
            "     %u,\n"
            "     %u,\n"
            "     %s_workgroup_argbuffer\n"
            " };\n",
            meta->name,
            meta->name,
            meta->num_args,
            meta->num_locals,
            meta->name);
  */
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

#define MAX_CMDLINE_LEN (32 * POCL_FILENAME_LENGTH)

void pocl_almaif_tce_compile(_cl_command_node *cmd, cl_kernel kernel,
                            cl_device_id device, int specialize) {

  if (cmd->type != CL_COMMAND_NDRANGE_KERNEL) {
    POCL_ABORT("Accel: trying to compile non-ndrange command\n");
  }

  void *data = cmd->device->data;
  AccelData *d = (AccelData *)data;
  tce_backend_data_t *bd =
      (tce_backend_data_t *)d->compilationData->backend_data;

  if (!kernel)
    kernel = cmd->command.run.kernel;
  if (!device)
    device = cmd->device;
  assert(kernel);
  assert(device);
  POCL_MSG_PRINT_INFO("COMPILATION BEFORE WG FUNC\n");
  POCL_LOCK(bd->tce_compile_lock);
  int error = pocl_llvm_generate_workgroup_function(cmd->program_device_i, device,
                                                    kernel, cmd, specialize);

  POCL_MSG_PRINT_INFO("COMPILATION AFTER WG FUNC\n");
  if (error) {
    POCL_UNLOCK(bd->tce_compile_lock);
    POCL_ABORT("TCE: pocl_llvm_generate_workgroup_function()"
                 " failed for kernel %s\n",
                 kernel->name);
  }

  // 12 == strlen (POCL_PARALLEL_BC_FILENAME)
  char inputBytecode[POCL_FILENAME_LENGTH + 13];

  assert(d != NULL);
  assert(cmd->command.run.kernel);

  char cachedir[POCL_FILENAME_LENGTH];
  pocl_cache_kernel_cachedir_path(cachedir, kernel->program, cmd->program_device_i,
                                  kernel, "", cmd, specialize);

  // output TPEF
  char assemblyFileName[POCL_FILENAME_LENGTH];
  snprintf(assemblyFileName, POCL_FILENAME_LENGTH, "%s%s", cachedir,
           "/parallel.tpef");

  char tempDir[POCL_FILENAME_LENGTH];
  strncpy(tempDir, cachedir, POCL_FILENAME_LENGTH);

  TTAMachine::Machine *mach = NULL;
  try {
    mach = TTAMachine::Machine::loadFromADF(bd->machine_file);
  } catch (Exception &e) {
    POCL_MSG_WARN("Error: %s\n", e.errorMessage().c_str());
    POCL_ABORT("Couldn't open mach\n");
  }

  if (!pocl_exists(assemblyFileName)) {
    char descriptor_content[64 * 1024];
    pocl_tce_write_kernel_descriptor(descriptor_content, (64 * 1024), cmd,
                                     kernel, device, specialize);

    error = snprintf(inputBytecode, POCL_FILENAME_LENGTH, "%s%s", cachedir,
                     POCL_PARALLEL_BC_FILENAME);

    //int AQL_queue_length = pocl_get_int_option("POCL_AQL_QUEUE_LENGTH",1);

    bool separatePrivateMem = true;
    bool separateCQMem = true;
    const TTAMachine::Machine::AddressSpaceNavigator& nav =
      mach->addressSpaceNavigator();
    for (int i = 0; i < nav.count(); i++){
      TTAMachine::AddressSpace *as = nav.item(i);
      if (as->hasNumericalId(TTA_ASID_GLOBAL)) {
        if (as->hasNumericalId(TTA_ASID_LOCAL)) {
          separateCQMem = false;
        }
        if (as->hasNumericalId(TTA_ASID_PRIVATE)) {
          separatePrivateMem = false;
        }
      }
    }
    int AQL_queue_length = d->Dev->CQMemory->Size / AQL_PACKET_LENGTH - 1;
    unsigned dmem_size = d->Dev->DataMemory->Size;
    unsigned cq_size = d->Dev->CQMemory->Size;

    bool relativeAddressing = d->Dev->RelativeAddressing;
    int i=0;
    char extraParams[POCL_FILENAME_LENGTH*8];
    i = snprintf(extraParams, (POCL_FILENAME_LENGTH * 8),
    "-DQUEUE_LENGTH=%i ", AQL_queue_length);
    if(!separatePrivateMem) {
      int fallback_mem_size = pocl_get_int_option(
          "POCL_ACCEL_PRIVATE_MEM_SIZE", ACCEL_DEFAULT_PRIVATE_MEM_SIZE);
      unsigned initsp = dmem_size + fallback_mem_size;
      unsigned private_mem_start = dmem_size;
      if(!separateCQMem) {
        initsp += cq_size;
        private_mem_start += cq_size;
      }
      if(!relativeAddressing) {
        initsp += d->Dev->DataMemory->PhysAddress;
        private_mem_start += d->Dev->DataMemory->PhysAddress;
      }
      i += snprintf(extraParams+i, (POCL_FILENAME_LENGTH * 8),
      "--init-sp=%u --data-start=%u ", initsp, private_mem_start);
    }
    if(!separateCQMem){
      unsigned queue_start = d->Dev->CQMemory->PhysAddress;
      if (relativeAddressing) {
        queue_start -= d->Dev->DataMemory->PhysAddress;
      }
      i += snprintf(extraParams+i, (POCL_FILENAME_LENGTH * 8),
      "-DQUEUE_START=%u ", queue_start);
    }

    char commandLine[MAX_CMDLINE_LEN];
    tceccCommandLine(commandLine, MAX_CMDLINE_LEN, &cmd->command.run, tempDir,
                     inputBytecode, // inputSrc
                     assemblyFileName, bd->machine_file, bd->core_count > 1,
                     device->endian_little, extraParams);

    POCL_MSG_PRINT_INFO("build command: \n%s", commandLine);

    error = system(commandLine);
    if (error != 0)
      POCL_ABORT("Error while running tcecc.\n");
  }

    TTAProgram::Program* prog = NULL;
    try{
      prog = TTAProgram::Program::loadFromTPEF(assemblyFileName, *mach);
    } catch (Exception &e) {
      POCL_MSG_WARN("Error: %s\n",e.errorMessage().c_str());
      POCL_ABORT("Couldn't open tpef %s after compilation\n", assemblyFileName);
    }

    char wg_func_name[4 * POCL_FILENAME_LENGTH];
    snprintf(wg_func_name, sizeof(wg_func_name), "%s_workgroup_argbuffer",
             cmd->command.run.kernel->name);
    if (prog->hasProcedure(wg_func_name)) {
      const TTAProgram::Procedure &proc = prog->procedure(wg_func_name);
      int kernel_address = proc.startAddress().location();

      char md_path[POCL_FILENAME_LENGTH];
      snprintf(md_path, POCL_FILENAME_LENGTH, "%s/kernel_address.txt",
               cachedir);
      char content[64];
      snprintf(content, 64, "kernel address = %d", kernel_address);
      pocl_write_file(md_path, content, strlen(content), 0, 0);
    } else {
      POCL_ABORT("Couldn't find wg_function procedure %s from the program\n",
                 wg_func_name);
    }

  char imem_file[POCL_FILENAME_LENGTH];
  snprintf(imem_file, POCL_FILENAME_LENGTH, "%s%s", cachedir, "/parallel.img");

  if (!pocl_exists(imem_file)) {
    char genbits_command[POCL_FILENAME_LENGTH * 8];
    // --dmemwidthinmaus 4
    snprintf(genbits_command, (POCL_FILENAME_LENGTH * 8),
             "SAVEDIR=$PWD; cd %s; generatebits --dmemwidthinmaus 4 "
             "--piformat=bin2n --diformat=bin2n --program "
             "parallel.tpef %s ; cd $SAVEDIR",
             cachedir, bd->machine_file);
    POCL_MSG_PRINT_INFO("running genbits: \n %s \n", genbits_command);
    error = system(genbits_command);
    if (error != 0)
      POCL_ABORT("Error while running generatebits.\n");
  }

  // with TCEMC, "data" is empty, "param" contains the kernel_cmd struct
  char data_img[POCL_FILENAME_LENGTH];
  snprintf(data_img, POCL_FILENAME_LENGTH, "%s%s", cachedir,
           "/parallel_local.img");
  char param_img[POCL_FILENAME_LENGTH];
  snprintf(param_img, POCL_FILENAME_LENGTH, "%s%s", cachedir,
           "/parallel_param.img");

  error = pocl_exists(imem_file);
  assert(error != 0 && "parallel.img does not exist!");
/*
  error = pocl_exists(data_img);
  assert(error != 0 && "parallel_local.img does not exist!");
*/
 /* error = pocl_exists(param_img);
  assert(error != 0 && "parallel_param.img does not exist!");
*/

  delete mach;
  delete prog;

  POCL_UNLOCK(bd->tce_compile_lock);
}

/* This is a version number that is supposed to increase when there is
 * a change in TCE or ALMAIF drivers that makes previous pocl-binaries
 * incompatible (e.g. a change in generated device image file names, etc) */
#define POCL_TCE_ALMAIF_BINARY_VERSION "2"

int pocl_almaif_tce_device_hash(const char *adf_file, const char *llvm_triplet,
                                char *output) {

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

char *pocl_tce_init_build(void *data) {
  AccelData *D = (AccelData *)data;
  tce_backend_data_t *bd =
      (tce_backend_data_t *)D->compilationData->backend_data;
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
    char tempfile[POCL_FILENAME_LENGTH];
    pocl_mk_tempname(tempfile, mach_tmpdir.c_str(), ".devext", NULL);

    std::string tceopgenCmd = std::string("tceopgen > ") + tempfile;

    POCL_MSG_PRINT_TCE("Running: %s \n", tceopgenCmd.c_str());

    error = system(tceopgenCmd.c_str());
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
