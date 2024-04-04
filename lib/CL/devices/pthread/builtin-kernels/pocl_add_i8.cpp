#include <CL/cl.h>
#include <cstdio>
#include <pocl_cl.h>
#include <pocl_debug.h>
#include <pocl_types.h>

#ifdef __cplusplus
extern "C" {
#endif

namespace {
bool INITIALIZED = false;
}

void init_pocl_add_i8() {
  POCL_MSG_PRINT_INFO("Initializing\n");
  INITIALIZED = true;
}

void free_pocl_add_i8() {
  POCL_MSG_PRINT_INFO("De-initializing\n");
  INITIALIZED = false;
}

// TODO: Hard-code wg size 1 to the cache item
// TODO: Add macro for adding the surrounding _pocl_kernel_..._workgroup
void _pocl_kernel_pocl_add_i8_workgroup(cl_uchar *args, cl_uchar *context,
                                        cl_ulong group_x, cl_ulong group_y,
                                        cl_ulong group_z) {
  if (INITIALIZED) {
    POCL_MSG_PRINT_INFO("Initialized\n");

    void **arguments = *(void ***)(args);
    char *in1_data = (char *)(arguments[0]);
    char *in2_data = (char *)(arguments[1]);
    char *out_data = (char *)(arguments[2]);

    // TODO: Do not have hard-coded upper bound. This requires supporting
    // wg sizes of cache item != 1 which I'm not sure is possible because
    // they're created statically.
    for (int i = 0; i < 8; ++i) {
      out_data[i] = in1_data[i] + in2_data[i];
    }
  } else {
    POCL_MSG_ERR("ERROR: Vector addition not initalized!\n");
  }
}

#ifdef __cplusplus
}
#endif
