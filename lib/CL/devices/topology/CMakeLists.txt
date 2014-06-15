add_compile_options(${HWLOC_CFLAGS})
add_library("pocl-devices-topology" OBJECT pocl_topology.c pocl_topology.h)
