
# TODO hwloc cflags !

set(POCL_DEVICE_SOURCES ${POCL_DEVICE_SOURCES}
  devices/topology/pocl_topology.c
  devices/topology/pocl_topology.h PARENT_SCOPE)

set_source_files_properties(devices/topology/pocl_topology.h
  devices/topology/pocl_topology.c PROPERTIES COMPILE_FLAGS "${HWLOC_CFLAGS}")
