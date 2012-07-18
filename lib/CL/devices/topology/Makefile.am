noinst_LTLIBRARIES = libpocl-devices-topology.la

libpocl_devices_topology_la_SOURCES = pocl_topology.c pocl_topology.h
libpocl_devices_topology_la_CFLAGS = $(HWLOC_EMBEDDED_CFLAGS)
libpocl_devices_topology_la_CPPFLAGS = $(HWLOC_EMBEDDED_CPPFLAGS)  -I$(top_srcdir)/include -I$(top_srcdir)/lib/CL/devices -I$(top_srcdir)/lib/CL
libpocl_devices_topology_la_LDFLAGS = -lltdl @PTHREAD_CFLAGS@ --version-info ${LIB_VERSION}
