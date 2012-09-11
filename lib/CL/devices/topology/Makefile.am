noinst_LTLIBRARIES = libpocl-devices-topology.la

libpocl_devices_topology_la_SOURCES = pocl_topology.c pocl_topology.h

libpocl_devices_topology_la_CPPFLAGS = -I$(top_srcdir)/fix-include -I$(top_srcdir)/include -I$(top_srcdir)/lib/CL/devices -I$(top_srcdir)/lib/CL $(HWLOC_CFLAGS) $(OCL_ICD_CFLAGS)
libpocl_devices_topology_la_LDFLAGS = @PTHREAD_CFLAGS@
libpocl_devices_topology_la_LIBADD = -lltdl $(HWLOC_LIBS)
