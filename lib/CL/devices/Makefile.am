# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for pocl/lib/devices.
# 
# Copyright (c) 2011-2012 Universidad Rey Juan Carlos and 
#                         Tampere University of Technology
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

SUBDIRS = topology pthread basic

noinst_LTLIBRARIES = libpocl-devices.la

libpocl_devices_la_SOURCES = devices.h devices.c bufalloc.c dev_image.h \
	prototypes.inc common.h common.c bufalloc.h cpuinfo.c cpuinfo.h
libpocl_devices_la_LIBADD = pthread/libpocl-devices-pthread.la \
	basic/libpocl-devices-basic.la topology/libpocl-devices-topology.la 

libpocl_devices_la_CPPFLAGS = \
  -I$(top_srcdir)/fix-include \
  -I$(top_srcdir)/include \
  -I$(top_srcdir)/lib/CL/devices \
  -I$(top_srcdir)/lib/CL/devices/tce \
  -I$(top_srcdir)/lib/CL \
  $(OCL_ICD_CFLAGS)

if TCE_AVAILABLE
SUBDIRS += tce
libpocl_devices_la_LIBADD += tce/libpocl-devices-tce.la
endif

# may be needed by some implementations
dist_pkgdata_DATA = bufalloc.h bufalloc.c

if BUILD_HSA
SUBDIRS += hsa
libpocl_devices_la_LIBADD += hsa/libpocl-devices-hsa.la
endif

EXTRA_DIST = CMakeLists.txt

