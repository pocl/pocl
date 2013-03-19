# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for pocl/lib/devices.
# 
# Copyright (c) 2011 Universidad Rey Juan Carlos
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

SUBDIRS = ttasim

noinst_LTLIBRARIES = libpocl-devices-tce.la
libpocl_devices_tce_la_SOURCES = tce_common.h tce_common.cc
libpocl_devices_tce_la_LIBADD = ttasim/libpocl-devices-tce-ttasim.la
libpocl_devices_tce_la_LDFLAGS = -lltdl @PTHREAD_CFLAGS@ --version-info ${LIB_VERSION}

libpocl_devices_tce_la_CPPFLAGS = `@TCE_CONFIG@ --includes` -I$(top_srcdir)/fix-include -I$(top_srcdir)/include \
-I$(top_srcdir)/lib/CL/devices -I$(top_srcdir)/lib/CL/devices/tce \
-I$(top_srcdir)/lib/CL $(OCL_ICD_CFLAGS)

dist_pkgdata_DATA = tta_device_main.c

