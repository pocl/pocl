// Check distro CPU selection and cache separation.
//
// Copyright (c) 2026 PoCL developers
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include "config.h"
#include "devices/common.h"
#include "pocl_llvm.h"
#include "poclu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
set_option (const char *name, const char *value)
{
#ifdef _WIN32
  _putenv_s (name, value ? value : "");
#else
  if (value)
    setenv (name, value, 1);
  else
    unsetenv (name);
#endif
}

/* Find the best host-supported candidate without a CPU override. */
static const char *
best_host_library (const char *const *candidates)
{
  const char *best = NULL;
  for (; *candidates; ++candidates)
    {
      set_option ("POCL_KERNELLIB_NAME", *candidates);
      const char *selected = pocl_get_distro_kernellib_variant ();
      if (selected && strcmp (selected, *candidates) == 0)
        best = *candidates;
    }
  set_option ("POCL_KERNELLIB_NAME", NULL);
  return best;
}

int
main (int argc, char **argv)
{
  const char *mode = argc > 1 ? argv[1] : "native";
  const char *expected_cpu = NULL;
  const char *expected_library = NULL;
  char expected_suffix[32];
  char *native_cpu = NULL;
  set_option ("POCL_LLVM_CPU_NAME", NULL);
  set_option ("POCL_KERNELLIB_NAME", NULL);
  set_option ("POCL_DEVICES", "cpu");

  if (strcmp (mode, "native") == 0)
    {
      native_cpu = pocl_get_llvm_cpu_name ();
      expected_cpu = native_cpu;
      if (strcmp (native_cpu, "generic") == 0)
        expected_cpu
          = pocl_get_distro_cpu_name (pocl_get_distro_kernellib_variant ());
    }
  else if (strcmp (mode, "library") == 0)
    {
      set_option ("POCL_KERNELLIB_NAME", "sse2");
      expected_cpu = "x86-64";
      expected_library = "-sse2";
    }
  else if (strcmp (mode, "override") == 0 || strcmp (mode, "both") == 0
           || strcmp (mode, "incompatible") == 0)
    {
      /* Enumeration executes no target code, even on hosts without SSSE3. */
      const char *candidates[] = { "sse2", "ssse3", NULL };
      const char *best = best_host_library (candidates);
      TEST_ASSERT (best != NULL);
      snprintf (expected_suffix, sizeof (expected_suffix), "-%s", best);
      set_option ("POCL_LLVM_CPU_NAME", "core2");
      expected_cpu = "core2";
      expected_library = expected_suffix;
      if (strcmp (mode, "incompatible") == 0)
        set_option ("POCL_KERNELLIB_NAME", "avx2");
      if (strcmp (mode, "both") == 0)
        {
          set_option ("POCL_KERNELLIB_NAME", "sse2");
          expected_library = "-sse2";
        }
    }
  else if (strcmp (mode, "baseline") == 0 || strcmp (mode, "znver3") == 0
           || strcmp (mode, "znver4") == 0)
    {
      const char *candidates[] = { "sse2",     "ssse3", "sse41",  "avx",
                                   "avx_f16c", "avx2",  "avx512", NULL };
      if (strcmp (mode, "baseline") == 0)
        candidates[1] = NULL;
      else if (strcmp (mode, "znver3") == 0)
        candidates[6] = NULL;
      const char *best = best_host_library (candidates);
      TEST_ASSERT (best != NULL);
      snprintf (expected_suffix, sizeof (expected_suffix), "-%s", best);
      expected_cpu = strcmp (mode, "baseline") == 0 ? "x86-64" : mode;
      expected_library = expected_suffix;
      set_option ("POCL_LLVM_CPU_NAME", expected_cpu);
    }
  else if (strcmp (mode, "unknown") == 0)
    {
      /* Test fallback without initializing an invalid target. */
      set_option ("POCL_LLVM_CPU_NAME", "not-a-real-cpu");
      TEST_ASSERT (strcmp (pocl_get_distro_kernellib_variant (), "generic") == 0);
      puts ("OK");
      return EXIT_SUCCESS;
    }
  else
    TEST_ASSERT (0 && "unknown test mode");

  cl_platform_id platform;
  cl_device_id device;
  CHECK_CL_ERROR (clGetPlatformIDs (1, &platform, NULL));
  CHECK_CL_ERROR (
    clGetDeviceIDs (platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL));
  TEST_ASSERT (expected_cpu && device->llvm_cpu);
  TEST_ASSERT (strcmp (device->llvm_cpu, expected_cpu) == 0);
  if (expected_library)
    {
      size_t len = strlen (device->kernellib_name);
      size_t suffix_len = strlen (expected_library);
      TEST_ASSERT (len >= suffix_len);
      TEST_ASSERT (
        strcmp (device->kernellib_name + len - suffix_len, expected_library)
        == 0);
    }

  /* Check cache separation without mutating the live device. */
  struct _cl_device_id copy = { 0 };
  copy.llvm_cpu = device->llvm_cpu;
  copy.kernellib_name = "kernel-test-generic";
  char *generic_hash = pocl_cpu_build_hash (&copy);
  copy.kernellib_name = "kernel-test-sse2";
  char *sse2_hash = pocl_cpu_build_hash (&copy);
  TEST_ASSERT (strcmp (generic_hash, sse2_hash) != 0);
  copy.llvm_cpu = "another-cpu";
  char *other_cpu_hash = pocl_cpu_build_hash (&copy);
  TEST_ASSERT (strcmp (sse2_hash, other_cpu_hash) != 0);
  free (generic_hash);
  free (sse2_hash);
  free (other_cpu_hash);
  free (native_cpu);
  puts ("OK");
  return EXIT_SUCCESS;
}
