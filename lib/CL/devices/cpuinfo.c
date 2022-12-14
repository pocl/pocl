/* cpuinfo.h - parsing of /proc/cpuinfo for OpenCL device info

   Copyright (c) 2012 Pekka Jääskeläinen
                 2013 Kalle Raiskila
   
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include <pocl_cl.h>
#include <string.h>

#ifndef _WIN32
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

#include "config.h"
#include "cpuinfo.h"

static const char* cpuinfo = "/proc/cpuinfo";
#define MAX_CPUINFO_SIZE 64*1024
//#define DEBUG_POCL_CPUINFO

//Linux' cpufrec interface
static const char* cpufreq_file="/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq";

// Vendor of PCI root bus
static const char *pci_bus_root_vendor_file = "/sys/bus/pci/devices/0000:00:00.0/vendor";

/* Strings to parse in /proc/cpuinfo. Else branch is for x86, x86_64 */
#if   defined  __powerpc__
 #define FREQSTRING "clock"
 #define MODELSTRING "cpu\t"
 #define DEFAULTVENDOR "AIM" // Apple-IBM-Motorola
 #define DEFAULTVENDORID 0x1014 // IBM
 #define VENDORSTRING "vendor"
#elif defined __arm__ || __aarch64__
 #define FREQSTRING " "
 #define MODELSTRING "CPU part"
 #define DEFAULTVENDOR "ARM"
 #define DEFAULTVENDORID 0x13b5 // ARM
 #define VENDORSTRING "CPU implementer"
#elif defined __mips
 #define FREQSTRING "BogoMIPS\t"
 #define MODELSTRING "cpu model\t"
 #define DEFAULTVENDORID 0x153f // MIPS
 #define DEFAULTVENDOR "MIPS"
#else
 #define FREQSTRING "cpu MHz"
 #define MODELSTRING "model name"
 #define DEFAULTVENDOR "Unknown x86"
 #define DEFAULTVENDORID 0x0
 #define VENDORSTRING "vendor_id"
#endif

static const char *cpuvendor_default = DEFAULTVENDOR;

/** 
 * Read the CPU maximum frequency from the Linux cpufreq interface.
 * If cpufreq is not available on current host processor, returns -1.
 * 
 * @return CPU maximum frequency in MHz, or -1 if cpufreq is not avaialble.
 */
static int
pocl_cpufreq_get_max()
{
  int retval=-1;
  if (access (cpufreq_file, R_OK) != 0)
    return -1;

  FILE *f = fopen (cpufreq_file, "r");
  int nread = fscanf(f, "%d", &retval);
  fclose(f);
  if (nread < 1)
    return -1;

  // KHz to MHz
  retval /= 1000; 
#ifdef DEBUG_POCL_CPUINFO
  printf("CPU max frequency (from cpufreq): %d\n", retval);
#endif
  return retval;
}

/**
 * Detects the maximum clock frequency of the CPU by parsing the cpuinfo.
 *
 * Assumes all cores have the same max clock freq. On some platforms, 
 * /proc/cpuinfo does not provide max CPU frequecny (ARM pandaboard), 
 * others give the *current* frequency, not the max (x86_64).
 *
 * @return The clock frequency in MHz, or -1 if couldn't figure it out.
 */
static int
pocl_cpuinfo_detect_max_clock_frequency()
{
  int cpufreq=-1;
  
  // First try to get the result from cpufreq interface.
  cpufreq = pocl_cpufreq_get_max();
  if( cpufreq != -1 )
    return cpufreq;

  if (access (cpuinfo, R_OK) != 0) 
      return -1;
  else 
    {
      FILE *f = fopen (cpuinfo, "r");
      char contents[MAX_CPUINFO_SIZE];
      int num_read = fread (contents, 1, MAX_CPUINFO_SIZE - 1, f);            
      float freq = 0.0f;
      fclose (f);
      contents[num_read] = '\0';

      /* Count the number of times 'processor' keyword is found which
         should give the number of cores overall in a multiprocessor
         system. In Meego Harmattan on ARM it prints Processor instead of
         processor */
      char* p = contents;
      if ((p = strstr (p, FREQSTRING)) != NULL &&
          (p = strstr (p, ": ")) != NULL)
        {
          if (sscanf (p, ": %f", &freq) == 0)
            {
#ifdef DEBUG_POCL_CPUINFO
              printf ("could not parse the cpu MHz field %f\n", freq);
              puts (p);
#endif
              return -1;
            }           
          else 
            {
#ifdef DEBUG_POCL_CPUINFO
              printf ("max_freq %d\n", (int)freq);
#endif
              return (int)freq;
            }
        }
    } 
  return -1;  
}


/**
 * Detects the number of parallel hardware threads supported by
 * the CPU by parsing the cpuinfo.
 *
 * @return The number of hardware threads, or -1 if couldn't figure it out.
 */
static int
pocl_cpuinfo_detect_compute_unit_count()
{
  if (access (cpuinfo, R_OK) != 0) 
      return -1;
  else 
    {
      FILE *f = fopen (cpuinfo, "r");
      char contents[MAX_CPUINFO_SIZE];
      int num_read = fread (contents, 1, MAX_CPUINFO_SIZE - 1, f);            
      int cores = 0;
      fclose (f);
      contents[num_read] = '\0';

      /* Count the number of times 'processor' keyword is found which
         should give the number of cores overall in a multiprocessor
         system. In Meego Harmattan on ARM it prints Processor instead of
         processor */
      char* p = contents;
      while ((p = strstr (p, "rocessor")) != NULL)
        {
          cores++;
          /* Skip to the end of the line. Otherwise causes two cores
             to be detected in case of, for example:
             Processor       : ARMv7 Processor rev 2 (v7l) */
          char* eol = strstr (p, "\n");
          if (eol != NULL)
              p = eol;
          ++p;
        }     
#ifdef DEBUG_POCL_CPUINFO
      printf("total cores %d\n", cores);
#endif
      if (cores == 0)
        return -1;

      int cores_per_cpu = 1;
      p = contents;
      if ((p = strstr (p, "cpu cores")) != NULL)
        {
          if (sscanf (p, ": %d\n", &cores_per_cpu) != 1)
            cores_per_cpu = 1;
#ifdef DEBUG_POCL_CPUINFO
          printf ("cores per cpu %d\n", cores_per_cpu);
#endif
        }

      int siblings = 1;
      p = contents;
      if ((p = strstr (p, "siblings")) != NULL)
        {
          if (sscanf (p, ": %d\n", &siblings) != 1)
            siblings = cores_per_cpu;
#ifdef DEBUG_POCL_CPUINFO
          printf ("siblings %d\n", siblings);
#endif
        }
      if (siblings > cores_per_cpu) {
#ifdef DEBUG_POCL_CPUINFO
        printf ("max threads %d\n", cores*(siblings/cores_per_cpu));
#endif
        return cores*(siblings/cores_per_cpu); /* hardware threading is on */
      } else {
#ifdef DEBUG_POCL_CPUINFO
        printf ("max threads %d\n", cores);
#endif
        return cores; /* only multicore, if not unicore*/
      }      
    } 
  return -1;  
}

#if __arm__ || __aarch64__
enum
{
  JEP106_ARM    = 0x41,
  JEP106_BRDCOM = 0x42,
  JEP106_CAVIUM = 0x43,
  JEP106_APM    = 0x50,
  JEP106_QCOM   = 0x51
};

static const struct
{
  /* JEDEC JEP106 code; /proc/cpuinfo, field "CPU implementer" */
  unsigned id;
  /* PCI vendor ID, to fill the device->vendor_id field */
  unsigned pci_vendor_id;
  char const *name;
}
vendor_list[] =
{
  { JEP106_ARM,    0x13b5, "ARM" },
  { JEP106_BRDCOM, 0x14e4, "Broadcom" },
  { JEP106_CAVIUM, 0x177d, "Cavium" },
  { JEP106_APM,    0x10e8, "Applied Micro" },
  { JEP106_QCOM,   0x5143, "Qualcomm" }
};

typedef struct
{
  unsigned id; /* part code; /proc/cpuinfo, field "CPU part" */
  char const *name;
} part_tuple_t;

static const part_tuple_t part_list_arm[] =
{
  { 0xd0a, "cortex-a75" },
  { 0xd09, "cortex-a73" },
  { 0xd08, "cortex-a72" },
  { 0xd07, "cortex-a57" },
  { 0xd05, "cortex-a55" },
  { 0xd04, "cortex-a35" },
  { 0xd03, "cortex-a53" },
  { 0xd01, "cortex-a32" },
  { 0xc0f, "cortex-a15" },
  { 0xc0e, "cortex-a17" },
  { 0xc0d, "cortex-a12" }, /* Rockchip RK3288 */
  { 0xc0c, "cortex-a12" },
  { 0xc09, "cortex-a9" },
  { 0xc08, "cortex-a8" },
  { 0xc07, "cortex-a7" },
  { 0xc05, "cortex-a5" }
};

static const part_tuple_t part_list_apm[] =
{
  { 0x0, "x-gene-1" }
};
#endif

static void
pocl_cpuinfo_get_cpu_name_and_vendor(cl_device_id device)
{
  /* If something fails here, have this as backup solution.
   * short_name is in the .data anyways.*/
  device->long_name = device->short_name;

  /* default vendor and vendor_id, in case it cannot be found by other means */
  device->vendor = cpuvendor_default;
  if (device->vendor_id == 0)
    device->vendor_id = DEFAULTVENDORID;

  /* read contents of /proc/cpuinfo */
  if (access (cpuinfo, R_OK) != 0)
    return;

  FILE *f = fopen (cpuinfo, "r");
  char contents[MAX_CPUINFO_SIZE];
  int num_read = fread (contents, 1, MAX_CPUINFO_SIZE - 1, f);            
  fclose(f);
  contents[num_read]='\0';

  char const *start, *end;
  /* find the vendor_id string an put */

#if __arm__ || __aarch64__
  unsigned vendor_id = -1;
  size_t i;
#endif
#ifdef VENDORSTRING
  do {
    start = strstr(contents, VENDORSTRING"\t: ");
    if (!start)
      break;
    start += strlen(VENDORSTRING"\t: ");
    end = strchr(start, '\n');
    if (!end)
      break;

#if __arm__ || __aarch64__
    if (1 == sscanf (start, "%x", &vendor_id))
      {
        for (i = 0; i < sizeof (vendor_list) / sizeof (vendor_list[0]); ++i)
          {
            if (vendor_id == vendor_list[i].id)
              {
                device->vendor_id = vendor_list[i].pci_vendor_id;
                start = vendor_list[i].name;
                end = start + strlen (vendor_list[i].name);
                break;
              }
          }
      }
#endif

    char *_vendor = (char *)malloc (end - start + 1);
    if (!_vendor)
      break;
    memcpy(_vendor, start, end-start);
    _vendor[end-start] = '\0';
    device->vendor = _vendor;
  } while (0);
#endif

  /* find the cpu description */
  start=strstr (contents, MODELSTRING"\t: ");
  if (start == NULL)
    return;
  start += strlen (MODELSTRING"\t: ");
  end = strchr (start, '\n');
  if (end == NULL)
    return;

#if __arm__ || __aarch64__
  unsigned part_id;
  if (1 == sscanf (start, "%x", &part_id))
    {
      part_tuple_t const *part_list = NULL;
      size_t part_count = 0;

      switch (vendor_id)
      {
        case JEP106_ARM:
          part_list = part_list_arm;
          part_count = sizeof (part_list_arm) / sizeof (part_list_arm[0]);
          break;
        case JEP106_APM:
          part_list = part_list_apm;
          part_count = sizeof (part_list_apm) / sizeof (part_list_apm[0]);
          break;
      }

      for (i = 0; i < part_count; ++i)
        {
          if (part_id == part_list[i].id)
            {
              start = part_list[i].name;
              end = start + strlen (part_list[i].name);
              break;
            }
        }
    }
#endif

  /* create the descriptive long_name for device */
  int len = strlen (device->short_name) + 1
            + (device->llvm_cpu ? strlen (device->llvm_cpu) : 0) + 1
            + (end - start) + 1;
  char *new_name = (char*)malloc (len);
  snprintf (new_name, len, "%s-%s-%s", device->short_name,
            (device->llvm_cpu ? device->llvm_cpu : ""), start);
  device->long_name = new_name;

  /* If the vendor_id field is still empty, we should get the PCI ID associated
   * with the CPU vendor (if there is one), to be ready for the (currently
   * provisional) OpenCL 3.0 specification that has finally clarified the
   * meaning of this field. To do this, we look at the vendor advertised by the
   * PCI root device. At least for Intel and AMD, this does indeed gives us the
   * expected value. (The alternative would be a look-up table for the vendor
   * string to the associated PCI ID.)
   */
  if (!device->vendor_id)
    {
      f = fopen (pci_bus_root_vendor_file, "r");
      if (f)
        {
          /* no error checking, if it failed we just won't have the info */
          num_read = fscanf (f, "%x", &device->vendor_id);
          fclose (f);
        }
    }
}

/*
 *
 * Sets up:
 *   vendor_id
 *   vendor (name)
 *   short name
 *   long name
 *
 *   max compute units IF NOT SET ALREADY
 *   max clock freq
 */

void
pocl_cpuinfo_detect_device_info(cl_device_id device) 
{
  int res;

  device->short_name = device->ops->device_name;

  if (device->max_compute_units == 0) {
    res = pocl_cpuinfo_detect_compute_unit_count();
    device->max_compute_units = (res > 0) ? (cl_uint)res : 0;
  }

  res = pocl_cpuinfo_detect_max_clock_frequency();
  device->max_clock_frequency = (res > 0) ? (cl_uint)res : 0;

  pocl_cpuinfo_get_cpu_name_and_vendor(device);
}
