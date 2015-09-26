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

#ifndef _MSC_VER
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

#include "config.h"
#include "cpuinfo.h"

const char* cpuinfo = "/proc/cpuinfo";
#define MAX_CPUINFO_SIZE 64*1024
//#define DEBUG_POCL_CPUINFO

//Linux' cpufrec interface
const char* cpufreq_file="/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq";

/* Strings to parse in /proc/cpuinfo. Else branch is for x86, x86_64 */
#if   defined  __powerpc__
 #define FREQSTRING "clock"
 #define MODELSTRING "cpu\t"
 #define DEFAULTVENDOR "AIM" // Apple-IBM-Motorola
 #define DEFAULTVENDORID 0x1014 // IBM
 #define VENDORSTRING "vendor"
#elif defined __arm__
 #define FREQSTRING " "
 #define MODELSTRING "Processor"
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
int pocl_cpufreq_get_max()
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
int
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
int
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

#ifdef POCL_ANDROID

#define SYSFS_CPU_NUM_CORES_NODE    "/sys/devices/system/cpu/possible"

int
pocl_sysfs_detect_compute_unit_count()
{
  int cores = -1;

  FILE *fp = fopen(SYSFS_CPU_NUM_CORES_NODE, "r");

  // cpu/possible will of format
  // 0        : for single-core devices
  // 0-(n-1)  : for n-core cpus
  if (fp)
    {
      cores = fgetc(fp) - '0';
      if (!feof(fp))         // If more than 1 cores
        {
          fgetc(fp);          // Ignore '-'
          fscanf(fp, "%d", &cores);
        }
      fclose(fp);
      cores ++;             // always printed as (n-1)
    }

  return cores;
}
#endif

void
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

  char *start, *end;
  /* find the vendor_id string an put */
#ifdef VENDORSTRING
  do {
    start = strstr(contents, VENDORSTRING"\t: ");
    if (!start)
      break;
    start += strlen(VENDORSTRING"\t: ");
    end = strchr(start, '\n');
    if (!end)
      break;
    char *_vendor = malloc(end-start + 1);
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

  /* create the descriptive long_name for device */
  int len = strlen (device->short_name) + (end-start) + 2;
  char *new_name = (char*)malloc (len);
  snprintf (new_name, len, "%s-%s", device->short_name, start);
  device->long_name = new_name;

}

void
pocl_cpuinfo_detect_device_info(cl_device_id device) 
{
  int res;
#ifdef POCL_ANDROID
  /* sysfs node seems more suitable for android kernels
     override the value provided by hwloc
   */
  device->max_compute_units = pocl_sysfs_detect_compute_unit_count();
#else
  if (device->max_compute_units == 0) {
    res = pocl_cpuinfo_detect_compute_unit_count();
    device->max_compute_units = (res > 0) ? (cl_uint)res : 0;
  }
#endif

  res = pocl_cpuinfo_detect_max_clock_frequency();
  device->max_clock_frequency = (res > 0) ? (cl_uint)res : 0;

  pocl_cpuinfo_get_cpu_name_and_vendor(device);
}
