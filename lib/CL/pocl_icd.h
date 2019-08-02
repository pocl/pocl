#include "config.h"

/* Installable Client Driver-realated things. */
#ifndef POCL_ICD_H
#define POCL_ICD_H

// stub out ICD related stuff 
#ifndef BUILD_ICD

#  define POCL_DEVICE_ICD_DISPATCH
#  define POCL_INIT_ICD_OBJECT(__obj__)

// rest of the file: ICD is enabled 
#else

extern struct _cl_icd_dispatch pocl_dispatch;  //from clGetPlatformIDs.c

#  define POCL_DEVICE_ICD_DISPATCH &pocl_dispatch,
#  define POCL_INIT_ICD_OBJECT(__obj__) (__obj__)->dispatch=&pocl_dispatch

/* Define the ICD dispatch structure that gets filled below. 
 * Prefer to get it from ocl-icd, as that has compile time type checking
 * of the function signatures. This checks that they are in correct order.
 */
#ifdef HAVE_OCL_ICD 
#include <ocl_icd.h>
#else
#define OCL_ICD_IDENTIFIED_FUNCTIONS 116
struct _cl_icd_dispatch {
        void *funcptr[166];
};
#endif

#endif
#endif

