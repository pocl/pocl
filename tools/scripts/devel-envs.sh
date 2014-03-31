# source this while in the pocl build dir
export POCL_BUILDING=1
export OCL_ICD_VENDORS=$PWD/ocl-vendors

#pocl test-cases don't link against pthreads, but libpocl does.
#this confuses gdb, unless we preload libpthread
#If libpocl is not built yet, this will fail...
export LD_PRELOAD=$(ldd lib/CL/.libs/libpocl.so | grep pthread | cut -f 3 -d' ')

#make sure we use the new built pocl, not some installed version.
#also, this is needed if the test binaries are run in gdb without the wrapper
#shell script automake generates
export LD_LIBRARY_PATH=$PWD/lib/CL/.libs:$PWD/lib/poclu/.libs/:$LD_LIBRARY_PATH

#sometimes useful variable when ICD fails (and we use ocl-icd)
#export OCL_ICD_DEBUG=15
