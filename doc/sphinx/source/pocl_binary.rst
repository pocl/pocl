Binary inputs format
--------------------

This section describe how to use ``clCreateProgramWithBinary`` with the POCL
implementation of OpenCL.

.. _khronos : https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clCreateProgramWithBinary.html

As mentionned in the khronos_ documentation, the parameter ``binaries`` of  
``clCreateProgramWithBinary`` can consist of either or both of device-specific
code and/or implementation-specific intermediate representation.

In POCL, both representations can be used.

Implementation-specific intermediate representation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

POCL implementation of ``clCreateProgramWithBinary`` enables the user to create
a program from a LLVM IR binary. This file needs to be one instance of the 
kernel, compiled as a OpenCL-C file (``-x cl``). POCL provides no tool to 
generate such a file.

Device-specific representation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

POCL implementation also enable the user to create a program from a
device-specific binary file. The benefit of this method is that the run of your
opencl code will not make any call to the compiler. 

To generate this file, use the POCL tool ``poclcc``. 

.. code-block:: c

 ./poclcc -o my_kernel.pocl my_kernel.cl

This tools gets the binary from the OpenCL function ``clGetProgramInfo``, and 
then dump it without any modifications.

``poclcc`` can take several parameters to choose for which device you what to
compile your kernel, and to specify some specific build options. If you want 
to see the complete list, run ``./poclcc -h``

When ``poclcc`` generates a binary file, it has not enough information to 
generate a code as optimized as it would have been if it has been created from 
source, build and enqueued in the same OpenCL code.

Here is an example on how to create a program from a file:

.. code-block:: c

 int main(void)
 {
   char *binary;
   size_t binary_size;
   FILE *f;

   f = fopen("youfile", "r"); // LLVM IR file or POCLCC binary file

   fseek(f, 0, SEEK_END);
   binary_size = ftell(f);
   rewind(f);

   binary = malloc(binary_size);
   fread(buffer, 1, binary_size, f);
   
   fclose(f);

   cl_platform_id platform_id;
   cl_device_id device_id;
   cl_context context;
   cl_program program;

   clGetPlatformIDs(1, &platform_id, NULL);
   clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);
   context = clCreateContext(0, 1, &device_id, NULL, NULL, NULL);

   program = clCreateProgramWithBinary(context, 1, &device_id, &binary_size, &binary, NULL, NULL);

   return 0;
 }
