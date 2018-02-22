OpenCL host library
-------------------

The API implementations of The OpenCL Runtime and the The OpenCL Platform Layer are
compiled to a single dynamic library (e.g., ``libpocl.so``). This library contains
all implementations and, if pocl is compiled in the 
`ICD mode <http://www.khronos.org/registry/cl/extensions/khr/cl_khr_icd.txt>`_,
is what the ICD loader accesses. In case pocl is instructed (via -DENABLE_ICD=0)
to compile a "directly linkable library", ``libOpenCL.so`` is produced
which can be linked directly to the OpenCL programs (instead of linking
against the ICD loader).

