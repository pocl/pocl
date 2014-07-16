Known unsupported OpenCL features
=================================

The known unsupported OpenCL (both 1.x and 2.x) features are
listed here as encountered.

Frontend/Clang
--------------

* cl_khr_f16: half precision float literals

  Compiling "3434.0h" fails with:
  error: invalid suffix 'h' on floating constant

  Tested with Clang 3.4 on 2014-07-10.

Overall
-------

* OpenCL 1.1: A.2 Multiple Host Threads

  Currently pocl guards only the reference counter updates
  with mutexes. Thus, it is more than likely that the full
  thread-safety is not there. We should go through the APIs 
  and add more locks to avoid race conditions.

  Checked on 2014-07-16.
