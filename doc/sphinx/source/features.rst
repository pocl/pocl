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

