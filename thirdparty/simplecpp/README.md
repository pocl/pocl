A fork from https://github.com/cppcheck-opensource/simplecpp containing only
the library files (simplecpp.h, simplecpp.cpp).

Based on upstream commit 370f18decdfa9ff75f2fda85ec3cf9072b82a77b (master,
after the 1.9.1 release; it contains an upstream fix for expanding
function-like macros whose arguments follow the expansion result).

PoCL modifications:

- preprocess(): unknown preprocessor directives and #pragmas other than
  "#pragma once" are emitted to the output unchanged instead of being
  silently dropped. PoCL needs the preprocessed source to be compilable by
  a different compiler on the remote server, so e.g.
  "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" must survive.

- the passthrough does not apply to #line: simplecpp already writes its
  own `#line <n> "<file>"` markers when stringifying the output, so
  passing an input #line through would emit the marker twice.
