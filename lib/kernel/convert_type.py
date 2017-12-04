#!/usr/bin/env python3

# OpenCL built-in library: type conversion functions
#
# Copyright (c) 2013 Victor Oliveira <victormatheus@gmail.com>
# Copyright (c) 2013 Jesse Towner <jessetowner@lavabit.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# This script generates the file convert_type.cl, which contains all of the
# OpenCL functions in the form:
#
# convert_<destTypen><_sat><_roundingMode>(<sourceTypen>)

types = ['char', 'uchar', 'short', 'ushort', 'int', 'uint', 'long', 'ulong',
         'half', 'float', 'double']
int_types = ['char', 'uchar', 'short', 'ushort', 'int', 'uint', 'long', 'ulong']
unsigned_types = ['uchar', 'ushort', 'uint', 'ulong']
float_types = ['float', 'double']
int64_types = ['long', 'ulong']
float16_types = ['half']
float64_types = ['double']
vector_sizes = ['', '2', '3', '4', '8', '16']
half_sizes = [('2',''), ('4','2'), ('8','4'), ('16','8')]

saturation = ['','_sat']
rounding_modes = ['_rtz','_rte','_rtp','_rtn']
float_prefix = {'float':'FLT_', 'double':'DBL_'}
float_suffix = {'float':'f', 'double':''}

bool_type = {'char'  : 'char',
             'uchar' : 'char',
             'short' : 'short',
             'ushort': 'short',
             'int'   : 'int',
             'uint'  : 'int',
             'long'  : 'long',
             'ulong' : 'long',
             'half'  : 'int',
             'float'  : 'int',
             'double' : 'long'}

unsigned_type = {'char'  : 'uchar',
                 'uchar' : 'uchar',
                 'short' : 'ushort',
                 'ushort': 'ushort',
                 'int'   : 'uint',
                 'uint'  : 'uint',
                 'long'  : 'ulong',
                 'ulong' : 'ulong'}

sizeof_type = {'char'  : 1, 'uchar'  : 1,
               'half'  : 2,
               'short' : 2, 'ushort' : 2,
               'int'   : 4, 'uint'   : 4,
               'long'  : 8, 'ulong'  : 8,
               'float' : 4, 'double' : 8}

limit_max = {'char'  : 'CHAR_MAX',
             'uchar' : 'UCHAR_MAX',
             'short' : 'SHRT_MAX',
             'ushort': 'USHRT_MAX',
             'int'   : 'INT_MAX',
             'uint'  : 'UINT_MAX',
             'long'  : 'LONG_MAX',
             'ulong' : 'ULONG_MAX'}

limit_max_float = {'char'  : '(0x1p+7f)',
             'uchar' : '(0x1p+8f)',
             'short' : '(0x1p+15f)',
             'ushort': '(0x1p+16f)',
             'int'   : '(0x1p+31f)',
             'uint'  : '(0x1p+32f)',
             'long'  : '(0x1p+63f)',
             'ulong' : '(0x1p+64f)'}

limit_min = {'char'  : 'CHAR_MIN',
             'uchar' : '0',
             'short' : 'SHRT_MIN',
             'ushort': '0',
             'int'   : 'INT_MIN',
             'uint'  : '0',
             'long'  : 'LONG_MIN',
             'ulong' : '0'}

limit_min_float = {'char'  : '(-0x1p+7f)',
             'uchar' : '0.0f',
             'short' : '(-0x1p+15f)',
             'ushort': '0.0f',
             'int'   : '(-0x1p+31f)',
             'uint'  : '0.0f',
             'long'  : '(-0x1p+63f)',
             'ulong' : '0.0f'}

fits_in_float = { 'char', 'uchar', 'short', 'ushort' }
fits_in_double = { 'char', 'uchar', 'short', 'ushort', 'int', 'uint', 'float' }

def conditional_guard(src, dst):
  int64_count = 0
  float64_count = 0
  float16_count = 0

  if src in int64_types:    
    int64_count += 1
  elif src in float64_types:
    float64_count += 1
  elif src in float16_types:
    float16_count += 1

  if dst in int64_types:
    int64_count += 1
  elif dst in float64_types:
    float64_count += 1
  elif dst in float16_types:
    float16_count += 1

  if int64_count > 0 or float64_count > 0 or float16_count > 0:
    defines = []
    if int64_count > 0:
      defines.append("defined(cl_khr_int64)")
    if float64_count > 0:
      defines.append("defined(cl_khr_fp64)")
    if float16_count > 0:
      defines.append("defined(cl_khr_fp16)")
    print("#if " + " && ".join(defines))
    return True
  return False

def fully_representable(src, dst):
  if dst == 'float':
    if src in fits_in_float:
      return True
    else:
      return False
  if dst == 'double':
    if src in fits_in_double:
      return True
    else:
      return False
  if dst == 'half':
    if src in ['char', 'uchar']:
      return True
    else:
      return False
  return False

print("""/* !!!! AUTOGENERATED FILE generated by convert_type.py !!!!!

   DON'T CHANGE THIS FILE. MAKE YOUR CHANGES TO convert_type.py AND RUN:
   $ ./generate-conversion-type-cl.sh

   OpenCL type conversion functions

   Copyright (c) 2013 Victor Oliveira <victormatheus@gmail.com>
   Copyright (c) 2013 Jesse Towner <jessetowner@lavabit.com>

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
""")

#
# Default Conversions
#
# All conversions are in accordance with the OpenCL specification,
# which cites the C99 conversion rules.
#
# Casting from floating point to integer results in conversions
# with truncation, so it should be suitable for the default convert
# functions.
#
# Conversions from integer to floating-point, and floating-point to
# floating-point through casting is done with the default rounding
# mode. While C99 allows dynamically changing the rounding mode
# during runtime, it is not a supported feature in OpenCL according
# to Section 7.1 - Rounding Modes in the OpenCL 1.2 specification.
#
# Therefore, we can assume for optimization purposes that the
# rounding mode is fixed to round-to-nearest-even. Platform target
# authors should ensure that the rounding-control registers remain
# in this state, and that this invariant holds.
#
# Also note, even though the OpenCL specification isn't entirely
# clear on this matter, we implement all rounding mode combinations
# even for integer-to-integer conversions. When such a conversion
# is used, the rounding mode is ignored.
#

def generate_default_conversion(src, dst, mode):
  close_conditional = conditional_guard(src, dst)

  # scalar conversions
  print("""_CL_ALWAYSINLINE _CL_OVERLOADABLE _CL_READNONE
{DST} convert_{DST}{M}({SRC} x)
{{
  return ({DST})x;
}}
""".format(SRC=src, DST=dst, M=mode))

  # vector conversions, done through decomposition to components
  for size, half_size in half_sizes:
    print("""_CL_ALWAYSINLINE _CL_OVERLOADABLE _CL_READNONE
{DST}{N} convert_{DST}{N}{M}({SRC}{N} x)
{{
  return ({DST}{N})(convert_{DST}{H}(x.lo), convert_{DST}{H}(x.hi));
}}
""".format(SRC=src, DST=dst, N=size, H=half_size, M=mode))

  # 3-component vector conversions
  print("""_CL_ALWAYSINLINE _CL_OVERLOADABLE _CL_READNONE
{DST}3 convert_{DST}3{M}({SRC}3 x)
{{
  return ({DST}3)(convert_{DST}2(x.s01), convert_{DST}(x.s2));
}}""".format(SRC=src, DST=dst, M=mode))

  if close_conditional:
    print("#endif")
  print()


for src in types:
    for dst in types:
      generate_default_conversion(src, dst, '')

for src in int_types:
  for dst in int_types:
    for mode in rounding_modes:
      generate_default_conversion(src, dst, mode)

#
# Saturated Conversions To Integers
#
# These functions are dependent on the unsaturated conversion functions
# generated above, and use clamp, max, min, and select to eliminate
# branching and vectorize the conversions.
#
# Again, as above, we allow all rounding modes for integer-to-integer
# conversions with saturation.
#

def generate_saturated_conversion(src, dst, size):
  # Header
  print()
  close_conditional = conditional_guard(src, dst)
  print("""_CL_ALWAYSINLINE _CL_OVERLOADABLE _CL_READNONE
{DST}{N} convert_{DST}{N}_sat({SRC}{N} x)
{{""".format(DST=dst, SRC=src, N=size))

  # FIXME: This is a work around for lack of select function with
  # signed third argument when the first two arguments are unsigned types.
  # We cast to the signed type for sign-extension, then do a bitcast to
  # the unsigned type.
  if dst in unsigned_types:
    bool_prefix = "as_{DST}{N}(convert_{BOOL}{N}".format(DST=dst, BOOL=bool_type[dst], N=size);
    bool_suffix = ")"
  else:
    bool_prefix = "convert_{BOOL}{N}".format(BOOL=bool_type[dst], N=size);
    bool_suffix = ""

  # Body
  if src == dst:

    # Conversion between same types
    print("  return x;")

  elif src in float_types:

    # Conversion from float to int
    print("""  {DST}{N} y = convert_{DST}{N}(x);
  y = select(y, ({DST}{N}){DST_MIN}, {BP}(x < ({SRC}{N}){DST_MIN_FLT}){BS});
  y = select(y, ({DST}{N}){DST_MAX}, {BP}(x >= ({SRC}{N}){DST_MAX_FLT}){BS});
  return y;""".format(SRC=src, DST=dst, N=size,
      DST_MIN=limit_min[dst], DST_MAX=limit_max[dst],
      DST_MIN_FLT=limit_min_float[dst], DST_MAX_FLT=limit_max_float[dst],
      BP=bool_prefix, BS=bool_suffix))

  else:

    # Integer to integer convesion with sizeof(src) == sizeof(dst)
    if sizeof_type[src] == sizeof_type[dst]:
      if src in unsigned_types:
        print("  x = min(x, ({SRC}){DST_MAX});".format(SRC=src, DST_MAX=limit_max[dst]))
      else:
        print("  x = max(x, ({SRC})0);".format(SRC=src))

    # Integer to integer conversion where sizeof(src) > sizeof(dst)
    elif sizeof_type[src] > sizeof_type[dst]:
      if src in unsigned_types:
        print("  x = min(x, ({SRC}){DST_MAX});".format(SRC=src, DST_MAX=limit_max[dst]))
      else:
        print("  x = clamp(x, ({SRC}){DST_MIN}, ({SRC}){DST_MAX});"
          .format(SRC=src, DST_MIN=limit_min[dst], DST_MAX=limit_max[dst]))

    # Integer to integer conversion where sizeof(src) < sizeof(dst)
    elif src not in unsigned_types and dst in unsigned_types:
        print("  x = max(x, ({SRC})0);".format(SRC=src))

    print("  return convert_{DST}{N}(x);".format(DST=dst, N=size))

  # Footer
  print("}")
  if close_conditional:
    print("#endif")


for src in types:
  for dst in int_types:
    for size in vector_sizes:
      if size != '' or (src not in float16_types and dst not in float16_types):
        generate_saturated_conversion(src, dst, size)


def generate_saturated_conversion_with_rounding(src, dst, size, mode):
  # Header
  print()
  close_conditional = conditional_guard(src, dst)

  # Body
  print("""_CL_ALWAYSINLINE _CL_OVERLOADABLE _CL_READNONE
{DST}{N} convert_{DST}{N}_sat{M}({SRC}{N} x)
{{
  return convert_{DST}{N}_sat(x);
}}
""".format(DST=dst, SRC=src, N=size, M=mode))

  # Footer
  if close_conditional:
    print("#endif")


for src in int_types:
  for dst in int_types:
    for size in vector_sizes:
      for mode in rounding_modes:
        generate_saturated_conversion_with_rounding(src, dst, size, mode)

#
# Conversions To/From Floating-Point With Rounding
#
# Note that we assume as above that casts from floating-point to
# integer are done with truncation, and that the default rounding
# mode is fixed to round-to-nearest-even, as per C99 and OpenCL
# rounding rules.
#
# These functions rely on the use of abs, ceil, fabs, floor,
# nextafter, sign, rint and the above generated conversion functions.
#
# Only conversions to integers can have saturation.
#

dfli = [('double', 'long'), ('double', 'ulong'),
                  ('float', 'long'), ('float', 'ulong'),
                  ('float', 'int'), ('float', 'uint')]

rtn_ret_constants = {
                    ('double','long'): '0x1.fffffffffffffp+62',
                    ('double','ulong'): '0x1.fffffffffffffp+63',
                    ('float','long'): '0x1.fffffep+62',
                    ('float','ulong'): '0x1.fffffep+63',
                    ('float','int'): '0x1.fffffep+30f',
                    ('float','uint'): '0x1.fffffep+31f',
                  }

rtn_thresholds = {
                  'long': '0x7fffffffffffffffL',
                  'ulong': '0xfffffffffffffffeUL',
                  'int': '0x7ffffffc',
                  'uint': '0xffffff80'
                 }

def generate_float_conversion(src, dst, size, mode, sat):
  # Header
  print()
  close_conditional = conditional_guard(src, dst)
  print("""_CL_ALWAYSINLINE _CL_OVERLOADABLE _CL_READNONE
{DST}{N} convert_{DST}{N}{S}{M}({SRC}{N} x)
{{""".format(SRC=src, DST=dst, N=size, M=mode, S=sat))

  if fully_representable(src, dst):
    print("  return convert_{DST}{N}(x);".format(DST=dst, N=size))
    print("}")
    if close_conditional:
      print("#endif")
    return
  # Perform conversion
  if dst in int_types:
    if mode == '_rte':
      print("  x = rint(x);");
    elif mode == '_rtp':
      print("  x = ceil(x);");
    elif mode == '_rtn':
      print("  x = floor(x);");
    print("  return convert_{DST}{N}{S}(x);".format(DST=dst, N=size, S=sat))
  elif mode == '_rte':
    print("  return convert_{DST}{N}(x);".format(DST=dst, N=size))
  else:
    print("  {DST}{N} r = convert_{DST}{N}(x);".format(DST=dst, N=size))
    if (dst, src) in dfli:
      print("  {SRC}{N} y = convert_{SRC}{N}_sat(r);".format(SRC=src, N=size))
    else:
      print("  {SRC}{N} y = convert_{SRC}{N}(r);".format(SRC=src, N=size))
    if mode == '_rtz':
      if src in int_types:
        print("  {USRC}{N} abs_x = abs(x);".format(USRC=unsigned_type[src], N=size))
        print("  {USRC}{N} abs_y = abs(y);".format(USRC=unsigned_type[src], N=size))
      else:
        print("  {SRC}{N} abs_x = fabs(x);".format(SRC=src, N=size))
        print("  {SRC}{N} abs_y = fabs(y);".format(SRC=src, N=size))
      print("  {DST}{N} res = select(r, nextafter(r, sign(r) * ({DST}{N})-INFINITY), convert_{BOOL}{N}(abs_y > abs_x));"
        .format(DST=dst, N=size, BOOL=bool_type[dst]))
    if mode == '_rtp':
      print("  {DST}{N} res = select(r, nextafter(r, ({DST}{N})INFINITY), convert_{BOOL}{N}(y < x));"
        .format(DST=dst, N=size, BOOL=bool_type[dst]))
    if mode == '_rtn':
      print("  {DST}{N} res = select(r, nextafter(r, ({DST}{N})-INFINITY), convert_{BOOL}{N}(y > x));"
        .format(DST=dst, N=size, BOOL=bool_type[dst]))
    if (dst, src) in dfli and mode in ['_rtn','_rtz']:
      print("  return select(res, ({DST}{N})({RETVAL}), convert_{BOOL}{N}(x >= {THRESH}));"
        .format(DST=dst, N=size, BOOL=bool_type[dst], RETVAL=rtn_ret_constants[(dst,src)], THRESH=rtn_thresholds[src]))
    else:
      print("  return res;")
  # Footer
  print("}")
  if close_conditional:
    print("#endif")


for src in float_types:
  for dst in int_types:
    for size in vector_sizes:
      for mode in rounding_modes:
        for sat in saturation:
          generate_float_conversion(src, dst, size, mode, sat)


for src in types:
  for dst in float_types:
    for size in vector_sizes:
      for mode in rounding_modes:
        generate_float_conversion(src, dst, size, mode, '')
