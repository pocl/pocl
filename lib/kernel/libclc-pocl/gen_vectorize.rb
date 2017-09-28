
FILES = ["acos", "asin", "atan",
         "acosh", "asinh", "atanh",
         "acospi", "asinpi", "atanpi",
         "pocl_fma",
         "ocml_helpers",
         "pown", "powr", "pow", "rootn", "pow_helpers",
         "fmod", "remainder", "remquo",
         "atan2pi",
         "expfrexp", "frfrexp", "frexp",
         "cosh", "sinh", "tanh",
         "cos", "sin", "tan", "sincos",
         "cospi", "sinpi", "tanpi",
         "log1p", "ep_log",
         "log_base", "log2", "logb",
         "sincos_helpers", "degrees", "radians"
         ]

ADDRSPACES = { "sincos" => true, "remquo" => true, "frexp" => true }

FILES.each do |name|

  f = File.open(name.to_s + ".cl", "w")

  f.puts %Q{#include "misc.h"\n\n}
  [1,2,3,4,8,16].each do |size|

    vecsize = (size > 1) ? size.to_s : ""

    f.puts "\n\n"

    f.puts %{
#ifdef HAVE_FMA32_#{size*32}
#define HAVE_FMA32 1
#else
#define HAVE_FMA32 0
#endif
}

    f.puts "#define SINGLEVEC" if (size == 1)
    f.puts "#define vtype float#{vecsize}"
    f.puts "#define v2type v2float#{vecsize}"
    f.puts "#define itype int#{vecsize}"
    f.puts "#define utype uint#{vecsize}"
    f.puts "#define inttype int#{vecsize}"

    f.puts "#define as_vtype as_float#{vecsize}"
    f.puts "#define as_itype as_int#{vecsize}"
    f.puts "#define as_utype as_uint#{vecsize}"

    f.puts "#define convert_vtype convert_float#{vecsize}"
    f.puts "#define convert_itype convert_int#{vecsize}"
    f.puts "#define convert_inttype convert_int#{vecsize}"
    f.puts "#define convert_uinttype convert_uint#{vecsize}"
    f.puts "#define convert_utype convert_uint#{vecsize}"

    f.puts %Q{\n#include "vtables.h"}
    f.puts %Q{\n#include "singlevec.h"}
    f.puts %Q{\n\n#include "sincos_helpers_fp32.h"}
    if ADDRSPACES[name]
      ["local", "global", "private"].each do |aspc|
        f.puts "#define ADDRSPACE #{aspc}"
        f.puts %Q{#include "#{name}_fp32.cl"}
        f.puts "#undef ADDRSPACE\n\n"
      end
    else
      f.puts %Q{#include "#{name}_fp32.cl"\n\n}
    end

    f.puts "#undef v2type"
    f.puts "#undef itype4"

    f.puts "#undef vtype"
    f.puts "#undef itype"
    f.puts "#undef inttype"
    f.puts "#undef utype"

    f.puts "#undef as_vtype"
    f.puts "#undef as_itype"
    f.puts "#undef as_utype"

    f.puts "#undef convert_vtype"
    f.puts "#undef convert_itype"
    f.puts "#undef convert_inttype"
    f.puts "#undef convert_uinttype"
    f.puts "#undef convert_utype"

    f.puts "#undef HAVE_FMA32"
    f.puts "#undef SINGLEVEC" if (size == 1)

  end

  f.puts "\n\n#ifdef cl_khr_fp64"
  f.puts "\n#pragma OPENCL EXTENSION cl_khr_fp64 : enable"

  [1,2,3,4,8,16].each do |size|

    vecsize = (size > 1) ? size.to_s : ""

    f.puts "\n\n"

    f.puts %{
#ifdef HAVE_FMA64_#{size*64}
#define HAVE_FMA64 1
#else
#define HAVE_FMA64 0
#endif
}

    f.puts "#define SINGLEVEC" if (size == 1)
    f.puts "#define vtype double#{vecsize}"
    f.puts "#define v2type v2double#{vecsize}"
    f.puts "#define itype long#{vecsize}"
    f.puts "#define utype ulong#{vecsize}"
    f.puts "#define uinttype uint#{vecsize}"
    f.puts "#define inttype int#{vecsize}"
    f.puts "#define utype4 v4uint#{vecsize}"
    f.puts "#define itype4 v4int#{vecsize}"

    f.puts "#define as_vtype as_double#{vecsize}"
    f.puts "#define as_itype as_long#{vecsize}"
    f.puts "#define as_utype as_ulong#{vecsize}"

    f.puts "#define convert_vtype convert_double#{vecsize}"
    f.puts "#define convert_itype convert_long#{vecsize}"
    f.puts "#define convert_inttype convert_int#{vecsize}"
    f.puts "#define convert_uinttype convert_uint#{vecsize}"
    f.puts "#define convert_utype convert_ulong#{vecsize}"


    f.puts %Q{\n#include "vtables.h"}
    f.puts %Q{\n#include "singlevec.h"}
    f.puts %Q{\n\n#include "sincos_helpers_fp64.h"}
    f.puts %Q{#include "ep_log.h"}

    if ADDRSPACES[name]
      ["local", "global", "private"].each do |aspc|
        f.puts "#define ADDRSPACE #{aspc}"
        f.puts %Q{#include "#{name}_fp64.cl"}
        f.puts "#undef ADDRSPACE\n\n"
      end
    else
      f.puts %Q{#include "#{name}_fp64.cl"\n\n}
    end

    f.puts "#undef v2type"
    f.puts "#undef itype4"
    f.puts "#undef utype4"
    f.puts "#undef uinttype"
    f.puts "#undef inttype"
    f.puts "#undef vtype"
    f.puts "#undef itype"
    f.puts "#undef utype"

    f.puts "#undef as_vtype"
    f.puts "#undef as_itype"
    f.puts "#undef as_utype"

    f.puts "#undef convert_vtype"
    f.puts "#undef convert_itype"
    f.puts "#undef convert_inttype"
    f.puts "#undef convert_uinttype"
    f.puts "#undef convert_utype"

    f.puts "#undef HAVE_FMA64"
    f.puts "#undef SINGLEVEC" if (size == 1)

  end

  f.puts "#endif"

end
