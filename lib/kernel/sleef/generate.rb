# OpenCL built-in library: generate OpenCL code in sleef-pocl/
#
# Copyright (c) 2017 Michal Babej / Tampere University of Technology
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

TYPEBITSIZE = {"half" => 16, "float" => 32, "double" => 64 }.freeze

DUAL_PREC = ["10", "35"].freeze
DUAL_PREC_05 = ["05", "35"].freeze

FUNCS = {
  native_sin: {name: 'sin', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: DUAL_PREC, native: true },
  native_cos: {name: 'cos', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: DUAL_PREC, native: true },
  native_tan: {name: 'tan', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: DUAL_PREC, native: true },

  asin: {name: 'asin', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: DUAL_PREC },
  acos: {name: 'acos', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: DUAL_PREC },
  atan: {name: 'atan', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: DUAL_PREC },
  atan2: {name: 'atan2', args: ["x", "y"], argtypes: [:fvec, :fvec], ret: :fvec, prec: DUAL_PREC },

  cbrt: {name: 'cbrt', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: DUAL_PREC },
  log: {name: 'log', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: DUAL_PREC },

  exp: {name: 'exp', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["10"] },
  pow: {name: 'pow', args: ["x", "y"], argtypes: [:fvec, :fvec], ret: :fvec, prec: ["10"] },
  pown: {name: 'pown', args: ["x", "y"], argtypes: [:fvec, :ivec], ret: :fvec, prec: ["10"] },
  powr: {name: 'powr', args: ["x", "y"], argtypes: [:fvec, :fvec], ret: :fvec, prec: ["10"] },

  sinh: {name: 'sinh', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["10"] },
  cosh: {name: 'cosh', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["10"] },
  tanh: {name: 'tanh', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["10"] },

  asinh: {name: 'asinh', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["10"] },
  acosh: {name: 'acosh', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["10"] },
  atanh: {name: 'atanh', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["10"] },

  exp2: {name: 'exp2', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["10"] },
  exp10: {name: 'exp10', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["10"] },
  expm1: {name: 'expm1', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["10"] },

  log10: {name: 'log10', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["10"] },
  log1p: {name: 'log1p', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["10"] },

  sinpi: {name: 'sinpi', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["05"] },
  cospi: {name: 'cospi', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["05"] },

  ldexp: {name: 'ldexp', args: ["x", "k"], argtypes: [:fvec, :ivec], ret: :fvec },

  frfrexp: {name: 'frfrexp', args: ["x"], argtypes: [:fvec], ret: :fvec },
  expfrexp: {name: 'expfrexp', args: ["x"], argtypes: [:fvec], ret: :ivec },

  ilogb: {name: 'ilogb', args: ["x"], argtypes: [:fvec], ret: :ivec },

  fma: {name: 'fma', args: ["x", "y", "z"], argtypes: [:fvec, :fvec, :fvec], ret: :fvec },
  sqrt: {name: 'sqrt', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["05"] },

  hypot: {name: 'hypot', args: ["x", "y"], argtypes: [:fvec, :fvec], ret: :fvec, prec: DUAL_PREC_05 },

  fabs: {name: 'fabs', args: ["x"], argtypes: [:fvec], ret: :fvec },
  copysign: {name: 'copysign', args: ["x", "y"], argtypes: [:fvec, :fvec], ret: :fvec },
  fmax: {name: 'fmax', args: ["x", "y"], argtypes: [:fvec, :fvec], ret: :fvec },
  fmin: {name: 'fmin', args: ["x", "y"], argtypes: [:fvec, :fvec], ret: :fvec },
  fdim: {name: 'fdim', args: ["x", "y"], argtypes: [:fvec, :fvec], ret: :fvec },

  trunc: {name: 'trunc', args: ["x"], argtypes: [:fvec], ret: :fvec },
  floor: {name: 'floor', args: ["x"], argtypes: [:fvec], ret: :fvec },

  ceil: {name: 'ceil', args: ["x"], argtypes: [:fvec], ret: :fvec },
  round: {name: 'round', args: ["x"], argtypes: [:fvec], ret: :fvec },
  rint: {name: 'rint', args: ["x"], argtypes: [:fvec], ret: :fvec },

  nextafter: {name: 'nextafter', args: ["x", "y"], argtypes: [:fvec, :fvec], ret: :fvec },
  fmod: {name: 'fmod', args: ["x", "y"], argtypes: [:fvec, :fvec], ret: :fvec },

  lgamma: {name: 'lgamma', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["10"] },
  tgamma: {name: 'tgamma', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["10"] },
  erf: {name: 'erf', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["10"] },
  erfc: {name: 'erfc', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["15"] }

}.freeze

SLEEF_SUFFIXES = {
  "double2" => "d2",
  "double4" => "d4",
  "double8" => "d8",

  "float4" => "f4",
  "float8" => "f8",
  "float16" => "f16"
}.freeze

# for functions without pointer argument

class GenVecFunc

  def initialize(f, type, vecsize)
    @outputfile = File.open(f.to_s + ".cl", "a")
    @vecsize = vecsize
    fail if @vecsize.nil?

    @type = type
    fail if @type.nil?
    @vectype = @type.dup
    @vectype += @vecsize.to_s if @vecsize > 1
    @itype = "int"
    @ivectype = @itype.dup
    @ivectype += @vecsize.to_s if @vecsize > 1

    @hash = FUNCS[f]
    raise ArgumentError, "#{f} is not a valid function" if not @hash
    @name = @hash[:name].to_s
    @args = @hash[:args].dup
    @ret = @hash[:ret]
    @prec = @hash[:prec]
    @is_native = @hash[:native]
    @native_prefix = @is_native ? "native_" : ""

    init_types()

    @fullargs = @argtypes.dup.zip(@args).map { |a| a.join " " }
    @fullargs.flatten!
    @args_str = @args.join(", ")
    @fullargs_str = @fullargs.join(", ")

  end

  def run()

    write "\n\n#ifdef cl_khr_fp64\n\n" if @type == "double"

    if @vecsize == 3

      args3to4 = @args.map {|a| a+"_3to4" }.join(", ")

      initargs4 = ""

      @args.each_with_index do |a, i|
        initargs4 << "    #{@arg3types[i]} #{a}_3to4 = (#{@arg3types[i]})(#{a}, (#{@type})0);\n"
      end

      # prototype for vec4
      argtypes3to4_str = @arg3types.join(", ")
      write "\n_CL_OVERLOADABLE\n#{@rettype_3to4} _cl_#{@native_prefix}#{@name}(#{argtypes3to4_str});\n"
    end

    # START
    write "\n_CL_OVERLOADABLE\n#{@rettype} _cl_#{@native_prefix}#{@name}(#{@fullargs_str})\n{"

    sleef_suffix = SLEEF_SUFFIXES[@vectype]
    basename = "Sleef_#{@name}#{sleef_suffix}"
    @name = @native_prefix + @name

    if sleef_suffix
      # call SLEEF function, fallback to divide-n-conquer
      vbits = TYPEBITSIZE[@type] * @vecsize
      native = maxprec(basename)
      fallback = divideNcq()

      double_vectors = (@type == "double") ?
                        "&& defined(SLEEF_DOUBLE_VEC_AVAILABLE)" : ""
      write %Q@
      #if defined(SLEEF_VEC_#{vbits}_AVAILABLE) #{double_vectors}
        #{native}
      #else
        #{fallback}
      #endif
      @
    else
      # C / divide n conquer
      if @vecsize == 1
        basename += "f" if @type == "float"
        write maxprec(basename)
      elsif @vecsize == 3
        s = %Q@
        #{initargs4}
        #{@rettype_3to4} r = _cl_#{@name}(#{args3to4});
        return r.xyz;
        @
        write s
      else
        write divideNcq()
      end

    end

    write "}"

    write "\n\n#endif /* cl_khr_fp64 */\n\n" if @type == "double"

    @outputfile.close
  end


private

  def init_types

    @argtypes = @hash[:argtypes].map { |t|
      case t
        when :fvec
          @vectype
        when :ivec
          @ivectype
        else
          nil
      end
    }

    @arg3types = @hash[:argtypes].map { |t|
      case t
        when :fvec
          @type + "4"
        when :ivec
          @itype + "4"
        else
          nil
      end
    }

    hsize = @vecsize / 2

    if @ret
      rt = case @ret
          when :fvec
            @type
          when :ivec
            @itype
          else
            nil
        end
      @rettype = rt.dup
      @rettype += @vecsize.to_s if @vecsize > 1
      @rethalftype = rt.dup
      @rethalftype += hsize.to_s if hsize > 1
      @rettype_3to4 = rt + "4"
    else
      @rettype = @vectype
      @rethalftype = @type
      @rethalftype += hsize.to_s if hsize > 1
      @rettype_3to4 = @type + "4"
    end


  end

  def write(str)
    @outputfile.puts(str)
  end

  def maxprec(basename)
    if @is_native
      return "return #{basename}_u#{@prec[1]}(#{@args_str});"
    end
    if @prec and @prec.size > 1
      return %Q@
      #ifdef MAX_PRECISION
        return #{basename}_u#{@prec[0]}(#{@args_str});
      #else
        return #{basename}_u#{@prec[1]}(#{@args_str});
      #endif
      @
    else
      if @prec
        return "    return #{basename}_u#{@prec[0]}(#{@args_str});"
      else
        return "    return #{basename}(#{@args_str});"
      end
    end
  end

  def divideNcq()
    argslo = @args.map {|a| a+'.lo' }.join(", ")
    argshi = @args.map {|a| a+'.hi' }.join(", ")

    %Q@
    #{@rethalftype} lo = _cl_#{@name}(#{argslo});
    #{@rethalftype} hi = _cl_#{@name}(#{argshi});
    return (#{@rettype})(lo, hi);
    @
  end


  end


#####################################################################

# for functions with pointer argument

class GenVecFuncPtr

  public
  def initialize(f)
    @outputfile = File.new(f, File::CREAT|File::TRUNC|File::RDWR, 0644)
    @outputfile.puts %q{#include "sleef_cl.h"}
  end

  def run(name, type, arg2, addrspace, prec)

    double_vectors = (type == "double") ?
                      "&& defined(SLEEF_DOUBLE_VEC_AVAILABLE)" : ""

    write "\n\n#ifdef cl_khr_fp64\n\n" if type == "double"

    if ["float", "double"].include? type

    write %Q@

    _CL_OVERLOADABLE
    #{type} _cl_#{name}(#{type} x, #{addrspace} #{type}* #{arg2})
    {
        Sleef_#{type}2 temp;
        #{maxprec2(name, (type == "double" ? "" : "f"), prec)}
        *#{arg2} = temp.y;
        return temp.x;
    }

    _CL_OVERLOADABLE
    #{type}3 _cl_#{name}(#{type}3 x, #{addrspace} #{type}3* #{arg2})
    {
        #{type}4 temp;
        #{type}4 x_3to4; x_3to4.xyz = x;
        #{type}4 r = _cl_#{name}(x_3to4, &temp);
        *#{arg2} = temp.xyz;
        return r.xyz;
    }
    @

    end

    ###############################

    if type == "float"

    write %Q@

    _CL_OVERLOADABLE
    #{type}2 _cl_#{name}(#{type}2 x, #{addrspace} #{type}2* #{arg2})
    {
        #{type} plo, phi;
        #{type} lo = _cl_#{name}(x.lo, &plo);
        #{type} hi = _cl_#{name}(x.hi, &phi);

        *#{arg2} = (#{type}2)(plo, phi);
         return (#{type}2)(lo, hi);
    }
    @

    elsif type == "double"

    write %Q@

    _CL_OVERLOADABLE
    #{type}16 _cl_#{name}(#{type}16 x, #{addrspace} #{type}16* #{arg2})
    {
        #{type}8 plo, phi;
        #{type}8 lo = _cl_#{name}(x.lo, &plo);
        #{type}8 hi = _cl_#{name}(x.hi, &phi);

        *#{arg2} = (#{type}16)(plo, phi);
         return (#{type}16)(lo, hi);
    }
    @

    else
    end

    ###############################

    abits = [128, 256, 512]
    if type == "float"
      sizes = [4, 8, 16]
      suffixes = ["f4", "f8", "f16"]
      structs = ["float4", "float8", "float16"]
    elsif type == "double"
      sizes = [2, 4, 8]
      suffixes = ["d2", "d4", "d8"]
      structs = ["double2", "double4", "double8"]
    else
    end

    3.times do |i|
      size = sizes[i]
      bits = abits[i]
      suffix = suffixes[i]
      half = size / 2
      half = "" if half == 1
      struct = structs[i]

      write %Q@

    _CL_OVERLOADABLE
    #{type}#{size} _cl_#{name}(#{type}#{size} x, #{addrspace} #{type}#{size}* #{arg2})
    {
      #if defined(SLEEF_VEC_#{bits}_AVAILABLE) #{double_vectors}
        Sleef_#{struct}_2 temp;
        #{maxprec2(name, suffix, prec)}
        *#{arg2} = temp.y;
        return temp.x;
      #else

        #{type}#{half} plo, phi;
        #{type}#{half} lo = _cl_#{name}(x.lo, &plo);
        #{type}#{half} hi = _cl_#{name}(x.hi, &phi);

        *#{arg2} = (#{type}#{size})(plo, phi);
         return (#{type}#{size})(lo, hi);

      #endif

    }
    @

    end

    write "\n\n#endif /* cl_khr_fp64 */\n\n" if type == "double"


  end

  private

  def maxprec2(name, suffix, prec)
    if prec
      return %Q@
      #ifdef MAX_PRECISION
        temp = Sleef_#{name}#{suffix}_u#{prec[0]}(x);
      #else
        temp = Sleef_#{name}#{suffix}_u#{prec[1]}(x);
      #endif
      @
    else
      return "temp = Sleef_#{name}#{suffix}(x);"
    end
  end

  def write(str)
    @outputfile.write(str)
  end

end

##############################################################################

FUNCS.each_key do |f|

  s = %Q{#include "sleef_cl.h"\n\n}
  File.write(f.to_s + ".cl", s)

  [1,2,3,4,8,16].each do |vecsize|

    g = GenVecFunc.new(f, "float", vecsize)
    g.run

  end

  [1,2,3,4,8,16].each do |vecsize|

    g = GenVecFunc.new(f, "double", vecsize)
    g.run

  end

end


g1 = GenVecFuncPtr.new("sincos.cl")
s = %q{#include "sleef_cl.h"\n\n}
File.write("sincos.cl", s)

["global", "local", "private"].each do |as|
  g1.run("sincos", "float", "cosval", as, DUAL_PREC)
  g1.run("sincos", "double", "cosval", as, DUAL_PREC)
end

g2 = GenVecFuncPtr.new("modf.cl")
s = %q{#include "sleef_cl.h"\n\n}
File.write("modf.cl", s)

["global", "local", "private"].each do |as|
  g2.run("modf", "float", "iptr", as, nil)
  g2.run("modf", "double", "iptr", as, nil)
end
