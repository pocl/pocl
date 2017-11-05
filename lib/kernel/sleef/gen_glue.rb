# OpenCL built-in library: connect SLEEF OpenCL code with SLEEF intrinsics code
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

TYPEBITSIZE = {"half" => 16, "float" => 32, "double" => 64 }

DUAL_PREC = ["_u10", "_u35"]
DUAL_PREC_05 = ["_u05", "_u35"]

FUNCS = {
  sin: {name: 'sin', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: DUAL_PREC },
  cos: {name: 'cos', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: DUAL_PREC },
  tan: {name: 'tan', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: DUAL_PREC },

  asin: {name: 'asin', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: DUAL_PREC },
  acos: {name: 'acos', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: DUAL_PREC },
  atan: {name: 'atan', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: DUAL_PREC },
  atan2: {name: 'atan2', args: ["x", "y"], argtypes: [:fvec, :fvec], ret: :fvec, prec: DUAL_PREC },

  cbrt: {name: 'cbrt', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: DUAL_PREC },
  log: {name: 'log', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: DUAL_PREC },

  exp: {name: 'exp', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["_u10"] },
  pow: {name: 'pow', args: ["x", "y"], argtypes: [:fvec, :fvec], ret: :fvec, prec: ["_u10"] },
  pown: {name: 'pown', args: ["x", "y"], argtypes: [:fvec, :ivec], ret: :fvec, prec: ["_u10"] },
  powr: {name: 'powr', args: ["x", "y"], argtypes: [:fvec, :fvec], ret: :fvec, prec: ["_u10"] },

  sinh: {name: 'sinh', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["_u10"] },
  cosh: {name: 'cosh', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["_u10"] },
  tanh: {name: 'tanh', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["_u10"] },

  asinh: {name: 'asinh', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["_u10"] },
  acosh: {name: 'acosh', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["_u10"] },
  atanh: {name: 'atanh', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["_u10"] },

  exp2: {name: 'exp2', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["_u10"] },
  exp10: {name: 'exp10', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["_u10"] },
  expm1: {name: 'expm1', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["_u10"] },

  log10: {name: 'log10', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["_u10"] },
  log1p: {name: 'log1p', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["_u10"] },

  sinpi: {name: 'sinpi', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["_u05"] },
  cospi: {name: 'cospi', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["_u05"] },

  fma: {name: 'fma', args: ["x", "y", "z"], argtypes: [:fvec, :fvec, :fvec], ret: :fvec },
  sqrt: {name: 'sqrt', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["_u05"] },

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

  lgamma: {name: 'lgamma', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["_u10"] },
  lgamma_r: {name: 'lgamma_r', args: ["x"], argtypes: [:fvec], ret: :struct, prec: ["_u10"] },
  tgamma: {name: 'tgamma', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["_u10"] },
  erf: {name: 'erf', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["_u10"] },
  erfc: {name: 'erfc', args: ["x"], argtypes: [:fvec], ret: :fvec, prec: ["_u15"] },

  frfrexp: {name: 'frfrexp', args: ["x"], argtypes: [:fvec], ret: :fvec },

  sincos: {name: 'sincos', args: ["x"], argtypes: [:fvec], ret: :struct, prec: DUAL_PREC },
  sincospi: {name: 'sincospi', args: ["x"], argtypes: [:fvec], ret: :struct, prec: DUAL_PREC_05 },
  modf: {name: 'modf', args: ["x"], argtypes: [:fvec], ret: :struct },

  ldexp: {name: 'ldexp', args: ["x", "k"], argtypes: [:fvec, :ivec], ret: :fvec },
  expfrexp: {name: 'expfrexp', args: ["x"], argtypes: [:fvec], ret: :ivec },
  ilogb: {name: 'ilogb', args: ["x"], argtypes: [:fvec], ret: :ivec },

#  select: {name: 'select', args: ["a", "b", "c"], argtypes: [:fvec, :fvec, :ivec], ret: :fvec, ismask: true },
#  nan: {name: 'nan', args: ["nancode"], argtypes: [:uvec], ret: :fvec, ismask: true },

}

SLEEF_TYPES = {
  "double2" => "d2",
  "double4" => "d4",
  "double8" => "d8",

  "float4" => "f4",
  "float8" => "f8",
  "float16" => "f16"
}

SLEEF_REG_TYPES = {
  "double2" => "reg128d",
  "double4" => "reg256d",
  "double8" => "reg512d",

  "float4" => "reg128f",
  "float8" => "reg256f",
  "float16" => "reg512f",

  "int4" => "reg128i",
  "int8" => "reg256i",
  "int16" => "reg512i",

  "long2" => "reg128i",
  "long4" => "reg256i",
  "long8" => "reg512i"
}


class GenVecFunc

  def initialize(file, func, type, vecsize)
    @outputfile = file
    @vecbits = TYPEBITSIZE[type] * vecsize

    @hash = FUNCS[func]
    raise ArgumentError, "#{func} is not a valid function" if not @hash

    @vecsize = vecsize
    @type = type
    fail if @type.nil?
    @vectype = @type + @vecsize.to_s

    if @hash[:ismask]
      @itype = (type == "double")  ? "long" : "int"
    else
      if (@type == "double") and ((@vecsize == 2) or (func == :expfrexp))
        @itype = "long"
      else
        @itype = "int"
      end
    end

    @ivectype = @itype + @vecsize.to_s
    @ivectype2 = @itype + (@vecsize * 2).to_s

    @utype = "u#{@itype}"
    @uvectype = "u#{@ivectype}"

    @structtype = "Sleef_#{@vectype}_2"

    @name = @hash[:name].to_s
    @args = @hash[:args].dup
    @ret = @hash[:ret]
    @prec = @hash[:prec] || [""]

    @ret_is_struct = (@ret == :struct)

    init_types()

    @fullargs = @argtypes.dup.zip(@args).map { |a| a.join " " }
    @fullargs.flatten!
    @args_str = @args.map { |a| a+"_in.r" }.join(", ")
    @fullargs_str = @fullargs.join(", ")

  end

  def run()

    @prec.each do |prec|
      sleef_suffix = SLEEF_TYPES[@vectype]
      callname = "Sleef_#{@name}#{sleef_suffix}#{prec}_intrin"
      if @type == "double" and @vecsize == 2 and ["ldexp", "ilogb", "pown"].include? @name
        basename = "Sleef_#{@name}#{sleef_suffix}#{prec}_long"
      else
        if @type == "double" and @name == "expfrexp"
          basename = "Sleef_#{@name}#{sleef_suffix}#{prec}_long"
        else
          basename = "Sleef_#{@name}#{sleef_suffix}#{prec}"
        end
      end

      write "\n_CL_ALWAYSINLINE #{@rettype} #{basename}(#{@fullargs_str})"
      write "{"
      @argtypes.each_with_index do |type, i|
        name = @args[i]
        r = @argregtypes[i]
        t = @argtypes[i]
        write "  union { #{t} t; #{r} r; } #{name}_in;"
        write "  #{name}_in.t = #{name};"
      end

      write "  union { #{@rettype} t; #{@retregtype} r; } ret;"
      write "  ret.r = #{callname}(#{@args_str});"
      write "  return ret.t;"

      write "}\n"
    end

  end


private

  def init_types

    freg = SLEEF_REG_TYPES[@vectype]
    ireg = SLEEF_REG_TYPES[@ivectype]

    @argtypes = @hash[:argtypes].map { |t|
      case t
        when :fvec
          @vectype
        when :ivec
          @ivectype
        when :uvec
          @uvectype
        else
          nil
      end
    }

    @argregtypes = @hash[:argtypes].map { |t|
      case t
        when :fvec
          freg
        when :ivec, :uvec
          ireg
        else
          nil
      end
    }

    if @ret
      @rettype = case @ret
          when :fvec
            @vectype
          when :ivec
            @ivectype
          when :uvec
            @uvectype
          when :struct
            @structtype
          else
            nil
        end
      if @ret_is_struct
        @retregtype = "Sleef_#{SLEEF_REG_TYPES[@vectype]}_2"
      else
        @retregtype = SLEEF_REG_TYPES[@rettype]
      end
    else
      @rettype = @vectype
      @retregtype = SLEEF_REG_TYPES[@rettype]
    end
  end

  def write(str)
    @outputfile.puts(str)
  end


end


#####################################################################

file = File.open("glue.c", "w+")

s = %Q{#include "sleef.h"\n\n}
file.puts s
s = %Q{#include "sleef_cl.h"\n\n}
file.puts s

file.puts "\n#ifdef SLEEF_VEC_128_AVAILABLE"

  FUNCS.each_key do |f|

    g = GenVecFunc.new(file, f, "float", 4)
    g.run

    file.puts "\n#ifdef SLEEF_DOUBLE_VEC_AVAILABLE"
    g = GenVecFunc.new(file, f, "double", 2)
    g.run
    file.puts "#endif\n\n"

  end

file.puts "#endif\n\n"

file.puts "\n#ifdef SLEEF_VEC_256_AVAILABLE"

  FUNCS.each_key do |f|

    g = GenVecFunc.new(file, f, "float", 8)
    g.run

    file.puts "\n#ifdef SLEEF_DOUBLE_VEC_AVAILABLE"
    g = GenVecFunc.new(file, f, "double", 4)
    g.run
    file.puts "#endif\n\n"

  end

file.puts "#endif\n\n"

file.puts "\n#ifdef SLEEF_VEC_512_AVAILABLE"

  FUNCS.each_key do |f|

    g = GenVecFunc.new(file, f, "float", 16)
    g.run

    file.puts "\n#ifdef SLEEF_DOUBLE_VEC_AVAILABLE"
    g = GenVecFunc.new(file, f, "double", 8)
    g.run
    file.puts "#endif\n\n"

  end

file.puts "#endif"
