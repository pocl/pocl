#!/usr/bin/python3
#
# A script to generate functions (SPIR-mangled with SPIR AS) that will call
# target-specific kernel library functions (with OpenCL-mangled names and AS).
#
# e.g. _Z5frexpfPU3AS3i(float %x, i32 addrspace(1)* %y)
# would call
# _Z9_cl_frexpfPU7CLlocali(float %x, i32 * %1)
#
# output is LLVM IR text format
#
# Usage: python3 generate_spir_wrapper.py >POCL_DIR/lib/kernel/spir_wrapper.ll
#
# Notes:
# 1) this expects the target kernel library to have a single AS (the default);
#    it inserts addrspace casts.
# 2) almost all vector variants of OpenCL functions are ignored
# 3) some library functions are missing (geometric)
# 4) target kernel library is expected to prefix functions. This is required
#    even if the mangled names are the same, because the calling conv
#    is different for SPIR and some LLVM pass will remove the calls
#    with mismatched calling conv.

import sys

POCL_LIB_PREFIX = "_cl_"

SINGLE_ARG = [
	"acos", "acosh", "acospi",
	"asin", "asinh", "asinpi",
	"atan", "atanh", "atanpi",
	"cbrt", "ceil",
	"cos", "cosh", "cospi",
	"erfc", "erf",
	"exp", "exp2","exp10", "expm1",
	"fabs", "floor",
	"lgamma",
	"log", "log10", "log2", "log1p",
	"rint", "round", "rsqrt",
	"sin", "sinh", "sinpi",
	"sqrt",
	"tan", "tanh", "tanpi",
	"tgamma", "trunc",
	"native_cos", "native_exp", "native_exp2", "native_exp10",
	"native_log", "native_log2", "native_log10",
	"native_recip", "native_rsqrt",
	"native_sin", "native_sqrt", "native_tan",
	"degrees", "radians", "sign"
]

SINGLE_ARG_I = [
	"abs", "clz", "popcount"
]

DUAL_ARG = [
	"atan2", "atan2pi",
	"copysign",
	"fdim", "fmax", "fmin", "fmod",
	"hypot", "nextafter", "pow", "powr",
	"maxmag", "minmag", "remainder",
	"native_divide", "native_powr",
	"max", "min", "step"
]

DUAL_ARG_I = [
	"abs_diff", "add_sat", "hadd", "rhadd",
	"max", "min", "mul_hi", "rotate", "sub_sat"
]

DUAL_ARG_PTR = [
	"fract", "sincos", "modf"
]

TRIPLE_ARG = [
	"fma", "mad", "clamp", "mix", "smoothstep"
]

TRIPLE_ARG_I = [
	"clamp", "mad_hi", "mad_sat"
]

SIG_TO_LLVM_TYPE_MAP = {
	"f": "float",
	"d": "double",

	"c": "i8",
	"h": "i8",

	"s": "i16",
	"t": "i16",

	"i": "i32",
	"j": "i32",

	"l": "i64",
	"m": "i64",

	"Dv4_f": "<4 x float>",
	"Dv4_d": "<4 x double>",
}

LLVM_TYPE_EXT_MAP = {
	"f": "",
	"d": "",

	"c": " signext ",
	"h": " zeroext ",

	"s": " signext ",
	"t": " zeroext ",

	"i": "",
	"j": "",

	"l": "",
	"m": "",

	"Dv4_f": "",
	"Dv4_d": ""
}


MANGLING_AS_SPIR = {
	"global": "PU3AS1",
	"local": "PU3AS3",
	"private": "P",
	"none": ""
}

MANGLING_AS_OCL = {
	"global": "PU8CLglobal",
	"local": "PU7CLlocal",
	"private": "PU9CLprivate",
	"none": ""
}

LLVM_SPIR_AS = {
	"global": " addrspace(1)",
	"local": " addrspace(3)",
	"private": " ",
	"none": " "
}


def llvm_arg_type(argtype, AS):
	if argtype[0] == 'P':
		return SIG_TO_LLVM_TYPE_MAP[argtype[1]] + AS + "*"
	else:
		return SIG_TO_LLVM_TYPE_MAP[argtype]


def mang_suffix(argtype, AS_prefix):
	if argtype[0] == 'P':
		return AS_prefix + argtype[1]
	else:
		return argtype


def generate_function(name, arg_type, arg_type_ext, multiAS, *args):
	"""

	:param name: function name
	:param arg_type: LLVM argument type ("i32", "float" etc) of retval
	:param arg_type_ext: retval's attributes ("signext" where required etc)
	:param multiAS: True = generate for all three SPIR AddrSpaces
	:param args: function arguments as mangled type names (i,j,m,f,d etc), not LLVM types
	"""
	ocl_func_name = POCL_LIB_PREFIX + name
	spir_func_name = name

	if not multiAS:
		addr_spaces = ["none"]
	else:
		addr_spaces = ["global", "local", "private"]

	for AS in addr_spaces:

		spir_mangled_func_suffix = []
		ocl_mangled_func_suffix = []
		callee_args = []
		caller_args = []
		decl_args = []
		addrspace_casts = []

		arg_i = 0
		# LLVM IR expects instructions to be numbered from 1
		llvm_i = 1
		for cast in args:
			spir_mangled_func_suffix.append(mang_suffix(cast, MANGLING_AS_SPIR[AS]))
			ocl_mangled_func_suffix.append(mang_suffix(cast, MANGLING_AS_OCL[AS]))
			spir_mangled_type = llvm_arg_type(cast, LLVM_SPIR_AS[AS])
			ocl_mangled_type = llvm_arg_type(cast, LLVM_SPIR_AS["none"])
			caller_arg = spir_mangled_type + arg_type_ext + " %" + chr(120+arg_i)
			caller_args.append(caller_arg)
			decl_args.append(ocl_mangled_type + arg_type_ext)
			if spir_mangled_type != ocl_mangled_type:
				addrspace_casts.append("  %%%u = addrspacecast %s to %s" % (llvm_i, caller_arg, ocl_mangled_type))
				callee_args.append(ocl_mangled_type + " %" + chr(48+llvm_i))
				llvm_i += 1
			else:
				callee_args.append(ocl_mangled_type + " %" + chr(120+arg_i))
			arg_i += 1

		spir_mangled_func_suffix = "".join(spir_mangled_func_suffix)
		ocl_mangled_func_suffix = "".join(ocl_mangled_func_suffix)
		caller_args = ", ".join(caller_args)
		callee_args = ", ".join(callee_args)
		decl_args = ", ".join(decl_args)

		spir_mangled_name = "@_Z%u%s%s" % (len(spir_func_name), spir_func_name, spir_mangled_func_suffix)
		ocl_mangled_name = "@_Z%u%s%s" % (len(ocl_func_name), ocl_func_name, ocl_mangled_func_suffix)

		print("declare %s %s(%s) local_unnamed_addr #0" % (arg_type_ext + arg_type, ocl_mangled_name, decl_args))
		print("")
		print("define spir_func %s %s(%s) local_unnamed_addr #0 {" % (arg_type_ext + arg_type, spir_mangled_name, caller_args))
		if addrspace_casts:
			for cast in addrspace_casts:
				print(cast)
		print("  %%call = tail call %s %s(%s)" % (arg_type, ocl_mangled_name, callee_args))
		print("  ret %s %%call" % arg_type)
		print("}\n\n")


##############################################################

SPIR_MODULE_PREFIX = {
64:
"""
; ModuleID = 'spir_wrapper.bc'
source_filename = "generate_spir_wrapper.py"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"
""",
32:
"""
; ModuleID = 'spir_wrapper.bc'
source_filename = "generate_spir_wrapper.py"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"
"""
}


triple = SPIR_MODULE_PREFIX[64]
if (len(sys.argv) > 1) and (sys.argv[1] == "32"):
	triple = SPIR_MODULE_PREFIX[32]

print(triple)

MANG_TYPES_32 = {
	"f": "f",
	"Pf": "Pf",
	'u': "j"
}

MANG_TYPES_64 = {
	"f": "d",
	"Pf": "Pd",
	'u': "m"
}

# FP
for llvm_type in ["float", "double"]:
	if llvm_type == "float":
		MANG_TYPE_MAP = MANG_TYPES_32
	else:
		MANG_TYPE_MAP = MANG_TYPES_64

	for f in SINGLE_ARG:
		generate_function(f, llvm_type, '', False, MANG_TYPE_MAP['f'])
	for f in DUAL_ARG:
		generate_function(f, llvm_type, '', False, MANG_TYPE_MAP['f'], MANG_TYPE_MAP['f'])
	for f in DUAL_ARG_PTR:
		generate_function(f, llvm_type, '', True, MANG_TYPE_MAP['f'], MANG_TYPE_MAP['Pf'])
	for f in TRIPLE_ARG:
		generate_function(f, llvm_type, '', False, MANG_TYPE_MAP['f'], MANG_TYPE_MAP['f'], MANG_TYPE_MAP['f'])

	# other signatures
	generate_function("ilogb", "i32", '', False, MANG_TYPE_MAP['f'])

	generate_function("ldexp", llvm_type, '', False, MANG_TYPE_MAP['f'], 'i')
	generate_function("pown", llvm_type, '', False, MANG_TYPE_MAP['f'], 'i')
	generate_function("rootn", llvm_type, '', False, MANG_TYPE_MAP['f'], 'i')

	generate_function("remquo", llvm_type, '', True, MANG_TYPE_MAP['f'], MANG_TYPE_MAP['f'], 'Pi')

	generate_function("nan", llvm_type, '', False, MANG_TYPE_MAP['u'])

	generate_function("lgamma_r", llvm_type, '', True, MANG_TYPE_MAP['f'], 'Pi')
	generate_function("frexp", llvm_type, '', True, MANG_TYPE_MAP['f'], 'Pi')


# vectors
generate_function("length", "<4 x float>", '', False, "Dv4_f")
generate_function("length", "<4 x double>", '', False, "Dv4_d")

# Integer
for mang_type in ['c', 'h', 's', 't', 'i', 'j', 'l', 'm']:
	for f in SINGLE_ARG_I:
		generate_function(f, SIG_TO_LLVM_TYPE_MAP[mang_type], LLVM_TYPE_EXT_MAP[mang_type], False, mang_type)
	for f in DUAL_ARG_I:
		generate_function(f, SIG_TO_LLVM_TYPE_MAP[mang_type], LLVM_TYPE_EXT_MAP[mang_type], False, mang_type, mang_type)
	for f in TRIPLE_ARG_I:
		generate_function(f, SIG_TO_LLVM_TYPE_MAP[mang_type], LLVM_TYPE_EXT_MAP[mang_type], False, mang_type, mang_type, mang_type)

# "mul24", "mad24" only take an i32
generate_function("mul24", SIG_TO_LLVM_TYPE_MAP['i'], LLVM_TYPE_EXT_MAP['i'], False, 'i', 'i')
generate_function("mul24", SIG_TO_LLVM_TYPE_MAP['j'], LLVM_TYPE_EXT_MAP['j'], False, 'j', 'j')
generate_function("mad24", SIG_TO_LLVM_TYPE_MAP['i'], LLVM_TYPE_EXT_MAP['i'], False, 'i', 'i', 'i')
generate_function("mad24", SIG_TO_LLVM_TYPE_MAP['j'], LLVM_TYPE_EXT_MAP['j'], False, 'j', 'j', 'j')

print("""

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.ocl.version = !{!0}
!opencl.spir.version = !{!0}
!llvm.ident = !{!1}
!llvm.module.flags = !{!2, !3}

!0 = !{i32 1, i32 2}
!1 = !{!"clang version 6.0.0"}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 7, !"PIC Level", i32 2}

""")
