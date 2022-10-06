#!/usr/bin/python3
#
# A script to generate wrapping functions (SPIR-mangled with SPIR AS) that will wrap
# calls to target-specific kernel library functions (with OpenCL-mangled names and AS).
#
# e.g. with x86-64 CPU target,
#    _Z5frexpfPU3AS3i(float %x, i32 addrspace(1)* %y)
# would call
#    _Z9_cl_frexpfPU7CLlocali(float %x, i32 * %1)
#
# output is LLVM IR text format.
#
# Usage: python3 generate_spir_wrapper.py >spir_wrapper.ll
# ... rename & place the file in the target-specific lib/kernel subdirectory.
#
# Notes for CPU SPIR wrapper:
# 1) this expects the target kernel library to have a single AS (the default);
#    it inserts addrspace casts.
# 2) the X86-64 ABI complicates things by coercing arguments and byval passing,
#    which means we need different wrappers for different CPU groups
#    (depending on largest register size available)
# 3) target kernel library is expected to prefix functions. This is required
#    even if the mangled names are the same, because the calling conv
#    is different for SPIR and some LLVM pass will remove the calls
#    with mismatched calling conv.
#
# Notes for CUDA SPIR wrapper:
# 1) mangling is not required for CUDA
# 2) address space casting is not required for CUDA
# 3) prefixing and SPIR calling convention are still required
#
# set the boolean variables below to correct values before calling this script

import sys
import argparse

parser = argparse.ArgumentParser(description='SPIR wrapper generator.')

parser.add_argument('-d', '--driver',
                    dest='driver',
                    default="cpu", choices=["cpu", "cuda"],
                    help='driver to generate wrapper for')

parser.add_argument('-r', '--register-size',
                    dest='reg_size',
                    default=128, type=int, choices=[128,256,512],
                    help='largest register size (cpu driver only)')

parser.add_argument('output',
                    type=argparse.FileType('w', encoding='utf-8'),
                    help='output file')

args = parser.parse_args()

# function prefix used by PoCL's kernel library
POCL_LIB_PREFIX = "_cl_"

# CUDA uses the same pointer AS-mangling as SPIR
# set to False for CUDA wrapper, True for CPU wrapper
MANGLE_OCL = True if args.driver == 'cpu' else False

# if True, creates AS casts of pointer arguments (all are cast to AS 0)
# set to False for CUDA wrapper, True for CPU wrapper
AS_CASTS_REQUIRED = True if args.driver == 'cpu' else False

# if set to True, does target specific (x86-64 CPU) hacks
# like argument coercing (2xfloat -> double), byval passing
# (for args that don't fit into registers)
X86_CALLING_ABI = True if args.driver == 'cpu' else False

# size of the largest CPU (SIMD) register. Values larger than this
# will be passed with byval
X86_ABI_REG_SIZE = args.reg_size

sys.stdout = args.output

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
	"log", "log10", "log2", "log1p", "logb",
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
	"abs", "clz", "ctz", "popcount"
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
	"abs_diff", "add_sat", "hadd",
	"max", "min", "mul_hi", "rhadd", "rotate", "sub_sat"
]

DUAL_ARG_PTR = [
	"fract", "sincos", "modf"
]

TRIPLE_ARG = [
	"bitselect", "clamp", "fma", "mad", "mix", "smoothstep"
]

TRIPLE_ARG_I = [
	"bitselect", "clamp", "mad_hi", "mad_sat"
]

OLD_ATOMICS_INT_ONLY = [
	"atomic_add",
	"atomic_sub",
	"atomic_or",
	"atomic_xor",
	"atomic_and",
	"atomic_min",
	"atomic_max",
]


SVM_ATOMICS_INT_ONLY = [
	"atomic_fetch_add",
	"atomic_fetch_sub",
	"atomic_fetch_or",
	"atomic_fetch_xor",
	"atomic_fetch_and",
	"atomic_fetch_min",
	"atomic_fetch_max"
]

SVM_ATOMICS_FLAGS = [
	"atomic_flag_test_and_set",
	"atomic_flag_clear"
]

SVM_ATOMICS_ALL = {
	"atomic_store",
	"atomic_load",
	"atomic_exchange",
	"atomic_compare_exchange_strong",
	"atomic_compare_exchange_weak",
}

SIG_TO_LLVM_TYPE_MAP = {
	"f": "float",
	"d": "double",

	"b": "i1",
	"v": "void",

	"c": "i8",
	"h": "i8",

	"s": "i16",
	"t": "i16",

	"i": "i32",
	"j": "i32",

	"l": "i64",
	"m": "i64",

	"Dv2_f": "<2 x float>",
	"Dv2_d": "<2 x double>",
	"Dv2_c": "<2 x i8>",
	"Dv2_h": "<2 x i8>",
	"Dv2_s": "<2 x i16>",
	"Dv2_t": "<2 x i16>",
	"Dv2_i": "<2 x i32>",
	"Dv2_j": "<2 x i32>",
	"Dv2_l": "<2 x i64>",
	"Dv2_m": "<2 x i64>",

	"Dv3_f": "<3 x float>",
	"Dv3_d": "<3 x double>",
	"Dv3_c": "<3 x i8>",
	"Dv3_h": "<3 x i8>",
	"Dv3_s": "<3 x i16>",
	"Dv3_t": "<3 x i16>",
	"Dv3_i": "<3 x i32>",
	"Dv3_j": "<3 x i32>",
	"Dv3_l": "<3 x i64>",
	"Dv3_m": "<3 x i64>",

	"Dv4_f": "<4 x float>",
	"Dv4_d": "<4 x double>",
	"Dv4_c": "<4 x i8>",
	"Dv4_h": "<4 x i8>",
	"Dv4_s": "<4 x i16>",
	"Dv4_t": "<4 x i16>",
	"Dv4_i": "<4 x i32>",
	"Dv4_j": "<4 x i32>",
	"Dv4_l": "<4 x i64>",
	"Dv4_m": "<4 x i64>",

	"Dv8_f": "<8 x float>",
	"Dv8_d": "<8 x double>",
	"Dv8_c": "<8 x i8>",
	"Dv8_h": "<8 x i8>",
	"Dv8_s": "<8 x i16>",
	"Dv8_t": "<8 x i16>",
	"Dv8_i": "<8 x i32>",
	"Dv8_j": "<8 x i32>",
	"Dv8_l": "<8 x i64>",
	"Dv8_m": "<8 x i64>",

	"Dv16_f": "<16 x float>",
	"Dv16_d": "<16 x double>",
	"Dv16_c": "<16 x i8>",
	"Dv16_h": "<16 x i8>",
	"Dv16_s": "<16 x i16>",
	"Dv16_t": "<16 x i16>",
	"Dv16_i": "<16 x i32>",
	"Dv16_j": "<16 x i32>",
	"Dv16_l": "<16 x i64>",
	"Dv16_m": "<16 x i64>",

	"12memory_order": "i32",
	"12memory_scope": "i32",

	'14ocl_image1d_ro': '%opencl.image1d_ro_t',
	'14ocl_image2d_ro': '%opencl.image2d_ro_t',
	'14ocl_image3d_ro': '%opencl.image3d_ro_t',
	'20ocl_image1d_array_ro': '%opencl.image1d_array_ro_t',
	'20ocl_image2d_array_ro': '%opencl.image2d_array_ro_t',
	'21ocl_image1d_buffer_ro': '%opencl.image1d_buffer_ro_t',

	'14ocl_image1d_rw': '%opencl.image1d_rw_t',
	'14ocl_image2d_rw': '%opencl.image2d_rw_t',
	'14ocl_image3d_rw': '%opencl.image3d_rw_t',
	'20ocl_image1d_array_rw': '%opencl.image1d_array_rw_t',
	'20ocl_image2d_array_rw': '%opencl.image2d_array_rw_t',
	'21ocl_image1d_buffer_rw': '%opencl.image1d_buffer_rw_t',

	'14ocl_image1d_wo': '%opencl.image1d_wo_t',
	'14ocl_image2d_wo': '%opencl.image2d_wo_t',
	'14ocl_image3d_wo': '%opencl.image3d_wo_t',
	'20ocl_image1d_array_wo': '%opencl.image1d_array_wo_t',
	'20ocl_image2d_array_wo': '%opencl.image2d_array_wo_t',
	'21ocl_image1d_buffer_wo': '%opencl.image1d_buffer_wo_t',

	'11ocl_sampler': '%opencl.sampler_t'
}

COERCE_VECTOR_MAP = {
	"<2 x float>": "double",

	"<2 x i8>": "i16",
	"<2 x i16>": "i32",
	"<2 x i32>": "double",

	"<3 x i8>": "i32",
	"<3 x i16>": "double",

	"<4 x i8>": "i32",
	"<4 x i16>": "double",

	"<8 x i8>": "double",
}

if X86_ABI_REG_SIZE == 128:
	BYVAL_VECTOR_MAP = {
		"<4 x i64>":"<4 x i64>* byval(<4 x i64>) align 32",
		"<4 x double>": "<4 x double>* byval(<4 x double>) align 32",
		"<8 x i32>":"<8 x i32>* byval(<8 x i32>) align 32",
		"<8 x float>": "<8 x float>* byval(<8 x float>) align 32",
		"<16 x i16>": "<16 x i16>* byval(<16 x i16>) align 32",

		"<8 x i64>":"<8 x i64>* byval(<8 x i64>) align 64",
		"<8 x double>": "<8 x double>* byval(<8 x double>) align 64",

		"<16 x i32>": "<16 x i32>* byval(<16 x i32>) align 64",
		"<16 x float>": "<16 x float>* byval(<16 x float>) align 64",

		"<16 x i64>":"<16 x i64>* byval(<16 x i64>) align 128",
		"<16 x double>": "<16 x double>* byval(<16 x double>) align 128",
	}

elif X86_ABI_REG_SIZE == 256:
	BYVAL_VECTOR_MAP = {
		"<8 x i64>":"<8 x i64>* byval(<8 x i64>) align 64",
		"<8 x double>": "<8 x double>* byval(<8 x double>) align 64",

		"<16 x i32>": "<16 x i32>* byval(<16 x i32>) align 64",
		"<16 x float>": "<16 x float>* byval(<16 x float>) align 64",

		"<16 x i64>":"<16 x i64>* byval(<16 x i64>) align 128",
		"<16 x double>": "<16 x double>* byval(<16 x double>) align 128",
	}

elif X86_ABI_REG_SIZE == 512:
	BYVAL_VECTOR_MAP = {
		"<16 x i64>":"<16 x i64>* byval(<16 x i64>) align 128",
		"<16 x double>": "<16 x double>* byval(<16 x double>) align 128",
	}

else:
	print("Error, X86_ABI_REG_SIZE must be one of 128, 256, 512")
	exit(2)


SIG_TO_TYPE_NAME_MAP = {
	"f": "float",
	"d": "double",

	"c": "char",
	"h": "uchar",

	"s": "short",
	"t": "ushort",

	"i": "int",
	"j": "uint",

	"l": "long",
	"m": "ulong",

	"Dv2_f": "float2",
	"Dv2_d": "double2",
	"Dv2_c": "char2",
	"Dv2_h": "uchar2",
	"Dv2_s": "short2",
	"Dv2_t": "ushort2",
	"Dv2_i": "int2",
	"Dv2_j": "uint2",
	"Dv2_l": "long2",
	"Dv2_m": "ulong2",

	"Dv3_f": "float3",
	"Dv3_d": "double3",
	"Dv3_c": "char3",
	"Dv3_h": "uchar3",
	"Dv3_s": "short3",
	"Dv3_t": "ushort3",
	"Dv3_i": "int3",
	"Dv3_j": "uint3",
	"Dv3_l": "long3",
	"Dv3_m": "ulong3",

	"Dv4_f": "float4",
	"Dv4_d": "double4",
	"Dv4_c": "char4",
	"Dv4_h": "uchar4",
	"Dv4_s": "short4",
	"Dv4_t": "ushort4",
	"Dv4_i": "int4",
	"Dv4_j": "uint4",
	"Dv4_l": "long4",
	"Dv4_m": "ulong4",

	"Dv8_f": "float8",
	"Dv8_d": "double8",
	"Dv8_c": "char8",
	"Dv8_h": "uchar8",
	"Dv8_s": "short8",
	"Dv8_t": "ushort8",
	"Dv8_i": "int8",
	"Dv8_j": "uint8",
	"Dv8_l": "long8",
	"Dv8_m": "ulong8",

	"Dv16_f": "float16",
	"Dv16_d": "double16",
	"Dv16_c": "char16",
	"Dv16_h": "uchar16",
	"Dv16_s": "short16",
	"Dv16_t": "ushort16",
	"Dv16_i": "int16",
	"Dv16_j": "uint16",
	"Dv16_l": "long16",
	"Dv16_m": "ulong16",

}

LLVM_TYPE_EXT_MAP = {
	"b": " zeroext ",
	"v": "",

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
	"Dv4_d": "",

	"12memory_order": "",
	"12memory_scope": "",
}


MANGLING_AS_SPIR = {
	"global": "PU3AS1",
	"constant": "PU3AS2",
	"local": "PU3AS3",
	"generic": "PU3AS4",
	"private": "P",
	"none": ""
}

MANGLING_AS_OCL = {
	"global": "PU8CLglobal",
	"constant": "PU10CLconstant",
	"local": "PU7CLlocal",
	"private": "PU9CLprivate",
	"generic": "PU9CLgeneric",
	"none": ""
}

LLVM_SPIR_AS = {
	"global": " addrspace(1)",
	"constant": " addrspace(2)",
	"local": " addrspace(3)",
	"generic": " addrspace(4)",
	"private": " ",
	"none": " "
}

# returns a LLVM argument type with addrspace
# e.g. for argtype=="Pi" returns "i32 *"
def llvm_arg_type(argtype, AS):
	if argtype.count("ocl_image")>0 or argtype.count("ocl_sampler")>0 :
		return SIG_TO_LLVM_TYPE_MAP[argtype] + AS + "*"
	if argtype[0] == 'P':
		idx = 1
		if argtype[1] == 'V' or argtype[1] == 'K':
			idx = 2
			if argtype[2] == 'A':
				idx = 3
		return SIG_TO_LLVM_TYPE_MAP[argtype[idx:]] + AS + "*"
	else:
		return SIG_TO_LLVM_TYPE_MAP[argtype]

# mangles a type for mangled function name
# e.g. for ("Pi", "PU3AS3") returns "PU3AS3i"
def mang_suffix(argtype, AS_prefix):
	if argtype[0] == 'P':
		if argtype[1] == 'V':
			if argtype[2] == 'A':
				return AS_prefix + "VU7_Atomic" + argtype[3:]
			else:
				return AS_prefix + "V" + argtype[2:]
		elif argtype[1] == 'K':
			return AS_prefix + "K" + argtype[2:]
		else:
			return AS_prefix + argtype[1:]
	else:
		return argtype

# arg type without qualifiers
def pure_arg_type(argtype):
	if argtype[0] == 'P':
		idx = 1
		if argtype[1] == 'V' or argtype[1] == 'K':
			idx = 2
			if argtype[2] == 'A':
				idx = 3
		return argtype[idx:]
	else:
		return argtype

# replace type keep qualifiers
def replace_arg_type(argtype, replacement):
	if argtype[0] == 'P':
		idx = 1
		if argtype[1] == 'V' or argtype[1] == 'K':
			idx = 2
			if argtype[2] == 'A':
				idx = 3
		return argtype[0:idx] + replacement
	else:
		return replacement

def coerce_llvm_vector_type(type):
	if not X86_CALLING_ABI:
		return type
	if type in COERCE_VECTOR_MAP:
		return COERCE_VECTOR_MAP[type]
	else:
		return type

def byval_llvm_vector_type(type):
	if not X86_CALLING_ABI:
		return type
	if type in BYVAL_VECTOR_MAP:
		return BYVAL_VECTOR_MAP[type]
	else:
		return type


def generate_function(name, ret_type, ret_type_ext, multiAS, *args):
	"""

	:param name: function name
	:param ret_type: LLVM type ("i32", "float" etc) of retval
	:param ret_type_ext: retval's attributes ("signext" where required etc)
	:param multiAS: True = generate for all three SPIR AddrSpaces
	:param args: function arguments as mangled type names (i,j,m,f,d etc), not LLVM types
	"""
	ocl_func_name = POCL_LIB_PREFIX + name
	spir_func_name = name

	if not multiAS:
		addr_spaces = ["none"]
	else:
		if name.startswith("atomic"): # TODO
			addr_spaces = ["global", "local"]  # , "generic"]
		elif name.count("image")>0 or name.startswith("prefetch"):
			addr_spaces = ["global"]
		elif name.startswith("vload"):
			addr_spaces = ["global", "local", "private", "constant"]
		else:
			addr_spaces = ["global", "local", "private"]

	for AS in addr_spaces:

		spir_mangled_func_suffix = []
		ocl_mangled_func_suffix = []
		callee_args = []
		caller_args = []
		decl_args = []
		all_instr = []

		arg_i = 0
		# instr index, LLVM IR expects instructions to be numbered from 1
		llvm_i = 1

		# arg without qualifiers, saved for mangling name compression
		last_pure_arg = None

		####### process args
		for cast in args:

			####### generate mangled arg type name for the mangled function name
			# convert repeated vectors into compressed names, e.g.
			#   Dv2_cDv2_c -> Dv2_cS_
			# this is also done for pointers to vectors, e.g.
			#   Dv2_cPULocalDv2_c -> Dv2_cPULocalS_
			pure_arg = pure_arg_type(cast)
			actual_arg = cast
			if pure_arg == last_pure_arg and last_pure_arg.startswith("D"):
				actual_arg = replace_arg_type(actual_arg, "S_")
			if pure_arg == last_pure_arg and last_pure_arg == '12memory_order':
				actual_arg = replace_arg_type(actual_arg, "S4_")
			last_pure_arg = pure_arg

			# convert arg type to mangled type with AS
			# e.g. "Pi" -> "PU3AS3i"
			spir_cast_arg = mang_suffix(actual_arg, MANGLING_AS_SPIR[AS])
			spir_mangled_func_suffix.append(spir_cast_arg)

			# mangle OpenCL differently if required by target
			# e.g. "Pi" -> "PU8CLglobali"
			if MANGLE_OCL:
				ocl_cast_arg = mang_suffix(actual_arg, MANGLING_AS_OCL[AS])
			else:
				ocl_cast_arg = spir_cast_arg
			ocl_mangled_func_suffix.append(ocl_cast_arg)

			####### generate caller (SPIR wrapper function) llvm arg types
			# get LLVM type name (e.g. <2 x i32>) for arg
			spir_arg_type = llvm_arg_type(cast, LLVM_SPIR_AS[AS])
			# get coerced type
			coerced_arg_type = coerce_llvm_vector_type(spir_arg_type)
			# get byval type
			byval_arg_type = byval_llvm_vector_type(spir_arg_type)
			# get type with OpenCL AS
			ocl_arg_type = spir_arg_type
			if AS_CASTS_REQUIRED:
				ocl_arg_type = llvm_arg_type(cast, LLVM_SPIR_AS["none"])

			# arg type + arg-index, e.g. "<2 x i32> %2"
			# SPIR wrapper (=caller) arg. This one is always non-coerced with SPIR AS
			noext_caller_arg = spir_arg_type + " %" + chr(97+arg_i)
			caller_args.append(noext_caller_arg)

			####### generate arg types for callee (wrapped target function) and its declaration
			# handle coerced args. Pointer args are not coerced so
			# this shouldn't interact with addrspace casts.
			if coerced_arg_type != spir_arg_type:
				# handle 3-sized vectors
				if spir_arg_type.startswith("<3"):
					four_vec = "<4" + spir_arg_type[2:]
					all_instr.append("  %%%u = shufflevector %s, %s undef, <4 x i32> <i32 0, i32 1, i32 2, i32 undef>" % (llvm_i, noext_caller_arg, spir_arg_type))
					bitcasted_arg = four_vec + " %" + chr(48+llvm_i)
					llvm_i += 1
				else:
					bitcasted_arg = noext_caller_arg

				all_instr.append("  %%%u = bitcast %s to %s" % (llvm_i, bitcasted_arg, coerced_arg_type))
				callee_args.append(coerced_arg_type + " %" + chr(48+llvm_i))
				decl_args.append(coerced_arg_type)
				llvm_i += 1
			else:
				# handle pointer args. insert addrspace casts if required by target
				if spir_arg_type != ocl_arg_type:
					all_instr.append("  %%%u = addrspacecast %s to %s" % (llvm_i, noext_caller_arg, ocl_arg_type))
					callee_args.append(ocl_arg_type + " %" + chr(48+llvm_i))
					decl_args.append(ocl_arg_type)
					llvm_i += 1
				else:
					# handle byval args. insert store / loads as necessary
					if byval_arg_type != spir_arg_type:
						# add alloca & store
						alignment = byval_arg_type.find("align")
						if alignment > 0:
							align = byval_arg_type[alignment:]
						else:
							align = "align 8"
						all_instr.append("  %%%u = alloca %s, %s" % (llvm_i, spir_arg_type, align))
						alloca_inst = "%" + chr(48+llvm_i)
						all_instr.append("  store %s, %s* %s, %s" % (noext_caller_arg, spir_arg_type, alloca_inst, align))
						callee_args.append(ocl_arg_type + "* " + alloca_inst)
						decl_args.append(byval_arg_type)
						llvm_i += 1
					else:
						# nothing to do, just pass plain arg
						callee_args.append(ocl_arg_type + " %" + chr(97+arg_i))
						decl_args.append(ocl_arg_type)
			arg_i += 1

		######## generate final mangled function names
		spir_mangled_func_suffix = "".join(spir_mangled_func_suffix)
		ocl_mangled_func_suffix = "".join(ocl_mangled_func_suffix)
		caller_args = ", ".join(caller_args)
		callee_args = ", ".join(callee_args)
		decl_args = ", ".join(decl_args)

		spir_mangled_name = "@_Z%u%s%s" % (len(spir_func_name), spir_func_name, spir_mangled_func_suffix)
		ocl_mangled_name = "@_Z%u%s%s" % (len(ocl_func_name), ocl_func_name, ocl_mangled_func_suffix)

		####### generate function body
		coerced_ret_type = coerce_llvm_vector_type(ret_type)

		print("declare %s %s(%s) local_unnamed_addr #0" % (coerced_ret_type, ocl_mangled_name, decl_args))
		print("")
		print("define spir_func %s %s(%s) local_unnamed_addr #0 {" % (ret_type_ext + ret_type, spir_mangled_name, caller_args))

		if all_instr:
			for cast in all_instr:
				print(cast)

		if ret_type == 'void':
			print("  tail call %s %s(%s)" % (ret_type, ocl_mangled_name, callee_args))
			print("  ret void" )
		else:
			if ret_type != coerced_ret_type:
				print("  %%coerced_call = tail call %s %s(%s)" % (coerced_ret_type, ocl_mangled_name, callee_args))
				if ret_type.startswith("<3"):
					four_vec = "<4" + ret_type[2:]
					print("  %%bc_ret = bitcast %s %%coerced_call to %s" % (coerced_ret_type, four_vec))
					print("  %%final_ret = shufflevector %s %%bc_ret, %s undef, <3 x i32> <i32 0, i32 1, i32 2>" % (four_vec, four_vec))
				else:
					print("  %%final_ret = bitcast %s %%coerced_call to %s" % (coerced_ret_type, ret_type))

				print("  ret %s %%final_ret" % ret_type)
			else:
				print("  %%call = tail call %s %s(%s)" % (ret_type, ocl_mangled_name, callee_args))
				print("  ret %s %%call" % ret_type)
		print("}\n\n")


##############################################################

SPIR_MODULE_PREFIX = {
64:
"""
; ModuleID = 'spir_wrapper.bc'
source_filename = "generate_spir_wrapper.py"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

%opencl.sampler_t = type opaque
%opencl.event_t = type opaque

%opencl.image1d_ro_t = type opaque
%opencl.image2d_ro_t = type opaque
%opencl.image3d_ro_t = type opaque

%opencl.image1d_wo_t = type opaque
%opencl.image2d_wo_t = type opaque
%opencl.image3d_wo_t = type opaque

%opencl.image1d_rw_t = type opaque
%opencl.image2d_rw_t = type opaque
%opencl.image3d_rw_t = type opaque

%opencl.image1d_array_ro_t = type opaque
%opencl.image1d_array_wo_t = type opaque
%opencl.image1d_array_rw_t = type opaque
%opencl.image2d_array_ro_t = type opaque
%opencl.image2d_array_wo_t = type opaque
%opencl.image2d_array_rw_t = type opaque

%opencl.image1d_buffer_ro_t = type opaque
%opencl.image1d_buffer_rw_t = type opaque
%opencl.image1d_buffer_wo_t = type opaque

""",


32:
"""
; ModuleID = 'spir_wrapper.bc'
source_filename = "generate_spir_wrapper.py"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

%opencl.sampler_t = type opaque
%opencl.event_t = type opaque

%opencl.image1d_ro_t = type opaque
%opencl.image2d_ro_t = type opaque
%opencl.image3d_ro_t = type opaque

%opencl.image1d_wo_t = type opaque
%opencl.image2d_wo_t = type opaque
%opencl.image3d_wo_t = type opaque

%opencl.image1d_rw_t = type opaque
%opencl.image2d_rw_t = type opaque
%opencl.image3d_rw_t = type opaque

%opencl.image1d_array_ro_t = type opaque
%opencl.image1d_array_wo_t = type opaque
%opencl.image1d_array_rw_t = type opaque
%opencl.image2d_array_ro_t = type opaque
%opencl.image2d_array_wo_t = type opaque
%opencl.image2d_array_rw_t = type opaque

%opencl.image1d_buffer_ro_t = type opaque
%opencl.image1d_buffer_rw_t = type opaque
%opencl.image1d_buffer_wo_t = type opaque

"""
}


triple = SPIR_MODULE_PREFIX[64]
if (len(sys.argv) > 1) and (sys.argv[1] == "32"):
	triple = SPIR_MODULE_PREFIX[32]

print(triple)

MANG_TYPES_32 = {
	"f": "f",
	"Pf": "Pf",
	'i': "i",
	'u': "j"
}

MANG_TYPES_64 = {
	"f": "d",
	"Pf": "Pd",
	'i': "l",
	'u': "m"
}

# math funcs, vectorized
for llvm_type in ["float", "double"]:
	for vector_size in [1,2,3,4,8,16]:

		if llvm_type == "float":
			arg_Pf = 'Pf'
			arg_f = 'f'
			arg_li = 'i'
			arg_lu = 'j'
		else:
			arg_Pf = 'Pd'
			arg_f = 'd'
			arg_li = 'l'
			arg_lu = 'm'
		arg_i = 'i'
		arg_Pi = 'Pi'
		ret_type = llvm_type
		rel_rettype = 'i32'

		if vector_size > 1:
			s = str(vector_size)
			arg_Pi = "PDv" + s + "_i"
			arg_i = "Dv" + s + "_i"
			arg_Pf = "PDv" + s + "_"+arg_f
			arg_f = "Dv" + s + "_"+arg_f
			arg_li = "Dv" + s + "_"+arg_li
			arg_lu = "Dv" + s + "_"+arg_lu
			ret_type = "<" + s + " x " + llvm_type + ">"
			if llvm_type == "float":
				rel_rettype = "<" + s + " x i32>"
			else:
				rel_rettype = "<" + s + " x i64>"

		for f in SINGLE_ARG:
			generate_function(f, ret_type, '', False, arg_f)
		for f in DUAL_ARG:
			generate_function(f, ret_type, '', False, arg_f, arg_f)
		for f in DUAL_ARG_PTR:
			generate_function(f, ret_type, '', True, arg_f, arg_Pf)
		for f in TRIPLE_ARG:
			generate_function(f, ret_type, '', False, arg_f, arg_f, arg_f)

		# math funcs with other / special signatures
		generate_function("ilogb", "i32", '', False, arg_f)

		generate_function("ldexp", ret_type, '', False, arg_f, arg_i)
		generate_function("pown", ret_type, '', False, arg_f, arg_i)
		generate_function("rootn", ret_type, '', False, arg_f, arg_i)

		generate_function("remquo", ret_type, '', True, arg_f, arg_f, arg_Pi)

		generate_function("nan", ret_type, '', False, arg_lu)

		generate_function("lgamma_r", ret_type, '', True, arg_f, arg_Pi)
		generate_function("frexp", ret_type, '', True, arg_f, arg_Pi)

		generate_function("signbit", rel_rettype, '', False, arg_f)

		generate_function("isequal", rel_rettype, '', False, arg_f, arg_f)
		generate_function("isnotequal", rel_rettype, '', False, arg_f, arg_f)
		generate_function("isgreater", rel_rettype, '', False, arg_f, arg_f)
		generate_function("isgreaterequal", rel_rettype, '', False, arg_f, arg_f)
		generate_function("isless", rel_rettype, '', False, arg_f, arg_f)
		generate_function("islessequal", rel_rettype, '', False, arg_f, arg_f)
		generate_function("islessgreater", rel_rettype, '', False, arg_f, arg_f)

		generate_function("isfinite", rel_rettype, '', False, arg_f)
		generate_function("isinf", rel_rettype, '', False, arg_f)
		generate_function("isnan", rel_rettype, '', False, arg_f)
		generate_function("isnormal", rel_rettype, '', False, arg_f)
		generate_function("isordered", rel_rettype, '', False, arg_f, arg_f)
		generate_function("isunordered", rel_rettype, '', False, arg_f, arg_f)


# geometric functions
generate_function("cross", "<4 x float>", '', False, "Dv4_f", "Dv4_f")
generate_function("cross", "<4 x double>", '', False, "Dv4_d", "Dv4_d")
generate_function("cross", "<3 x float>", '', False, "Dv3_f", "Dv3_f")
generate_function("cross", "<3 x double>", '', False, "Dv3_d", "Dv3_d")

for W in ["1", "2","3","4"]:
	arg_f = "f"
	arg_d = "d"
	if W != '1':
		arg_f = "Dv" + W + "_"+arg_f
		arg_d = "Dv" + W + "_"+arg_d
	generate_function("dot", "float", '', False, arg_f, arg_f)
	generate_function("dot", "double", '', False, arg_d, arg_d)
	generate_function("distance", "float", '', False, arg_f, arg_f)
	generate_function("distance", "double", '', False, arg_d, arg_d)
	generate_function("length", "float", '', False, arg_f)
	generate_function("length", "double", '', False, arg_d)
	generate_function("normalize", "float", '', False, arg_f)
	generate_function("normalize", "double", '', False, arg_d)
	generate_function("fast_distance", "float", '', False, arg_f, arg_f)
	generate_function("fast_distance", "double", '', False, arg_d, arg_d)
	generate_function("fast_length", "float", '', False, arg_f)
	generate_function("fast_length", "double", '', False, arg_d)
	generate_function("fast_normalize", "float", '', False, arg_f)
	generate_function("fast_normalize", "double", '', False, arg_d)

# upsample has special arguments
UPSAMPLE_1ST_ARG = {
	's': 'c',
	't': 'h',
	'i': 's',
	'j': 't',
	'l': 'i',
	'm': 'j'
}

UPSAMPLE_2ND_ARG = {
	's': 'h',
	't': 'h',
	'i': 't',
	'j': 't',
	'l': 'j',
	'm': 'j'
}

# upsample
for mang_type in ['s', 't', 'i', 'j', 'l', 'm']:
	for vector_size in [1,2,3,4,8,16]:
		arg_1st = UPSAMPLE_1ST_ARG[mang_type]
		arg_2nd = UPSAMPLE_2ND_ARG[mang_type]
		if vector_size > 1:
			s = str(vector_size)
			arg_1st = "Dv" + s + "_" + arg_1st
			arg_2nd = "Dv" + s + "_" + arg_2nd
		generate_function("upsample", SIG_TO_LLVM_TYPE_MAP[mang_type], '', False, arg_1st, arg_2nd)

# vload / vstore
for arg_type in ['c', 'h', 's', 't', 'i', 'j', 'l', 'm', 'f', 'd']:
	for vector_size in [2,3,4,8,16]:
		arg = arg_type
		PConstArg = 'PK' + arg_type
		PArg = 'P' + arg_type
		if vector_size > 1:
			s = str(vector_size)
#			PConstArg = "PDv" + s + "_" + arg_type
			arg = "Dv" + s + "_" + arg_type

		generate_function("vload"+str(vector_size), SIG_TO_LLVM_TYPE_MAP[arg], '', True, 'm', PConstArg)
		generate_function("vstore"+str(vector_size), SIG_TO_LLVM_TYPE_MAP['v'], '', True, arg, 'm', PArg)
		generate_function("prefetch", SIG_TO_LLVM_TYPE_MAP['v'], '', True, arg, PConstArg, 'm')

INVERT_SIGN = {
	"c": "h",
	"h": "c",

	"s": "t",
	"t": "s",

	"i": "j",
	"j": "i",

	"l": "m",
	"m": "l",
}

# Integer
for arg_type in ['c', 'h', 's', 't', 'i', 'j', 'l', 'm']:
	for vector_size in [1,2,3,4,8,16]:

		arg_i = arg_type
		arg_inv = INVERT_SIGN[arg_type]
		signext = LLVM_TYPE_EXT_MAP[arg_type]
		if vector_size > 1:
			s = str(vector_size)
			arg_i = "Dv" + s + "_" + arg_i
			arg_inv = "Dv" + s + "_" + arg_inv
			signext = ''

		for f in SINGLE_ARG_I:
			generate_function(f, SIG_TO_LLVM_TYPE_MAP[arg_i], signext, False, arg_i)
		for f in DUAL_ARG_I:
			generate_function(f, SIG_TO_LLVM_TYPE_MAP[arg_i], signext, False, arg_i, arg_i)
		for f in TRIPLE_ARG_I:
			generate_function(f, SIG_TO_LLVM_TYPE_MAP[arg_i], signext, False, arg_i, arg_i, arg_i)
		# TODO unsigned ??
		generate_function("any", "i32", '', False, arg_i)
		generate_function("all", "i32", '', False, arg_i)
		# must generate with last arg un/signed types too
		generate_function("select", SIG_TO_LLVM_TYPE_MAP[arg_i], '', False, arg_i, arg_i, arg_i)
		generate_function("select", SIG_TO_LLVM_TYPE_MAP[arg_i], '', False, arg_i, arg_i, arg_inv)

# shuffle / shuffle2
for arg_type in ['c', 'h', 's', 't', 'i', 'j', 'l', 'm']:
	if arg_type in ['c', 's', 'i', 'l']:
		uarg_type = INVERT_SIGN[arg_type]
	else:
		uarg_type = arg_type

	for vector_size in [2,3,4,8,16]:
		for perm_size in [2,3,4,8,16]:
			ret_type = "Dv" + str(perm_size) + "_" + arg_type
			in_type = "Dv" + str(vector_size) + "_" + arg_type
			mask_type = "Dv" + str(perm_size) + "_" + uarg_type
			generate_function("shuffle", SIG_TO_LLVM_TYPE_MAP[ret_type], '', False, in_type, mask_type)
			generate_function("shuffle2", SIG_TO_LLVM_TYPE_MAP[ret_type], '', False, in_type, in_type, mask_type)

# convert
for dst_type in ['c', 'h', 's', 't', 'i', 'j', 'l', 'm', 'f', 'd']:
	for src_type in ['c', 'h', 's', 't', 'i', 'j', 'l', 'm', 'f', 'd']:
		for sat in ['', '_sat']:
			if (sat == '_sat') and (dst_type in ['f','d']):
				continue
			for rounding in ['','_rtp','_rtn','_rte','_rtz']:
				for vector_size in [1,2,3,4,8,16]:
					dst_t = dst_type
					src_t = src_type
					signext = LLVM_TYPE_EXT_MAP[dst_type]
					if vector_size > 1:
						s = str(vector_size)
						dst_t = "Dv" + s + "_" + dst_t
						src_t = "Dv" + s + "_" + src_t
						signext = ''
					generate_function('convert_'+SIG_TO_TYPE_NAME_MAP[dst_t]+sat+rounding, SIG_TO_LLVM_TYPE_MAP[dst_t], signext, False, src_t)

IMG_COLOR_TYPE_MAP = {
	'i' : 'Dv4_i',
	'ui' : 'Dv4_j',
	'f' : 'Dv4_f',
}

IMG_COORD_SIZE = {
	'image1d': 1,
	'image2d': 2,
	'image3d': 4,
	'image1d_array': 2,
	'image2d_array': 4,
	'image1d_buffer': 1
}


# images
for img_type in ['image1d', 'image2d', 'image3d', 'image1d_array', 'image2d_array', 'image1d_buffer']:
	coord_float_type = 'f'
	coord_int_type = 'i'
	size = IMG_COORD_SIZE[img_type]
	if size > 1:
		coord_float_type = "Dv" + str(size) + "_f"
		coord_int_type = "Dv" + str(size) + "_i"
	for access in ['ro', 'wo', 'rw']:
		img_type_llvm = 'ocl_' + img_type + "_" + access
		img_type_llvm = str(len(img_type_llvm)) + img_type_llvm

		for color_type in ['i', 'ui', 'f']:
			# return value for read_image or input value for write_image
			mang_color_type = IMG_COLOR_TYPE_MAP[color_type]

			if access != 'wo':
				generate_function('read_image'+color_type, SIG_TO_LLVM_TYPE_MAP[mang_color_type], '', True, img_type_llvm, '11ocl_sampler', coord_float_type)
				generate_function('read_image'+color_type, SIG_TO_LLVM_TYPE_MAP[mang_color_type], '', True, img_type_llvm, '11ocl_sampler', coord_int_type)
				generate_function('read_image'+color_type, SIG_TO_LLVM_TYPE_MAP[mang_color_type], '', True, img_type_llvm, coord_int_type)
			if access != 'ro':
				generate_function('write_image'+color_type, 'void', '', True, img_type_llvm, coord_int_type, mang_color_type)

		generate_function('get_image_channel_data_type', 'i32', '', True, img_type_llvm)
		generate_function('get_image_channel_order', 'i32', '', True, img_type_llvm)
		generate_function('get_image_width', 'i32', '', True, img_type_llvm)
		if img_type.startswith('image2') or img_type.startswith('image3'):
			generate_function('get_image_height', 'i32', '', True, img_type_llvm)
		if img_type in ['image2d_t','image2d_array_t','image2d_depth_t', 'image2d_array_depth_t']:
			generate_function('get_image_dim', '<2 x i32>', '', True, img_type_llvm)
		if img_type == 'image3d':
			generate_function('get_image_depth', 'i32', '', True, img_type_llvm)
			generate_function('get_image_dim', '<4 x i32>', '', True, img_type_llvm)
		if img_type in ['image2d_array_t', 'image2d_array_depth_t', 'image1d_array_t']:
			generate_function('get_image_array_size', 'i64', '', True, img_type_llvm)


# Atomics
for mang_type in ['i', 'j', "l", "m"]:
	for f in SVM_ATOMICS_INT_ONLY:
		generate_function(f, SIG_TO_LLVM_TYPE_MAP[mang_type], LLVM_TYPE_EXT_MAP[mang_type], True, 'PVA'+mang_type, mang_type)
		generate_function(f+"_explicit", SIG_TO_LLVM_TYPE_MAP[mang_type], LLVM_TYPE_EXT_MAP[mang_type], True, 'PVA'+mang_type, mang_type, "12memory_order")
		generate_function(f+"_explicit", SIG_TO_LLVM_TYPE_MAP[mang_type], LLVM_TYPE_EXT_MAP[mang_type], True, 'PVA'+mang_type, mang_type, "12memory_order", "12memory_scope")

for mang_type in ['i', 'j', "l", "m"]:
	for f in OLD_ATOMICS_INT_ONLY:
		generate_function(f, SIG_TO_LLVM_TYPE_MAP[mang_type], LLVM_TYPE_EXT_MAP[mang_type], True, 'PV'+mang_type, mang_type)
	# dec, inc take only 1 argument
	generate_function("atomic_inc", SIG_TO_LLVM_TYPE_MAP[mang_type], LLVM_TYPE_EXT_MAP[mang_type], True, 'PV'+mang_type)
	generate_function("atomic_dec", SIG_TO_LLVM_TYPE_MAP[mang_type], LLVM_TYPE_EXT_MAP[mang_type], True, 'PV'+mang_type)

for mang_type in ['i', 'j', "l", "m", "f", "d"]:
	generate_function("atomic_xchg", SIG_TO_LLVM_TYPE_MAP[mang_type], LLVM_TYPE_EXT_MAP[mang_type], True, 'PV'+mang_type, mang_type)
	generate_function("atomic_cmpxchg", SIG_TO_LLVM_TYPE_MAP[mang_type], LLVM_TYPE_EXT_MAP[mang_type], True, 'PV'+mang_type, mang_type, mang_type)


for mang_type in ['i', 'j', "l", "m", "f", "d"]:
	generate_function("atomic_init", SIG_TO_LLVM_TYPE_MAP['v'], LLVM_TYPE_EXT_MAP['v'], True, 'PVA'+mang_type, mang_type)
	for f in SVM_ATOMICS_ALL:
		args = None
		cmpxchg = False
		orders = ["12memory_order"]
		if f == "atomic_store":
			args = ['PVA'+mang_type, mang_type]
			ret = "v"
		if f == "atomic_load":
			args = ['PVA'+mang_type]
			ret = mang_type
		if f == "atomic_exchange":
			args = ['PVA'+mang_type, mang_type]
			ret = mang_type
		if f == "atomic_compare_exchange_strong" or f == "atomic_compare_exchange_weak":
			args = ['PVA'+mang_type, 'P'+mang_type, mang_type]
			ret = "b"
			orders = ["12memory_order", "12memory_order"]

		generate_function(f, SIG_TO_LLVM_TYPE_MAP[mang_type], LLVM_TYPE_EXT_MAP[mang_type], True, *args)
		generate_function(f+"_explicit", SIG_TO_LLVM_TYPE_MAP[mang_type], LLVM_TYPE_EXT_MAP[mang_type], True, *args, *orders)
		generate_function(f+"_explicit", SIG_TO_LLVM_TYPE_MAP[mang_type], LLVM_TYPE_EXT_MAP[mang_type], True, *args, *orders, "12memory_scope")

for f in SVM_ATOMICS_FLAGS:
	generate_function(f, SIG_TO_LLVM_TYPE_MAP['b'], LLVM_TYPE_EXT_MAP['b'], True, 'PVAi')
	generate_function(f+"_explicit", SIG_TO_LLVM_TYPE_MAP['b'], LLVM_TYPE_EXT_MAP['b'], True, 'PVAi', "12memory_order")
	generate_function(f+"_explicit", SIG_TO_LLVM_TYPE_MAP['b'], LLVM_TYPE_EXT_MAP['b'], True, 'PVAi', "12memory_order", "12memory_scope")



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
