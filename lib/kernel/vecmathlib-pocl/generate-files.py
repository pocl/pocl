#! /usr/bin/env python

import re, sys



# Types:
SI = "SI"                       # int/long
SK = "SK"                       # int (even for double)
SF = "SF"                       # float/double
VB = "VB"                       # boolN
VI = "VI"                       # intN/longN
VJ = "VJ"                       # intN/longN (except int1 for double1)
VK = "VK"                       # intN (even for doubleN)
VU = "VU"                       # uintN/ulongN
VF = "VF"                       # floatN/doubleN
PVK = "PVK"                     # pointer to VK
PVF = "PVF"                     # pointer to VF

# Each function is described by a tuple with the following entries:
#    1. name
#    2. external argument types (see above)
#    3. external return type
#    4. vecmathlib argument types (see above)
#    5. vecmathlib return type
# This allows generating externally visible functions with different
# signatures, e.g. to support OpenCL.
vmlfuncs = [
    # Section 6.12.2
    ("acos"     , [VF        ], VF, [VF        ], VF),
    ("acosh"    , [VF        ], VF, [VF        ], VF),
    ("asin"     , [VF        ], VF, [VF        ], VF),
    ("asinh"    , [VF        ], VF, [VF        ], VF),
    ("atan"     , [VF        ], VF, [VF        ], VF),
    ("atanh"    , [VF        ], VF, [VF        ], VF),
    ("cbrt"     , [VF        ], VF, [VF        ], VF),
    ("ceil"     , [VF        ], VF, [VF        ], VF),
    ("copysign" , [VF, VF    ], VF, [VF, VF    ], VF),
    ("cos"      , [VF        ], VF, [VF        ], VF),
    ("cosh"     , [VF        ], VF, [VF        ], VF),
    ("exp"      , [VF        ], VF, [VF        ], VF),
    ("exp2"     , [VF        ], VF, [VF        ], VF),
    ("exp10"    , [VF        ], VF, [VF        ], VF),
    ("expm1"    , [VF        ], VF, [VF        ], VF),
    ("fabs"     , [VF        ], VF, [VF        ], VF),
    ("fdim"     , [VF, VF    ], VF, [VF, VF    ], VF),
    ("floor"    , [VF        ], VF, [VF        ], VF),
    ("fma"      , [VF, VF, VF], VF, [VF, VF, VF], VF),
    ("fmax"     , [VF, VF    ], VF, [VF, VF    ], VF),
    ("fmin"     , [VF, VF    ], VF, [VF, VF    ], VF),
    ("fmod"     , [VF, VF    ], VF, [VF, VF    ], VF),
    ("hypot"    , [VF, VF    ], VF, [VF, VF    ], VF),
    ("ilogb_"   , [VF        ], VJ, [VF        ], VI), # should return VK
    ("ldexp_"   , [VF, VJ    ], VF, [VF, VI    ], VF), # should take VK
    ("ldexp_"   , [VF, SK    ], VF, [VF, SI    ], VF), # should take VK
    ("log"      , [VF        ], VF, [VF        ], VF),
    ("log2"     , [VF        ], VF, [VF        ], VF),
    ("log10"    , [VF        ], VF, [VF        ], VF),
    ("log1p"    , [VF        ], VF, [VF        ], VF),
    ("pow"      , [VF, VF    ], VF, [VF, VF    ], VF),
    ("remainder", [VF, VF    ], VF, [VF, VF    ], VF),
    ("rint"     , [VF        ], VF, [VF        ], VF),
    ("round"    , [VF        ], VF, [VF        ], VF),
    ("rsqrt"    , [VF        ], VF, [VF        ], VF),
    ("sin"      , [VF        ], VF, [VF        ], VF),
    ("sinh"     , [VF        ], VF, [VF        ], VF),
    ("sqrt"     , [VF        ], VF, [VF        ], VF),
    ("tan"      , [VF        ], VF, [VF        ], VF),
    ("tanh"     , [VF        ], VF, [VF        ], VF),
    ("trunc"    , [VF        ], VF, [VF        ], VF),
    
    # Section 6.12.6
    ("isfinite" , [VF        ], VJ, [VF        ], VB),
    ("isinf"    , [VF        ], VJ, [VF        ], VB),
    ("isnan"    , [VF        ], VJ, [VF        ], VB),
    ("isnormal" , [VF        ], VJ, [VF        ], VB),
    ("signbit"  , [VF        ], VJ, [VF        ], VB),
    ]
    
directfuncs = [
    # Section 6.12.2
    ("acospi"        , [VF         ], VF, "acos(x0)/(scalar_t)M_PI"),
    ("asinpi"        , [VF         ], VF, "asin(x0)/(scalar_t)M_PI"),
    ("atanpi"        , [VF         ], VF, "atan(x0)/(scalar_t)M_PI"),
    ("atan2"         , [VF, VF     ], VF, "({ vector_t a=atan(x0/x1); x1>(scalar_t)0.0f ? a : x1<(scalar_t)0.0f ? a+copysign((scalar_t)M_PI,x0) : copysign((scalar_t)M_PI_2,x0); })"),
    ("atan2pi"       , [VF, VF     ], VF, "atan2(x0,x1)/(scalar_t)M_PI"),
    ("cospi"         , [VF         ], VF, "cos((scalar_t)M_PI*x0)"),
    ("fmax"          , [VF, SF     ], VF, "fmax(x0,(vector_t)x1)"),
    ("fmin"          , [VF, SF     ], VF, "fmin(x0,(vector_t)x1)"),
    ("fract"         , [VF, PVF    ], VF, "*x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN)"),
    ("frexp"         , [VF, PVK    ], VF, "*x1=ilogb(x0), ldexp(x0,-ilogb(x0))"),
    ("ilogb"         , [VF         ], VK, "convert_kvector_t(({ __attribute__((__overloadable__)) jvector_t ilogb_(vector_t); jvector_t jmin=sizeof(jvector_t)==sizeof(int)?INT_MIN:LONG_MIN; jvector_t r=ilogb_(x0); select(r, (jvector_t)FP_ILOGB0, r==jmin); }))"),
    ("ldexp"         , [VF, VK     ], VF, "({ __attribute__((__overloadable__)) vector_t ldexp_(vector_t,jvector_t); ldexp_(x0,convert_ivector_t(x1)); })"),
    ("ldexp"         , [VF, SK     ], VF, "({ __attribute__((__overloadable__)) vector_t ldexp_(vector_t,kscalar_t); ldexp_(x0,(kscalar_t)x1); })"),
    ("logb"          , [VF         ], VF, "convert_vector_t(ilogb(x0))"),
    ("mad"           , [VF, VF, VF ], VF, "fma(x0,x1,x2)"),
    ("maxmag"        , [VF, VF     ], VF, "fabs(x0)>fabs(x1) ? x0 : fabs(x1)>fabs(x0) ? x1 : fmax(x0,x1)"),
    ("minmag"        , [VF, VF     ], VF, "fabs(x0)<fabs(x1) ? x0 : fabs(x1)<fabs(x0) ? x1 : fmin(x0,x1)"),
    ("modf"          , [VF, PVF    ], VF, "*x1=trunc(x0), copysign(x0-trunc(x0),x0)"),
    ("nan"           , [VU         ], VF, "(scalar_t)0.0f/(scalar_t)0.0f"),
    ("pown"          , [VF, VK     ], VF, "pow(x0,convert_vector_t(x1))"),
    ("powr"          , [VF, VF     ], VF, "pow(x0,x1)"),
    ("remquo"        , [VF, VF, PVK], VF, "({ vector_t k=rint(x0/x1); *x2=(convert_kvector_t(k)&0x7f)*(1-2*convert_kvector_t(signbit(k))); x0-k*x1; })"),
    ("rootn"         , [VF, VK     ], VF, "pow(x0,(scalar_t)1.0f/convert_vector_t(x1))"),
    ("sincos"        , [VF, PVF    ], VF, "*x1=cos(x0), sin(x0)"),
    ("sinpi"         , [VF         ], VF, "sin((scalar_t)M_PI*x0)"),
    ("tanpi"         , [VF         ], VF, "tan((scalar_t)M_PI*x0)"),
    
    # Section 6.12.2, half_ functions
    ("half_cos"      , [VF         ], VF, "cos(x0)"),
    ("half_divide"   , [VF, VF     ], VF, "x0/x1"),
    ("half_exp"      , [VF         ], VF, "exp(x0)"),
    ("half_exp2"     , [VF         ], VF, "exp2(x0)"),
    ("half_exp10"    , [VF         ], VF, "exp10(x0)"),
    ("half_log"      , [VF         ], VF, "log(x0)"),
    ("half_log2"     , [VF         ], VF, "log2(x0)"),
    ("half_log10"    , [VF         ], VF, "log10(x0)"),
    ("half_powr"     , [VF, VF     ], VF, "powr(x0,x1)"),
    ("half_recip"    , [VF         ], VF, "(scalar_t)1.0f/x0"),
    ("half_rsqrt"    , [VF         ], VF, "rsqrt(x0)"),
    ("half_sin"      , [VF         ], VF, "sin(x0)"),
    ("half_sqrt"     , [VF         ], VF, "sqrt(x0)"),
    ("half_tan"      , [VF         ], VF, "tan(x0)"),
    # Section 6.12.2, native_ functions
    ("native_cos"    , [VF         ], VF, "cos(x0)"),
    ("native_divide" , [VF, VF     ], VF, "x0/x1"),
    ("native_exp"    , [VF         ], VF, "exp(x0)"),
    ("native_exp2"   , [VF         ], VF, "exp2(x0)"),
    ("native_exp10"  , [VF         ], VF, "exp10(x0)"),
    ("native_log"    , [VF         ], VF, "log(x0)"),
    ("native_log2"   , [VF         ], VF, "log2(x0)"),
    ("native_log10"  , [VF         ], VF, "log10(x0)"),
    ("native_powr"   , [VF, VF     ], VF, "powr(x0,x1)"),
    ("native_recip"  , [VF         ], VF, "(scalar_t)1.0f/x0"),
    ("native_rsqrt"  , [VF         ], VF, "rsqrt(x0)"),
    ("native_sin"    , [VF         ], VF, "sin(x0)"),
    ("native_sqrt"   , [VF         ], VF, "sqrt(x0)"),
    ("native_tan"    , [VF         ], VF, "tan(x0)"),
    
    # Section 6.12.4
    ("clamp"         , [VF, VF, VF ], VF, "fmin(fmax(x0,x1),x2)"),
    ("clamp"         , [VF, SF, SF ], VF, "fmin(fmax(x0,x1),x2)"),
    ("degrees"       , [VF         ], VF, "(scalar_t)(180/M_PI)*x0"),
    ("max"           , [VF, VF     ], VF, "fmax(x0,x1)"),
    ("max"           , [VF, SF     ], VF, "fmax(x0,x1)"),
    ("min"           , [VF, VF     ], VF, "fmin(x0,x1)"),
    ("min"           , [VF, SF     ], VF, "fmin(x0,x1)"),
    ("mix"           , [VF, VF, VF ], VF, "x0+(x1-x0)*x2"),
    ("mix"           , [VF, VF, SF ], VF, "x0+(x1-x0)*x2"),
    ("radians"       , [VF         ], VF, "(scalar_t)(M_PI/180)*x0"),
    ("step"          , [VF, VF     ], VF, "x1<x0 ? (vector_t)(scalar_t)0.0f : (vector_t)(scalar_t)1.0f"),
    ("step"          , [SF, VF     ], VF, "x1<x0 ? (vector_t)(scalar_t)0.0f : (vector_t)(scalar_t)1.0f"),
    ("smoothstep"    , [VF, VF, VF ], VF, "({ vector_t t = clamp((x2-x0)/(x1-x0), (scalar_t)0.0f, (scalar_t)1.0f); t*t*((scalar_t)3.0-(scalar_t)2.0*t); })"),
    ("smoothstep"    , [SF, SF, VF ], VF, "({ vector_t t = clamp((x2-x0)/(x1-x0), (scalar_t)0.0f, (scalar_t)1.0f); t*t*((scalar_t)3.0-(scalar_t)2.0*t); })"),
    ("sign"          , [VF         ], VF, "copysign(x0!=(scalar_t)0.0f ? (vector_t)(scalar_t)1.0f : (vector_t)(scalar_t)0.0f,x0)"),
    
    # Section 6.12.6
    ("isequal"       , [VF, VF     ], VJ, "x0==x1"),
    ("isnotequal"    , [VF, VF     ], VJ, "x0!=x1"),
    ("isgreater"     , [VF, VF     ], VJ, "x0>x1"),
    ("isgreaterequal", [VF, VF     ], VJ, "x0>=x1"),
    ("isless"        , [VF, VF     ], VJ, "x0<x1"),
    ("islessequal"   , [VF, VF     ], VJ, "x0<=x1"),
    ("islessgreater" , [VF, VF     ], VJ, "x0<x1 || x0>x1"),
    ("isordered"     , [VF, VF     ], VJ, "!isunordered(x0,x1)"),
    ("isunordered"   , [VF, VF     ], VJ, "isnan(x0) || isnan(x1)"),
]

# Missing functions from 6.12.2: erfc, erf, lgamma, lgamma_r,
# nextafter, tgamma

# Unchecked: 6.12.3 (integer functions)

# Missing functions from 6.12.6 (relational functions): any, all,
# bitselect, select

# Unchecked: 6.12.7 (vector data load and store functions)

# Unchecked: 6.12.12 (miscellaneous vector functions)



# This is always prepended to the generated function names.
func_prefix = "_cl_"

# Some of the functions need prefixes to avoid using the C standard
# library ones.
masked_functions = [
    "acos",
    "asin",
    "atan",
    "atan2",
    "ceil",
    "copysign",
    "cos",
    "exp",
    "exp2",
    "fabs",
    "floor",
    "fma",
    "fmax",
    "fmin",
    "log",
    "log2",
    "pow",
    "rint",
    "round",
    "sin",
    "sqrt",
    "tan",
    "trunc",
]

# This is prepended to masked function names.
mask_prefix = ""

def prefixed(name):
    if name in masked_functions: name = mask_prefix + name
    return func_prefix + name



outfile = None
outfile_did_truncate = set()
def out(str): outfile.write("%s\n" % str)
def out_open(name):
    global outfile
    global outfile_did_truncate
    if outfile: raise "file already open"
    is_first_open = name not in outfile_did_truncate
    if is_first_open:
        outfile = open(name, "w")
        outfile.close()
        outfile_did_truncate.add(name)
        print name,
        sys.stdout.flush()
    outfile = open(name, "a")
    return is_first_open
def out_close():
    global outfile
    outfile.close()
    outfile = None

declfile = None
def decl(str):
    if str=="" or str.startswith("//") or str.startswith("#"):
        declfile.write("%s\n" % str)
    else:
        declfile.write("__attribute__((__overloadable__)) %s;\n" % str)
def decl_open(name):
    global declfile
    declfile = open(name, "w")
def decl_close():
    global declfile
    declfile.close()
    declfile = None



def mktype(tp, vectype):
    (space, basetype, sizename) = (
        re.match("(global|local|private)?(float|double)([0-9]*)", vectype).
        groups())
    size = 1 if sizename=="" else int(sizename)
    if tp==SK:
        if size==1: return "int"
        return "int" if basetype=="float" else "long"
    if tp==SF:
        return basetype
    if tp==VI:
        ibasetype = "int" if basetype=="float" else "long"
        return "%s%s" % (ibasetype, sizename)
    if tp==VJ:
        if size==1: return "int"
        ibasetype = "int" if basetype=="float" else "long"
        return "%s%s" % (ibasetype, sizename)
    if tp==VK:
        return "int%s" % sizename
    if tp==PVK:
        if space=="": raise "wrong address space"
        return "%s int%s*" % (space, sizename)
    if tp==VU:
        ibasetype = "uint" if basetype=="float" else "ulong"
        return "%s%s" % (ibasetype, sizename)
    if tp==VF:
        return "%s%s" % (basetype, sizename)
    if tp==PVF:
        if space=="": raise "wrong address space"
        return "%s %s%s*" % (space, basetype, sizename)
    raise "unreachable"

def mkvmltype(tp, vectype):
    if tp==SI: return vectype+"::int_t"
    if tp==SF: return vectype+"::real_t"
    if tp==VB: return vectype+"::boolvec_t"
    if tp in (VI,VJ): return vectype+"::intvec_t"
    if tp==VF: return vectype
    raise "unreachable"



def output_vmlfunc_vml(func, vectype):
    (name, args, ret, vmlargs, vmlret) = func
    out("// Implement %s by calling vecmathlib" % name)
    (basetype, size) = re.match("([A-Za-z]+)([0-9]*)", vectype).groups()
    size = 1 if size=="" else int(size)
    vmltype = "vecmathlib::realvec<%s,%d>" % (basetype, size)
    vmlinttype = "%s::intvec_t" % vmltype
    vmlbooltype = "%s::boolvec_t" % vmltype
    funcargstr = ", ".join(map(lambda (n, arg):
                                   "%s x%d" % (mktype(arg, vectype), n),
                               zip(range(0, 100), args)))
    funcretstr = mktype(ret, vectype)
    decl("%s %s(%s)" % (funcretstr, prefixed(name), funcargstr))
    out("%s %s(%s)" % (funcretstr, prefixed(name), funcargstr))
    out("{")
    for (n, arg, vmlarg) in zip(range(0, 100), args, vmlargs):
        out("  %s y%d = bitcast<%s,%s >(x%d);" %
            (mkvmltype(vmlarg, vmltype), n,
             mktype(arg, vectype), mkvmltype(vmlarg, vmltype), n))
    callargstr = ", ".join(map(lambda (n, arg): "y%d" % n,
                               zip(range(0, 100), args)))
    callretstr = mkvmltype(vmlret, vmltype)
    name1 = name[:-1] if name.endswith("_") else name
    out("  %s r = vecmathlib::%s(%s);" % (callretstr, name1, callargstr))
    # We may need to convert from the VML type to the OpenCL type
    # before bitcasting. This may be a real conversion, e.g. bool to
    # int. This may also involve a change in size (e.g. long to int),
    # but only if the type is scalar. These conversions are applied
    # before bitcasting.
    # convfunc: conversion function to call
    # convtype: result type of conversion, also input to bitcast
    # bitcasttype: output of bitcast; may differ from function result
    #              if a size change is needed
    # TODO: Why is this here, and not e.g. near the signbit definition
    #       in the table above?
    if vmlret==ret:
        convfunc    = ""
        convtype    = callretstr
        bitcasttype = funcretstr
    else:
        if vmlret==VI and ret in (VJ,VK):
            convfunc    = ""
            convtype    = callretstr
        elif vmlret==VB and ret in (VJ,VK):
            if size==1:
                # for scalars, true==+1
                convfunc = "vecmathlib::convert_int"
            else:
                # for vectors, true==-1
                convfunc = "-vecmathlib::convert_int"
            convtype = vmlinttype
        else:
            raise "missing"
        if ret in (VJ,VK):
            bitcasttype = mktype(VI, vectype)
        else:
            raise "missing"
    out("  return bitcast<%s,%s>(%s(r));" % (convtype, bitcasttype, convfunc))
    out("}")

def output_vmlfunc_special(specialtype, func, vectype):
    (name, args, ret, vmlargs, vmlret) = func
    if specialtype=="builtin":
        specialname = "builtin"
    elif specialtype=="pseudo":
        specialname = "libm"
    elif specialtype=="test":
        specialname = "vecmathlib"
    else:
        specialname = "???"
    out("// Implement %s by calling %s" % (name, specialname))
    (basetype, size) = re.match("([A-Za-z]+)([0-9]*)", vectype).groups()
    size = 1 if size=="" else int(size)
    othertype = "vecmathlib::real%svec<%s,%d>" % (specialtype, basetype, size)
    otherinttype = "%s::intvec_t" % othertype
    funcargstr = ", ".join(map(lambda (n, arg):
                                   "%s x%d" % (mktype(arg, vectype), n),
                               zip(range(0, 100), args)))
    funcretstr = mktype(ret, vectype)
    decl("%s %s(%s)" % (funcretstr, prefixed(name), funcargstr))
    out("%s %s(%s)" % (funcretstr, prefixed(name), funcargstr))
    out("{")
    for (n, arg) in zip(range(0, 100), args):
        out("  %s y%d = x%d;" % (mkvmltype(arg, othertype), n, n))
    callargstr = ", ".join(map(lambda (n, arg): "y%d" % n,
                               zip(range(0, 100), args)))
    # callretstr = othertype if ret==VF else otherinttype
    callretstr = mkvmltype(vmlret, othertype)
    name1 = name[:-1] if name.endswith("_") else name
    out("  %s r = %s(%s);" % (callretstr, name1, callargstr))
    # We may need to convert from the VML type to the OpenCL type
    # before bitcasting. This may be a real conversion, e.g. bool to
    # int. This may also involve a change in size (e.g. long to int),
    # but only if the type is scalar. These conversions are applied
    # before bitcasting.
    # convfunc: conversion function to call
    # convtype: result type of conversion, also input to bitcast
    # bitcasttype: output of bitcast; may differ from function result
    #              if a size change is needed
    # TODO: Why is this here, and not e.g. near the signbit definition
    #       in the table above?
    if vmlret==ret:
        convfunc = ""
    else:
        if vmlret==VI and ret in (VJ,VK):
            convfunc = ""
        elif vmlret==VB and ret in (VJ,VK):
            if size==1:
                # for scalars, true==+1
                convfunc = "vecmathlib::convert_int"
            else:
                # for vectors, true==-1
                convfunc = "-vecmathlib::convert_int"
        else:
            raise "missing"
    out("  return %s(r)[0];" % convfunc)
    out("}")

def output_vmlfunc_builtin(func, vectype):
    output_vmlfunc_special("builtin", func, vectype)

def output_vmlfunc_libm(func, vectype):
    output_vmlfunc_special("pseudo", func, vectype)

def output_vmlfunc_upcast(func, vectype):
    (name, args, ret, vmlargs, vmlret) = func
    out("// Implement %s by using a larger vector size" % name)
    (basetype, size) = re.match("([A-Za-z]+)([0-9]*)", vectype).groups()
    size = 1 if size=="" else int(size)
    size2 = 4 if size==3 else size*2 # next power of 2
    size2 = "" if size2==1 else str(size2)
    if size==1: raise "can't upcast scalars"
    othertype = "%s%s" % (basetype, size2)
    declargstr = ", ".join(map(lambda (n, arg): "%s" % mktype(arg, othertype),
                               zip(range(0, 100), args)))
    out("%s %s(%s);" % (mktype(ret, othertype), prefixed(name), declargstr))
    funcargstr = ", ".join(map(lambda (n, arg):
                                   "%s x%d" % (mktype(arg, vectype), n),
                               zip(range(0, 100), args)))
    decl("%s %s(%s)" % (mktype(ret, vectype), prefixed(name), funcargstr))
    out("%s %s(%s)" % (mktype(ret, vectype), prefixed(name), funcargstr))
    out("{")
    for (n, arg) in zip(range(0, 100), args):
        out("  %s y%d = bitcast<%s,%s>(x%d);" %
            (mktype(arg, othertype), n,
             mktype(arg, vectype), mktype(arg, othertype), n))
    callargstr = ", ".join(map(lambda (n, arg): "y%d" % n,
                               zip(range(0, 100), args)))
    out("  %s r = %s(%s);" %
        (mktype(ret, othertype), prefixed(name), callargstr))
    out("  return bitcast<%s,%s>(r);" %
        (mktype(ret, othertype), mktype(ret, vectype)))
    out("}")

def output_vmlfunc_split(func, vectype):
    (name, args, ret, vmlargs, vmlret) = func
    out("// Implement %s by splitting into a smaller vector size" % name)
    (basetype, size) = re.match("([A-Za-z]+)([0-9]*)", vectype).groups()
    size = 1 if size=="" else int(size)
    size2 = (size+1) / 2        # divide by 2, rounding up
    size2 = "" if size2==1 else str(size2)
    othertype = "%s%s" % (basetype, size2)
    declargstr = ", ".join(map(lambda (n, arg): "%s" % mktype(arg, othertype),
                               zip(range(0, 100), args)))
    out("%s %s(%s);" % (mktype(ret, othertype), prefixed(name), declargstr))
    funcargstr = ", ".join(map(lambda (n, arg):
                                   "%s x%d" % (mktype(arg, vectype), n),
                               zip(range(0, 100), args)))
    decl("%s %s(%s)" % (mktype(ret, vectype), prefixed(name), funcargstr))
    out("%s %s(%s)" % (mktype(ret, vectype), prefixed(name), funcargstr))
    out("{")
    if ret in (SF, SK):
        split_ret = SF
    elif ret in (VI, VJ, VK):
        split_ret = VI
    elif ret in (VF):
        split_ret = VF
    else:
        raise "missing"
    for (n, arg) in zip(range(0, 100), args):
        out("  pair_%s y%d = bitcast<%s,pair_%s>(x%d);" %
            (mktype(arg, othertype), n,
             mktype(arg, vectype), mktype(arg, othertype), n))
    out("  pair_%s r;" % mktype(split_ret, othertype))
    # in OpenCL: for scalars, true==+1, but for vectors, true==-1
    conv = ""
    if vmlret==VB:
        if ret in (VJ,VK):
            if size2=="":
                conv = "-"
        else:
            raise "missing"
    for suffix in ("lo", "hi"):
        callargstr = ", ".join(map(lambda (n, arg): "y%d.%s" % (n, suffix),
                                   zip(range(0, 100), args)))
        out("  r.%s = %s%s(%s);" % (suffix, conv, prefixed(name), callargstr))
    out("  pocl_static_assert(sizeof(pair_%s) == sizeof(%s));" %
        (mktype(split_ret, othertype), mktype(ret, vectype)))
    out("  return bitcast<pair_%s,%s>(r);" %
        (mktype(split_ret, othertype), mktype(ret, vectype)))
    out("}")



def output_directfunc_direct(func, vectype):
    (name, args, ret, impl) = func
    out("// Implement %s directly" % name)
    (space, basetype, sizename) = (
        re.match("(global|local|private)?(float|double)([0-9]*)", vectype).
        groups())
    size = 1 if sizename=="" else int(sizename)
    funcargstr = ", ".join(map(lambda (n, arg):
                                   "%s x%d" % (mktype(arg, vectype), n),
                               zip(range(0, 100), args)))
    funcretstr = mktype(ret, vectype)
    decl("%s %s(%s)" % (funcretstr, prefixed(name), funcargstr))
    out("__attribute__((__overloadable__))");
    out("%s %s(%s)" % (funcretstr, prefixed(name), funcargstr))
    out("{")
    out("  typedef %s kscalar_t;" % mktype(SK, vectype))
    out("  typedef %s scalar_t;" % mktype(SF, vectype))
    out("  typedef %s ivector_t;" % mktype(VI, vectype))
    out("  typedef %s jvector_t;" % mktype(VJ, vectype))
    out("  typedef %s kvector_t;" % mktype(VK, vectype))
    out("  typedef %s vector_t;" % mktype(VF, vectype))
    out("#define convert_ivector_t convert_%s" % mktype(VI, vectype))
    out("#define convert_jvector_t convert_%s" % mktype(VJ, vectype))
    out("#define convert_kvector_t convert_%s" % mktype(VK, vectype))
    out("#define convert_vector_t convert_%s" % mktype(VF, vectype))
    out("  return %s;" % impl)
    out("#undef convert_ivector_t")
    out("#undef convert_jvector_t")
    out("#undef convert_kvector_t")
    out("#undef convert_vector_t")
    out("}")



def output_vmlfunc(func):
    (name, args, ret, vmlargs, vmlret) = func
    is_first_open = out_open("%s.cc" % name)
    if is_first_open:
        out("// Note: This file has been automatically generated. "
            "Do not modify.")
        out("")
        out("#include \"pocl-compat.h\"")
        out("")
    else:
        out("")
        out("")
        out("")
    decl("")
    decl("// %s: %s -> %s" % (name, args, ret))
    decl("#undef %s" % name)
    if prefixed(name) != name:
        decl("#define %s %s" % (name, prefixed(name)))
    out("// %s: %s -> %s" % (name, args, ret))
    for basetype in ["float", "double"]:
        if basetype=="double":
            out("")
            out("#ifdef cl_khr_fp64")
        for size in [1, 2, 3, 4, 8, 16]:
            # Ignore this prototype for size==1 if there are any
            # scalar arguments; this prevents duplicate definitions
            if size==1 and any(map(lambda arg: arg in (SI, SK, SF), args)):
                continue
            sizename = '' if size==1 else str(size)
            vectype = basetype + sizename
            # always use vecmathlib if available
            out("")
            out("// %s: VF=%s" % (name, vectype))
            out("#if defined VECMATHLIB_HAVE_VEC_%s_%d && "
                "! defined POCL_VECMATHLIB_BUILTIN" %
                (basetype.upper(), size))
            output_vmlfunc_vml(func, vectype)
            if size==1:
                # a scalar type: use libm
                out("#elif ! defined POCL_VECMATHLIB_BUILTIN")
                output_vmlfunc_libm(func, vectype)
                out("#else")
                output_vmlfunc_builtin(func, vectype)
            else:
                # a vector type: try upcasting to next power of 2
                sizes = [4, 8, 16]
                sizes = [s for s in sizes if s>size]
                condstr = ""
                for s in sizes:
                    if condstr != "":
                        condstr = "%s || " % condstr
                    condstr = ("%sdefined VECMATHLIB_HAVE_VEC_%s_%d" %
                               (condstr, basetype.upper(), s))
                if condstr != "":
                    out("#elif (%s) && ! defined POCL_VECMATHLIB_BUILTIN " %
                        condstr)
                    output_vmlfunc_upcast(func, vectype)
                # a vector type: split into smaller vector type
                out("#else")
                output_vmlfunc_split(func, vectype)
            out("#endif")
        if basetype=="double":
            out("")
            out("#endif // #ifdef cl_khr_fp64")
    out_close()



def output_directfunc(func):
    (name, args, ret, impl) = func
    is_first_open = out_open("%s.cl" % name)
    if is_first_open:
        out("// Note: This file has been automatically generated. "
            "Do not modify.")
        out("")
        out("// Needed for fract()")
        out("#define POCL_FRACT_MIN   0x1.fffffffffffffp-1")
        out("#define POCL_FRACT_MIN_F 0x1.fffffep-1f")
        out("")
        out("// If double precision is not supported, then define")
        out("// single-precision (dummy) values to avoid compiler warnings")
        out("// for double precision values")
        out("#ifndef khr_fp64")
        out("#  undef M_PI")
        out("#  define M_PI M_PI_F")
        out("#  undef M_PI_2")
        out("#  define M_PI_2 M_PI_2_F")
        out("#  undef LONG_MAX")
        out("#  define LONG_MAX INT_MAX")
        out("#  undef LONG_MIN")
        out("#  define LONG_MIN INT_MIN")
        out("#  undef POCL_FRACT_MIN")
        out("#  define POCL_FRACT_MIN POCL_FRACT_MIN_F")
        out("#endif")
        out("")
    else:
        out("")
        out("")
        out("")
    decl("")
    decl("// %s: %s -> %s" % (name, args, ret))
    decl("#undef %s" % name)
    if prefixed(name) != name:
        decl("#define %s %s" % (name, prefixed(name)))
    out("// %s: %s -> %s" % (name, args, ret))
    if any(map(lambda arg: arg in (PVK, PVF), args)):
        spaces = ["global", "local", "private"]
    else:
        spaces = [""]
    for basetype in ["float", "double"]:
        if ((name.startswith("half_") or name.startswith("native_")) and
            basetype=="double"):
            continue
        if basetype=="double":
            out("")
            out("#ifdef cl_khr_fp64")
        for size in [1, 2, 3, 4, 8, 16]:
            # Ignore this prototype for size==1 if there are any
            # scalar arguments; this prevents duplicate definitions
            if size==1 and any(map(lambda arg: arg in (SI, SK, SF), args)):
                continue
            sizename = '' if size==1 else str(size)
            for space in spaces:
                vectype = space + basetype + sizename
                # always use vecmathlib if available
                out("")
                out("// %s: VF=%s" % (name, vectype))
                output_directfunc_direct(func, vectype)
        if basetype=="double":
            out("")
            out("#endif // #ifdef cl_khr_fp64")
    out_close()



decl_open("kernel-vecmathlib.h")
decl("// Note: This file has been automatically generated. Do not modify.")
decl("#ifndef KERNEL_VECMATHLIB_H")
decl("#define KERNEL_VECMATHLIB_H 1")
map(output_vmlfunc, vmlfuncs)
map(output_directfunc, directfuncs)
decl("")
decl("#endif // #ifndef KERNEL_VECMATHLIB_H")
decl_close()
print
