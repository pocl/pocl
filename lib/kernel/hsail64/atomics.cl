/* This file is from libclc, licensed:

Copyright (c) 2011-2014 by the contributors listed in CREDITS.TXT

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


/* These are defined in _kernel.h to be aliases of atomic_*,
 * which breaks the following macros, so undef them for now
 * (64bit atomics are all named "atom_" not "atomic_")
 */
#undef atom_add
#undef atom_sub
#undef atom_xchg
#undef atom_inc
#undef atom_dec
#undef atom_cmpxchg
#undef atom_min
#undef atom_max
#undef atom_and
#undef atom_or
#undef atom_xor

#define ATOMIC_FUNC_DEFINE(RET_SIGN, ARG_SIGN, TYPE, CL_FUNCTION, CLC_FUNCTION, CL_ADDRSPACE, LLVM_ADDRSPACE) \
_CL_OVERLOADABLE  RET_SIGN TYPE CL_FUNCTION (volatile CL_ADDRSPACE RET_SIGN TYPE *p, RET_SIGN TYPE val) { \
    return (RET_SIGN TYPE)__clc_##CLC_FUNCTION##_addr##LLVM_ADDRSPACE((volatile CL_ADDRSPACE ARG_SIGN TYPE*)p, (ARG_SIGN TYPE)val); \
}

/* For atomic functions that don't need different bitcode dependending on argument signedness */
#define ATOMIC_FUNC_SIGN(TYPE, FUNCTION, CL_ADDRSPACE, LLVM_ADDRSPACE) \
    signed TYPE __clc_##FUNCTION##_addr##LLVM_ADDRSPACE(volatile CL_ADDRSPACE signed TYPE*, signed TYPE); \
    ATOMIC_FUNC_DEFINE(signed, signed, TYPE, FUNCTION, FUNCTION, CL_ADDRSPACE, LLVM_ADDRSPACE) \
    ATOMIC_FUNC_DEFINE(unsigned, signed, TYPE, FUNCTION, FUNCTION, CL_ADDRSPACE, LLVM_ADDRSPACE)

#define ATOMIC_FUNC_ADDRSPACE(TYPE, FUNCTION) \
    ATOMIC_FUNC_SIGN(TYPE, FUNCTION, global, 1) \
    ATOMIC_FUNC_SIGN(TYPE, FUNCTION, local, 3)

#define ATOMIC_FUNC(FUNCTION) \
    ATOMIC_FUNC_ADDRSPACE(int, atomic ## FUNCTION) \
    ATOMIC_FUNC_ADDRSPACE(long, atom ## FUNCTION)

#define ATOMIC_FUNC_DEFINE_3_ARG(RET_SIGN, ARG_SIGN, TYPE, CL_FUNCTION, CLC_FUNCTION, CL_ADDRSPACE, LLVM_ADDRSPACE) \
_CL_OVERLOADABLE  RET_SIGN TYPE CL_FUNCTION (volatile CL_ADDRSPACE RET_SIGN TYPE *p, RET_SIGN TYPE cmp, RET_SIGN TYPE val) { \
	return (RET_SIGN TYPE)__clc_##CLC_FUNCTION##_addr##LLVM_ADDRSPACE((volatile CL_ADDRSPACE ARG_SIGN TYPE*)p, (ARG_SIGN TYPE)cmp, (ARG_SIGN TYPE)val); \
}

/* For atomic functions that don't need different bitcode dependending on argument signedness */
#define ATOMIC_FUNC_SIGN_3_ARG(TYPE, FUNCTION, CL_ADDRSPACE, LLVM_ADDRSPACE) \
    signed TYPE  __clc_##FUNCTION##_addr##LLVM_ADDRSPACE(volatile CL_ADDRSPACE signed TYPE*, signed TYPE, signed TYPE); \
    ATOMIC_FUNC_DEFINE_3_ARG(signed, signed, TYPE, FUNCTION, FUNCTION, CL_ADDRSPACE, LLVM_ADDRSPACE) \
    ATOMIC_FUNC_DEFINE_3_ARG(unsigned, signed, TYPE, FUNCTION, FUNCTION, CL_ADDRSPACE, LLVM_ADDRSPACE)

#define ATOMIC_FUNC_ADDRSPACE_3_ARG(TYPE, FUNCTION) \
    ATOMIC_FUNC_SIGN_3_ARG(TYPE, FUNCTION, global, 1) \
    ATOMIC_FUNC_SIGN_3_ARG(TYPE, FUNCTION, local, 3)

#define ATOMIC_FUNC_3_ARG(FUNCTION) \
    ATOMIC_FUNC_ADDRSPACE_3_ARG(int, atomic ## FUNCTION) \
    ATOMIC_FUNC_ADDRSPACE_3_ARG(long, atom ## FUNCTION)

ATOMIC_FUNC(_add)
ATOMIC_FUNC(_and)
ATOMIC_FUNC(_or)
ATOMIC_FUNC(_sub)
ATOMIC_FUNC(_xchg)
ATOMIC_FUNC(_xor)
ATOMIC_FUNC_3_ARG(_cmpxchg)

// 32bit

signed int __clc_atomic_max_addr1(volatile global signed int*, signed int);
signed int __clc_atomic_max_addr3(volatile local signed int*, signed int);
uint __clc_atomic_umax_addr1(volatile global uint*, uint);
uint __clc_atomic_umax_addr3(volatile local uint*, uint);

ATOMIC_FUNC_DEFINE(signed, signed, int, atomic_max, atomic_max, global, 1)
ATOMIC_FUNC_DEFINE(signed, signed, int, atomic_max, atomic_max, local, 3)
ATOMIC_FUNC_DEFINE(unsigned, unsigned, int, atomic_max, atomic_umax, global, 1)
ATOMIC_FUNC_DEFINE(unsigned, unsigned, int, atomic_max, atomic_umax, local, 3)

signed int __clc_atomic_min_addr1(volatile global signed int*, signed int);
signed int __clc_atomic_min_addr3(volatile local signed int*, signed int);
uint __clc_atomic_umin_addr1(volatile global uint*, uint);
uint __clc_atomic_umin_addr3(volatile local uint*, uint);

ATOMIC_FUNC_DEFINE(signed, signed, int, atomic_min, atomic_min, global, 1)
ATOMIC_FUNC_DEFINE(signed, signed, int, atomic_min, atomic_min, local, 3)
ATOMIC_FUNC_DEFINE(unsigned, unsigned, int, atomic_min, atomic_umin, global, 1)
ATOMIC_FUNC_DEFINE(unsigned, unsigned, int, atomic_min, atomic_umin, local, 3)

// 64bit

signed long __clc_atom_max_addr1(volatile global signed long*, signed long);
signed long __clc_atom_max_addr3(volatile local signed long*, signed long);
ulong __clc_atom_umax_addr1(volatile global ulong*, ulong);
ulong __clc_atom_umax_addr3(volatile local ulong*, ulong);

ATOMIC_FUNC_DEFINE(signed, signed, long, atom_max, atom_max, global, 1)
ATOMIC_FUNC_DEFINE(signed, signed, long, atom_max, atom_max, local, 3)
ATOMIC_FUNC_DEFINE(unsigned, unsigned, long, atom_max, atom_umax, global, 1)
ATOMIC_FUNC_DEFINE(unsigned, unsigned, long, atom_max, atom_umax, local, 3)

signed long __clc_atom_min_addr1(volatile global signed long*, signed long);
signed long __clc_atom_min_addr3(volatile local signed long*, signed long);
ulong __clc_atom_umin_addr1(volatile global ulong*, ulong);
ulong __clc_atom_umin_addr3(volatile local ulong*, ulong);

ATOMIC_FUNC_DEFINE(signed, signed, long, atom_min, atom_min, global, 1)
ATOMIC_FUNC_DEFINE(signed, signed, long, atom_min, atom_min, local, 3)
ATOMIC_FUNC_DEFINE(unsigned, unsigned, long, atom_min, atom_umin, global, 1)
ATOMIC_FUNC_DEFINE(unsigned, unsigned, long, atom_min, atom_umin, local, 3)
