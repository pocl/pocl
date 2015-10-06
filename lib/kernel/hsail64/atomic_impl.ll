target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:64:64-p8:32:32-p9:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"
target triple = "hsail64"

; This file is originally from libclc

; *************************************************************************

define i32 @__clc_atomic_add_addr1(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile add i32 addrspace(1)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_add_addr3(i32 addrspace(2)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile add i32 addrspace(2)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_and_addr1(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile and i32 addrspace(1)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_and_addr3(i32 addrspace(2)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile and i32 addrspace(2)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_cmpxchg_addr1(i32 addrspace(1)* nocapture %ptr, i32 %compare, i32 %value) nounwind alwaysinline {
entry:
  %0 = cmpxchg volatile i32 addrspace(1)* %ptr, i32 %compare, i32 %value seq_cst seq_cst
  %1 = extractvalue { i32, i1 } %0, 0
  ret i32 %1
}

define i32 @__clc_atomic_cmpxchg_addr3(i32 addrspace(2)* nocapture %ptr, i32 %compare, i32 %value) nounwind alwaysinline {
entry:
  %0 = cmpxchg volatile i32 addrspace(2)* %ptr, i32 %compare, i32 %value seq_cst seq_cst
  %1 = extractvalue { i32, i1 } %0, 0
  ret i32 %1
}

define i32 @__clc_atomic_max_addr1(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile max i32 addrspace(1)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_max_addr3(i32 addrspace(2)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile max i32 addrspace(2)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_min_addr1(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile min i32 addrspace(1)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_min_addr3(i32 addrspace(2)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile min i32 addrspace(2)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_or_addr1(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile or i32 addrspace(1)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_or_addr3(i32 addrspace(2)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile or i32 addrspace(2)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_umax_addr1(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile umax i32 addrspace(1)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_umax_addr3(i32 addrspace(2)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile umax i32 addrspace(2)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_umin_addr1(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile umin i32 addrspace(1)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_umin_addr3(i32 addrspace(2)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile umin i32 addrspace(2)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_sub_addr1(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile sub i32 addrspace(1)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_sub_addr3(i32 addrspace(2)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile sub i32 addrspace(2)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_xchg_addr1(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile xchg i32 addrspace(1)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_xchg_addr3(i32 addrspace(2)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile xchg i32 addrspace(2)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_xor_addr1(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile xor i32 addrspace(1)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_xor_addr3(i32 addrspace(2)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile xor i32 addrspace(2)* %ptr, i32 %value seq_cst
  ret i32 %0
}

; *************************************************************************

define i64 @__clc_atom_add_addr1(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile add i64 addrspace(1)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc_atom_add_addr3(i64 addrspace(2)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile add i64 addrspace(2)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc_atom_and_addr1(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile and i64 addrspace(1)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc_atom_and_addr3(i64 addrspace(2)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile and i64 addrspace(2)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc_atom_cmpxchg_addr1(i64 addrspace(1)* nocapture %ptr, i64 %compare, i64 %value) nounwind alwaysinline {
entry:
  %0 = cmpxchg volatile i64 addrspace(1)* %ptr, i64 %compare, i64 %value seq_cst seq_cst
  %1 = extractvalue { i64, i1 } %0, 0
  ret i64 %1
}

define i64 @__clc_atom_cmpxchg_addr3(i64 addrspace(2)* nocapture %ptr, i64 %compare, i64 %value) nounwind alwaysinline {
entry:
  %0 = cmpxchg volatile i64 addrspace(2)* %ptr, i64 %compare, i64 %value seq_cst seq_cst
  %1 = extractvalue { i64, i1 } %0, 0
  ret i64 %1
}

define i64 @__clc_atom_max_addr1(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile max i64 addrspace(1)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc_atom_max_addr3(i64 addrspace(2)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile max i64 addrspace(2)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc_atom_min_addr1(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile min i64 addrspace(1)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc_atom_min_addr3(i64 addrspace(2)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile min i64 addrspace(2)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc_atom_or_addr1(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile or i64 addrspace(1)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc_atom_or_addr3(i64 addrspace(2)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile or i64 addrspace(2)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc_atom_umax_addr1(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile umax i64 addrspace(1)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc_atom_umax_addr3(i64 addrspace(2)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile umax i64 addrspace(2)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc_atom_umin_addr1(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile umin i64 addrspace(1)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc_atom_umin_addr3(i64 addrspace(2)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile umin i64 addrspace(2)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc_atom_sub_addr1(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile sub i64 addrspace(1)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc_atom_sub_addr3(i64 addrspace(2)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile sub i64 addrspace(2)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc_atom_xchg_addr1(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile xchg i64 addrspace(1)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc_atom_xchg_addr3(i64 addrspace(2)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile xchg i64 addrspace(2)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc_atom_xor_addr1(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile xor i64 addrspace(1)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc_atom_xor_addr3(i64 addrspace(2)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile xor i64 addrspace(2)* %ptr, i64 %value seq_cst
  ret i64 %0
}
