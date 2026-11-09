target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

declare spir_func i64 @_Z13get_global_idj(i32)

define spir_kernel void @divrem(ptr addrspace(1) %x, ptr addrspace(1) %v,
                                ptr addrspace(1) %num, ptr addrspace(1) %den,
                                ptr addrspace(1) %out, ptr addrspace(1) %vout) {
  %i = call spir_func i64 @_Z13get_global_idj(i32 0)

  ; x / x, x % x
  %px = getelementptr i32, ptr addrspace(1) %x, i64 %i
  %xv = load i32, ptr addrspace(1) %px
  %xsd = sdiv i32 %xv, %xv
  %xud = udiv i32 %xv, %xv
  %xsr = srem i32 %xv, %xv
  %xur = urem i32 %xv, %xv

  ; n / d, n % d with runtime divisors (including 0 and INT_MIN / -1)
  %pn = getelementptr i32, ptr addrspace(1) %num, i64 %i
  %pd = getelementptr i32, ptr addrspace(1) %den, i64 %i
  %n = load i32, ptr addrspace(1) %pn
  %d = load i32, ptr addrspace(1) %pd
  %sd = sdiv i32 %n, %d
  %ud = udiv i32 %n, %d
  %sr = srem i32 %n, %d
  %ur = urem i32 %n, %d

  %o = mul i64 %i, 8
  %o0 = getelementptr i32, ptr addrspace(1) %out, i64 %o
  store i32 %xsd, ptr addrspace(1) %o0
  %o1 = getelementptr i32, ptr addrspace(1) %o0, i64 1
  store i32 %xud, ptr addrspace(1) %o1
  %o2 = getelementptr i32, ptr addrspace(1) %o0, i64 2
  store i32 %xsr, ptr addrspace(1) %o2
  %o3 = getelementptr i32, ptr addrspace(1) %o0, i64 3
  store i32 %xur, ptr addrspace(1) %o3
  %o4 = getelementptr i32, ptr addrspace(1) %o0, i64 4
  store i32 %sd, ptr addrspace(1) %o4
  %o5 = getelementptr i32, ptr addrspace(1) %o0, i64 5
  store i32 %ud, ptr addrspace(1) %o5
  %o6 = getelementptr i32, ptr addrspace(1) %o0, i64 6
  store i32 %sr, ptr addrspace(1) %o6
  %o7 = getelementptr i32, ptr addrspace(1) %o0, i64 7
  store i32 %ur, ptr addrspace(1) %o7

  ; v / v, v % v on vectors
  %pv = getelementptr <4 x i32>, ptr addrspace(1) %v, i64 %i
  %vv = load <4 x i32>, ptr addrspace(1) %pv
  %vsd = sdiv <4 x i32> %vv, %vv
  %vud = udiv <4 x i32> %vv, %vv
  %vsr = srem <4 x i32> %vv, %vv
  %vur = urem <4 x i32> %vv, %vv
  %vo = mul i64 %i, 4
  %vo0 = getelementptr <4 x i32>, ptr addrspace(1) %vout, i64 %vo
  store <4 x i32> %vsd, ptr addrspace(1) %vo0
  %vo1 = getelementptr <4 x i32>, ptr addrspace(1) %vo0, i64 1
  store <4 x i32> %vud, ptr addrspace(1) %vo1
  %vo2 = getelementptr <4 x i32>, ptr addrspace(1) %vo0, i64 2
  store <4 x i32> %vsr, ptr addrspace(1) %vo2
  %vo3 = getelementptr <4 x i32>, ptr addrspace(1) %vo0, i64 3
  store <4 x i32> %vur, ptr addrspace(1) %vo3
  ret void
}
