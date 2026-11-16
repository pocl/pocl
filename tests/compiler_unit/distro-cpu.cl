// Check kernel and imported builtin target attributes.
// Regenerate distro-cpu.spirv with:
// clang -target spir64 -x cl -cl-std=CL1.2 -Xclang -finclude-default-header \
//   -O0 -emit-llvm -c distro-cpu.cl -o distro-cpu.bc
// llvm-spirv distro-cpu.bc -o distro-cpu.spirv
kernel void
distro_cpu (global float *out)
{
  size_t i = get_global_id (0);
  int exponent;
  out[i] = frexp (out[i], &exponent) + (float)exponent;
}
