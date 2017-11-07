/************************/

int main() {

#if defined(PURE_C)

  // glibc
  return 0;

#elif defined(VEC128)

  #if defined(__aarch64__)
    // __ARM_ARCH_ISA_A64
    // ARM64 should always have FMA
    return 0;
  #elif defined(__ARM_NEON)
    // TODO proper ARM detection
    #error ARM32 FMA detection not implemented
    return -1;
  #elif defined(__AVX2__) || defined(__FMA4__)
    return 0;
  #else
    #error FMA status unknown
    return -1;
  #endif

#elif defined(VEC256)

  #if defined(__AVX2__) || defined(__FMA4__)
    return 0;
  #else
    #error FMA status unknown
    return -1;
  #endif

#elif defined(VEC512)

  #if defined(__AVX512F__)
    return 0;
  #else
    #error FMA status unknown
    return -1;
  #endif

#else
  #error FMA status unknown
  return -1;
#endif
}
