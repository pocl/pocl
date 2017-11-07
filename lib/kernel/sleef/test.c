
/************************/
#if defined(PURE_C)

  #define CONFIG 1

#elif defined(VEC128)

  #ifdef __ARM_NEON
    #define CONFIG 1

  #elif defined(__AVX2__)
    #define CONFIG 1

  #elif defined(__SSE4_1__)
    #define CONFIG 4

  #elif defined(__SSE3__)
    #define CONFIG 3

  #elif defined(__SSE2__)
    #define CONFIG 2

  #else
    #error 128bit vectors unavailable
  #endif

#elif defined(VEC256)

  #if defined(__AVX2__)
    #define CONFIG 1

  #elif defined(__FMA4__)
    #define CONFIG 4

  #elif defined(__AVX__)
    #define CONFIG 1

  #else
    #error 256bit vectors unavailable
  #endif

#elif defined(VEC512)

  #ifdef __AVX512F__
    #define CONFIG 1
  #else
    #error 512bit vectors unavailable
  #endif

#else
#error Please specify valid vector size with -DVECxxx
#endif


int main() {
  return 0;
}
