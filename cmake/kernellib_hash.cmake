file(SHA1 "${KERNELBC}" S1)
file(SHA1 "${INCLUDEDIR}/_kernel.h" S2)
file(SHA1 "${INCLUDEDIR}/_kernel_c.h" S3)
file(SHA1 "${INCLUDEDIR}/pocl_types.h" S4)

file(WRITE "${OUTPUT}" "#define POCL_KERNELLIB_SHA1 \"${S1}${S2}${S3}${S4}\"")
