
# TODO: We need to set both target-triple and cpu-type when
# building, since the ABI depends on both. We can either add flags
# to all the scripts, or set the respective flags here in
# *_CLANG_FLAGS and *_LLC_FLAGS. Note that clang and llc use
# different option names to set these. Note that clang calls the
# triple "target" and the cpu "architecture", which is different
# from llc.

# Normalise the triple. Otherwise, clang normalises it when
# passing it to llc, which is then different from the triple we
# pass to llc. This would lead to inconsistent bytecode files,
# depending on whether they are generated via clang or directly
# via llc.

if(NOT DEFINED LLC_TRIPLE)
  if(CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "LLC_TRIPLE must be provided when cross-compiling!")
  endif()
  message(STATUS "Find out LLC target triple (for host ${LLVM_HOST_TARGET})")
  set(_EMPTY_C_FILE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/tripletfind.c")
  file(WRITE "${_EMPTY_C_FILE}" "")

  execute_process(COMMAND ${HOST_CLANG} "--target=${LLVM_HOST_TARGET}" -x c ${_EMPTY_C_FILE} -S -emit-llvm -o - RESULT_VARIABLE RES_VAR OUTPUT_VARIABLE OUTPUT_VAR)
  if(RES_VAR)
    message(FATAL_ERROR "Error ${RES_VAR} while determining target triple")
  endif()
  if(OUTPUT_VAR MATCHES "target triple = \"([^\"]+)")
    string(STRIP "${CMAKE_MATCH_1}" LLC_TRIPLE)
  else()
    message(FATAL_ERROR "Could not find target triple in llvm output")
  endif()

  # TODO the armv7hl normalize
  string(REPLACE "armv7l-" "armv7-" LLC_TRIPLE "${LLC_TRIPLE}")

  set(LLC_TRIPLE "${LLC_TRIPLE}" CACHE INTERNAL "LLC_TRIPLE")
endif()

# FIXME: The cpu name printed by llc --version is the same cpu that will be
# targeted if you pass -mcpu=native to llc, so we could replace this auto-detection
# with just: set(LLC_HOST_CPU "native"), however, we can't do this at the moment
# because of the work-around for arm1176jz-s.
if(NOT DEFINED LLC_HOST_CPU_AUTO)
  if(CMAKE_CROSSCOMPILING)
    # user must provide target's LLC_HOST_CPU
    set(LLC_HOST_CPU_AUTO "unknown")
  else()
    message(STATUS "Find out LLC host CPU with ${LLVM_LLC}")
    execute_process(COMMAND ${HOST_LLVM_LLC} "--version" RESULT_VARIABLE RES_VAR OUTPUT_VARIABLE OUTPUT_VAR)
    if(RES_VAR)
      message(FATAL_ERROR "Error ${RES_VAR} while determining LLC host CPU")
    endif()

    if(OUTPUT_VAR MATCHES "Host CPU: ([^ ]*)")
      # sigh... STRING(STRIP is to workaround regexp bug in cmake
      string(STRIP "${CMAKE_MATCH_1}" LLC_HOST_CPU_AUTO)
    else()
      message(FATAL_ERROR "Couldnt determine host CPU from llc output")
    endif()

    #TODO better
    if(CMAKE_LIBRARY_ARCHITECTURE MATCHES "gnueabihf" AND LLC_HOST_CPU_AUTO MATCHES "arm1176jz-s")
      set(LLC_HOST_CPU_AUTO "arm1176jzf-s")
    endif()
  endif()
endif()

if((LLC_HOST_CPU_AUTO MATCHES "unknown") AND (NOT LLC_HOST_CPU))
  message(FATAL_ERROR "LLVM could not recognize your CPU model automatically. Please run CMake with -DLLC_HOST_CPU=<cpu> (you can find valid names with: llc -mcpu=help)")
else()
  set(LLC_HOST_CPU_AUTO "${LLC_HOST_CPU_AUTO}" CACHE INTERNAL "Autodetected CPU")
endif()

if(DEFINED LLC_HOST_CPU)
  if(NOT LLC_HOST_CPU STREQUAL LLC_HOST_CPU_AUTO)
    message(STATUS "Autodetected CPU ${LLC_HOST_CPU_AUTO} overridden by user to ${LLC_HOST_CPU}")
  endif()
  set(SELECTED_HOST_CPU "${LLC_HOST_CPU}")
  set(HOST_CPU_FORCED 1 CACHE INTERNAL "CPU is forced by user")
else()
  set(SELECTED_HOST_CPU "${LLC_HOST_CPU_AUTO}")
  set(HOST_CPU_FORCED 0 CACHE INTERNAL "CPU is forced by user")
endif()

####################################################################

# Some architectures have -march and -mcpu reversed
if(NOT DEFINED CLANG_MARCH_FLAG)
  message(STATUS "Checking clang -march vs. -mcpu flag")
  custom_try_compile_clang_silent("" "return 0;" RES --target=${LLC_TRIPLE} -march=${SELECTED_HOST_CPU})
  if(NOT RES)
    set(CLANG_MARCH_FLAG "-march=")
  else()
    custom_try_compile_clang_silent("" "return 0;" RES --target=${LLC_TRIPLE} -mcpu=${SELECTED_HOST_CPU})
    if(NOT RES)
      set(CLANG_MARCH_FLAG "-mcpu=")
    else()
      message(FATAL_ERROR "Could not determine whether to use -march or -mcpu with clang")
    endif()
  endif()
  message(STATUS "  Using ${CLANG_MARCH_FLAG}")

  set(CLANG_MARCH_FLAG ${CLANG_MARCH_FLAG} CACHE INTERNAL "Clang option used to specify the target cpu")
endif()

####################################################################

# Clang documentation on Language Extensions:
# __fp16 is supported on every target, as it is purely a storage format
# _Float16 is currently only supported on the following targets... SPIR, x86
# Limitations:
#     The _Float16 type requires SSE2 feature and above due to the instruction
#        limitations. When using it on i386 targets, you need to specify -msse2
#        explicitly.
#     For targets without F16C feature or above, please make sure:
#     Use GCC 12.0 and above if you are using libgcc.
#     If you are using compiler-rt, use the same version with the compiler.
#        Early versions provided FP16 builtins in a different ABI. A workaround is
#        to use a small code snippet to check the ABI if you cannot make sure of it.

if(NOT DEFINED CLANG_SUPPORTS_FLOAT16_ON_CPU)
  set(CLANG_SUPPORTS_FLOAT16_ON_CPU 0)
  message(STATUS "Checking Device-side (Clang/LLVM) support for _Float16 type")
    custom_try_compile_clang_silent("_Float16 callfp16(_Float16 a) { return a * 1.8f16; };" "_Float16 x=callfp16((_Float16)argc);"
    RESV --target=${LLC_TRIPLE} ${CLANG_MARCH_FLAG}${SELECTED_HOST_CPU})
  if(RESV EQUAL 0)
    message(STATUS "Clang supports _Float16 type on CPU")
    set(CLANG_SUPPORTS_FLOAT16_ON_CPU 1)
  else()
    message(STATUS "Clang doesn't support _Float16 type on CPU")
  endif()
endif()

####################################################################

# TODO we should check double support of the target somehow (excluding emulation),
# for now just provide an option
if(NOT DEFINED HOST_CPU_SUPPORTS_DOUBLE)
  if(X86)
    set(HOST_CPU_SUPPORTS_DOUBLE ON CACHE INTERNAL "FP64, always enabled on X86(-64)" FORCE)
  else()
    option(HOST_CPU_SUPPORTS_DOUBLE "Enable FP64 support for Host CPU device" ON)
  endif()
endif()
