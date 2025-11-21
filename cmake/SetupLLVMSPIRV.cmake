
if(LLVM_ALL_TARGETS MATCHES "SPIRV")
  set(LLVM_HAS_SPIRV_TARGET 1 CACHE BOOL "LLVM SPIRV target")
else()
  set(LLVM_HAS_SPIRV_TARGET 0 CACHE BOOL "LLVM SPIRV target")
endif()

# check the binary version
if(HOST_LLVM_SPIRV AND (NOT CMAKE_CROSSCOMPILING))
  execute_process(
      COMMAND "${HOST_LLVM_SPIRV}" "--version"
      OUTPUT_VARIABLE LLVM_SPIRV_VERSION_VALUE
      RESULT_VARIABLE LLVM_SPIRV_VERSION_RETVAL
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(${LLVM_SPIRV_VERSION_RETVAL} EQUAL 0)
      string(REGEX MATCH "LLVM version ([0-9]*)" LLVM_SPIRV_VERSION_MATCH ${LLVM_SPIRV_VERSION_VALUE})
      if(NOT ${CMAKE_MATCH_1} EQUAL ${LLVM_VERSION_MAJOR})
        unset(LLVM_SPIRV CACHE)
      endif()
  else()
    unset(LLVM_SPIRV CACHE)
    unset(LLVM_SPIRV)
  endif()
endif()

if(TARGET_LLVM_SPIRV)
  message(STATUS "Using Target llvm-spirv: ${TARGET_LLVM_SPIRV}")
else()
  message(STATUS "Did NOT find usable llvm-spirv!")
endif()

if(TARGET_SPIRV_LINK)
  message(STATUS "Found Target spirv-link: ${TARGET_SPIRV_LINK}")
else()
  message(STATUS "Did NOT find spirv-link!")
endif()

# Search for the LLVMSPIRV library. First try to find it in the LLVM's include directories.
find_path(LLVM_SPIRV_INCLUDEDIR "LLVMSPIRVLib.h"
  PATHS "${LLVM_INCLUDE_DIRS}" NO_DEFAULT_PATH PATH_SUFFIXES "LLVMSPIRVLib")
find_library(LLVM_SPIRV_LIB "LLVMSPIRVLib" PATHS "${LLVM_LIBDIR}" NO_DEFAULT_PATH)

# fallback to searching for unversioned translator library & headers;
# however if we found a version-matching llvm-spirv executable, prefer that
# at least on Ubuntu it's possible to have mismatching libLLVMSPIRV
if(NOT TARGET_LLVM_SPIRV)
  find_path(LLVM_SPIRV_INCLUDEDIR "LLVMSPIRVLib.h" PATH_SUFFIXES "LLVMSPIRVLib")
  find_library(LLVM_SPIRV_LIB "LLVMSPIRVLib" PATHS "${LLVM_LIBDIR}")
endif()

if(LLVM_SPIRV_INCLUDEDIR AND LLVM_SPIRV_LIB AND (NOT DEFINED HAVE_LLVM_SPIRV_LIB))

  if(UNIX)
    set(LINK_OPTS LINK_OPTIONS "-Wl,-rpath,${LLVM_LIBDIR}")
  else()
    unset(LINK_OPTS)
  endif()

  if(NOT LLVM_SPIRV_LIB_MAXVER)
    file(READ "${LLVM_SPIRV_INCLUDEDIR}/LLVMSPIRVOpts.h" LLVMOPTS_CONTENT)
    string(REGEX MATCH "MaximumVersion[ ]+=[ ]+SPIRV_1_([0-9]+)" MAXVER_MATCH "${LLVMOPTS_CONTENT}")
    if(MAXVER_MATCH)
      math(EXPR MAXVER_INT "65536 + (${CMAKE_MATCH_1} * 256)")
      message(STATUS "found Maximum SPIRV version supported by libLLVMSPIRV: ${MAXVER_INT}")
      set(LLVM_SPIRV_LIB_MAXVER ${MAXVER_INT} CACHE STRING "maximum SPIR-V version supported by libLLVMSPIRV")
    else()
      message(STATUS "failed to find Maximum SPIRV version supported by libLLVMSPIRV")
      set(LLVM_SPIRV_LIB_MAXVER 0 CACHE STRING "maximum SPIR-V version supported by libLLVMSPIRV")
    endif()
  endif()

  if(LLVM_SPIRV_LIB_MAXVER)
    set(HAVE_LLVM_SPIRV_LIB ON CACHE BOOL "have libLLVMSPIRV")
  else()
    set(HAVE_LLVM_SPIRV_LIB OFF CACHE BOOL "have libLLVMSPIRV")
  endif()
endif()

if(HAVE_LLVM_SPIRV_LIB)
  message(STATUS "LLVMSPIRV library found: ${LLVM_SPIRV_INCLUDEDIR} | ${LLVM_SPIRV_LIB}")
else()
  message(STATUS "LLVMSPIRV library NOT found: ${LLVM_SPIRV_INCLUDEDIR} | ${LLVM_SPIRV_LIB}")
endif()

set_expr(HAVE_SPIRV_LINK TARGET_SPIRV_LINK)
set_expr(HAVE_LLVM_SPIRV TARGET_LLVM_SPIRV)
set_expr(HAVE_LLVM_OPT TARGET_LLVM_OPT)

if(HAVE_SPIRV_LINK AND (NOT DEFINED SPIRV_LINK_HAS_USE_HIGHEST_VERSION)
  AND NOT CMAKE_CROSSCOMPILING)
  # TODO crosscompiling
  execute_process(
      COMMAND "${TARGET_SPIRV_LINK}" "--help"
      OUTPUT_VARIABLE SPIRV_LINK_OUTPUT_VALUE
      RESULT_VARIABLE SPIRV_LINK_RETVAL
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if("${SPIRV_LINK_OUTPUT_VALUE}" MATCHES "use-highest-version")
    set(SPIRV_LINK_HAS_USE_HIGHEST_VERSION 1 CACHE BOOL "spirv-link --use-highest-version")
  else()
    set(SPIRV_LINK_HAS_USE_HIGHEST_VERSION 0 CACHE BOOL "spirv-link --use-highest-version")
  endif()
endif()
