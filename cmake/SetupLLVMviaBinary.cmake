
# Set up LLVM variables using the llvm-config executable

if(NOT LLVM_CONFIG_BIN)
  message(FATAL_ERROR "llvm-config not found !")
endif()

if(STATIC_LLVM)
  set(LLVM_LIB_MODE --link-static)
else()
  set(LLVM_LIB_MODE --link-shared)
endif()

run_llvm_config(LLVM_PREFIX --prefix)
# on windows, llvm-config returs "C:\llvm_prefix/bin" mixed style paths,
# and cmake doesn't like the "\" - thinks its an escape char..
file(TO_CMAKE_PATH "${LLVM_PREFIX}" LLVM_PREFIX_CMAKE)

macro(replace_llvm_prefix_cmake VARIABLE_NAME)
  string(REPLACE "${LLVM_PREFIX}\\" "${LLVM_PREFIX_CMAKE}/" ${VARIABLE_NAME} "${${VARIABLE_NAME}}")
  string(REPLACE "${LLVM_PREFIX}" "${LLVM_PREFIX_CMAKE}" ${VARIABLE_NAME} "${${VARIABLE_NAME}}")
endmacro(replace_llvm_prefix_cmake)

run_llvm_config(LLVM_CFLAGS --cflags)
replace_llvm_prefix_cmake(LLVM_CFLAGS)
run_llvm_config(LLVM_CXXFLAGS --cxxflags)
replace_llvm_prefix_cmake(LLVM_CXXFLAGS)
run_llvm_config(LLVM_CPPFLAGS --cppflags)
replace_llvm_prefix_cmake(LLVM_CPPFLAGS)
run_llvm_config(LLVM_LDFLAGS --ldflags)
replace_llvm_prefix_cmake(LLVM_LDFLAGS)
run_llvm_config(LLVM_BINDIR --bindir)
replace_llvm_prefix_cmake(LLVM_BINDIR)
run_llvm_config(LLVM_LIBDIR --libdir)
replace_llvm_prefix_cmake(LLVM_LIBDIR)
run_llvm_config(LLVM_INCLUDE_DIRS --includedir)
replace_llvm_prefix_cmake(LLVM_INCLUDE_DIRS)
run_llvm_config(LLVM_CMAKEDIR --cmakedir)
replace_llvm_prefix_cmake(LLVM_CMAKEDIR)

run_llvm_config(LLVM_ALL_TARGETS --targets-built)
if (NOT DEFINED LLVM_HOST_TARGET)
  run_llvm_config(LLVM_HOST_TARGET --host-target)
endif()

run_llvm_config(LLVM_BUILD_MODE --build-mode)
run_llvm_config(LLVM_ASSERTS_BUILD --assertion-mode)

if(MSVC)
  string(REPLACE "-L${LLVM_LIBDIR}" "" LLVM_LDFLAGS "${LLVM_LDFLAGS}")
  string(STRIP "${LLVM_LDFLAGS}" LLVM_LDFLAGS)
  file(TO_CMAKE_PATH "${LLVM_LDFLAGS}" LLVM_LDFLAGS)
endif()

####################################################################

# In windows llvm-config reports --target=x86_64-pc-windows-msvc
# however this causes clang to use MicrosoftCXXMangler, which does not
# yet support mangling for extended vector types (with llvm 3.5)
# so for now hardcode LLVM_HOST_TARGET to be x86_64-pc with windows
# TODO is this still required ???
if(WIN32 AND (NOT MINGW))
  # Using the following target causes clang to invoke gcc for linking
  # instead of MSVC's link.exe.
  # TODO: lower LLVM version requirement until the above issue is hit.
  if (NOT MSVC OR LLVM_VERSION_MAJOR LESS 18)
    set(LLVM_HOST_TARGET "x86_64-pc")
  endif()
endif()

# A few work-arounds for llvm-config issues

# - pocl doesn't compile with '-pedantic'
#LLVM_CXX_FLAGS=$($LLVM_CONFIG --cxxflags | sed -e 's/ -pedantic / /g')
string(REPLACE " -pedantic" "" LLVM_CXXFLAGS "${LLVM_CXXFLAGS}")

#llvm-config clutters CXXFLAGS with a lot of -W<whatever> flags.
#(They are not needed - we want to use -Wall anyways)
#This is a problem if LLVM was built with a different compiler than we use here,
#and our compiler chokes on unrecognized command-line options.
string(REGEX REPLACE "-W[^ ]*" "" LLVM_CXXFLAGS "${LLVM_CXXFLAGS}")

# Ubuntu's llvm reports "arm-unknown-linux-gnueabihf" triple, then if one tries
# `clang --target=arm-unknown-linux-gnueabihf ...` it will produce armv6 code,
# even if one's running armv7;
# Here we replace the "arm" string with whatever's in CMAKE_HOST_SYSTEM_PROCESSOR
# which should be "armv6l" on rasp pi, or "armv7l" on my cubieboard, hopefully its
# more reasonable and reliable than llvm's own host flags
if(NOT CMAKE_CROSSCOMPILING)
  string(REPLACE "arm-" "${CMAKE_HOST_SYSTEM_PROCESSOR}-" LLVM_HOST_TARGET "${LLVM_HOST_TARGET}")
endif()

separate_arguments(LLVM_CXXFLAGS_LIST NATIVE_COMMAND "${LLVM_CXXFLAGS}")

####################################################################

# this needs to be done with LLVM_LIB_MODE because it affects the output
run_llvm_config(LLVM_SYSLIBS --system-libs ${LLVM_LIB_MODE})
string(STRIP "${LLVM_SYSLIBS}" LLVM_SYSLIBS)
string(REPLACE " " ";" LLVM_SYSLIBS "${LLVM_SYSLIBS}")

# TODO UNSOLVED
# Clang library depends on this system library.
if(MINGW)
  list(APPEND LLVM_SYSLIBS "-lversion")
elseif(MSVC)
  list(APPEND LLVM_SYSLIBS version.lib)
endif()

####################################################################

unset(LLVM_LIBS)
run_llvm_config(LLVM_LIBS --libs ${LLVM_LIB_MODE})
string(STRIP "${LLVM_LIBS}" LLVM_LIBS)
file(TO_CMAKE_PATH "${LLVM_LIBS}" LLVM_LIBS)
if(NOT LLVM_LIBS)
  message(FATAL_ERROR "llvm-config --libs did not return anything, perhaps wrong setting of STATIC_LLVM ?")
endif()
# Convert LLVM_LIBS from string -> list format to make handling them easier
separate_arguments(LLVM_LIBS)

# With Visual Studio llvm-config gives invalid list of static libs (libXXXX.a instead of XXXX.lib)
# we extract the pure names (LLVMLTO, LLVMMipsDesc etc) and let find_library do its job
# TODO extract names directly via llvm-config ?
foreach(LIBFLAG ${LLVM_LIBS})
  STRING(REGEX REPLACE "^-l(.*)$" "\\1" LIB_NAME ${LIBFLAG})
  list(APPEND LLVM_LIBNAMES "${LIB_NAME}")
endforeach()

set(LLVM_LINK_LIBRARIES)
set(LLVM_LINK_DIRS)
foreach(LIBNAME ${LLVM_LIBNAMES})
  if(EXISTS ${LIBNAME})
    set(L_LIBFILE_${LIBNAME} ${LIBNAME})
  else()
    find_library(L_LIBFILE_${LIBNAME} NAMES "${LIBNAME}" HINTS "${LLVM_LIBDIR}")
    if(NOT L_LIBFILE_${LIBNAME})
      message(FATAL_ERROR "Could not find LLVM library ${LIBNAME}, perhaps wrong setting of STATIC_LLVM ?")
    endif()
  endif()

  if(APPLE AND STATIC_LLVM AND VISIBILITY_HIDDEN)
    # -hidden-l doesn't accept paths, so split into name + directory
    get_filename_component(LIB_NAME ${L_LIBFILE_${LIBNAME}} NAME_WE)
    string(REPLACE "lib" "" LIB_BASE ${LIB_NAME})
    list(APPEND LLVM_LINK_LIBRARIES "-Wl,-hidden-l${LIB_BASE}")
    get_filename_component(LIB_DIR ${L_LIBFILE_${LIBNAME}} DIRECTORY)
    list(APPEND LLVM_LINK_DIRS ${LIB_DIR})
  else()
    list(APPEND LLVM_LINK_LIBRARIES ${L_LIBFILE_${LIBNAME}})
  endif()
endforeach()

if(LLVM_LINK_DIRS)
  list(REMOVE_DUPLICATES LLVM_LINK_DIRS)
endif()

####################################################################

if(STATIC_LLVM)
  set(CLANG_LIBNAMES clangCodeGen clangFrontendTool clangFrontend clangAPINotes clangDriver
      clangSerialization  clangParse clangSema clangRewrite clangRewriteFrontend
      clangStaticAnalyzerFrontend clangStaticAnalyzerCheckers
      clangStaticAnalyzerCore clangAnalysis clangEdit clangAST clangASTMatchers clangLex clangSupport clangBasic)
else()
  # For non-static builds, link against a single shared library
  # instead of multiple component shared libraries.
  if("${LLVM_LIBNAMES}" MATCHES "LLVMTCE")
    set(CLANG_LIBNAMES clangTCE-cpp)
  else()
    set(CLANG_LIBNAMES clang-cpp)
  endif()
endif()

unset(CLANG_LINK_LIBRARIES)
unset(CLANG_LINK_DIRS)
foreach(LIBNAME ${CLANG_LIBNAMES})
  find_library(C_LIBFILE_${LIBNAME} NAMES "${LIBNAME}" HINTS "${LLVM_LIBDIR}")
  if(NOT C_LIBFILE_${LIBNAME})
    message(FATAL_ERROR "Could not find Clang library ${LIBNAME}, perhaps wrong setting of STATIC_LLVM ?")
  endif()

  if(APPLE AND STATIC_LLVM AND VISIBILITY_HIDDEN)
    # -hidden-l doesn't accept paths, so split into name + directory
    get_filename_component(LIB_NAME ${C_LIBFILE_${LIBNAME}} NAME_WE)
    string(REPLACE "lib" "" LIB_BASE ${LIB_NAME})
    list(APPEND CLANG_LINK_LIBRARIES "-Wl,-hidden-l${LIB_BASE}")
    get_filename_component(LIB_DIR ${C_LIBFILE_${LIBNAME}} DIRECTORY)
    list(APPEND CLANG_LINK_DIRS ${LIB_DIR})
  else()
    list(APPEND CLANG_LINK_LIBRARIES ${C_LIBFILE_${LIBNAME}})
  endif()
endforeach()

if(CLANG_LINK_DIRS)
  list(REMOVE_DUPLICATES CLANG_LINK_DIRS)
endif()

# if enabled, CPU driver on Windows will use lld-link (invoked via library API)
# to link final kernel object files, instead of the default Clang driver linking.
set(CPU_USE_LLD_LINK_WIN32 OFF)
# TODO WIN32 or MSVC ? does this work with MINGW ?
if(ENABLE_HOST_CPU_DEVICES AND MSVC AND ENABLE_LLVM AND STATIC_LLVM AND X86)
  find_library(LIB_LLD_COFF NAMES "lldCOFF" HINTS "${LLVM_LIBDIR}")
  find_library(LIB_LLD_COMMON NAMES "lldCommon" HINTS "${LLVM_LIBDIR}")
  if(LIB_LLD_COFF AND LIB_LLD_COMMON)
    message(STATUS "Using lld-link via library to link kernels for CPU devices")
    set(CPU_USE_LLD_LINK_WIN32 ON)
  list(APPEND LLVM_LINK_LIBRARIES ${LIB_LLD_COFF} ${LIB_LLD_COMMON})
  endif()
endif()

####################################################################
