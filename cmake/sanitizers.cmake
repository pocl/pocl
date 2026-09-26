# Sanitizers work with GCC and Clang as the host compiler.
if(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
  option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
  option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
  option(ENABLE_UBSAN "Enable UBSanitizer" OFF)
  option(ENABLE_LSAN "Enable LeakSanitizer" OFF)
elseif(ENABLE_ASAN OR ENABLE_TSAN OR ENABLE_UBSAN OR ENABLE_LSAN)
  message(FATAL_ERROR "Sanitizers require GCC or Clang as the host compiler")
endif()

if(ENABLE_ASAN OR ENABLE_TSAN OR ENABLE_UBSAN OR ENABLE_LSAN)
  if(NOT CMAKE_C_COMPILER_ID STREQUAL CMAKE_CXX_COMPILER_ID)
    message(FATAL_ERROR "Sanitizers need the same C and C++ compiler, got "
                        "${CMAKE_C_COMPILER_ID} and ${CMAKE_CXX_COMPILER_ID}")
  endif()
  if(ENABLE_TSAN AND (ENABLE_ASAN OR ENABLE_LSAN))
    message(FATAL_ERROR "ThreadSanitizer cannot be combined with "
                        "AddressSanitizer or LeakSanitizer")
  endif()
endif()


set(SANITIZER_OPTIONS "")
set(SANITIZER_LIBS "")

if(ENABLE_ASAN)
  list(APPEND SANITIZER_OPTIONS "-fsanitize=address" "-fsanitize-recover=address")
  list(APPEND SANITIZER_LIBS "asan")
endif()

if(ENABLE_LSAN)
  if(ENABLE_ASAN)
    message(STATUS "LeakSanitizer is part of AddressSanitizer")
  else()
    list(APPEND SANITIZER_OPTIONS "-fsanitize=leak")
    list(APPEND SANITIZER_LIBS "lsan")
  endif()
endif()

if(ENABLE_TSAN)
  list(APPEND SANITIZER_OPTIONS "-fsanitize=thread")
  list(APPEND SANITIZER_LIBS "tsan")
endif()

if(ENABLE_UBSAN)
  list(APPEND SANITIZER_OPTIONS "-fsanitize=undefined")
  list(APPEND SANITIZER_LIBS "ubsan")
  if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
    list(APPEND SANITIZER_OPTIONS "-fno-sanitize=function")
  endif()
endif()

if(SANITIZER_OPTIONS)
  if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
    # SANITIZER_LIBS are GCC's runtime libraries. Clang links its own
    # runtime into executables when -fsanitize= is given at link time.
    set(SANITIZER_LIBS "")
    add_link_options(${SANITIZER_OPTIONS})
    string(JOIN " " SANITIZER_EXE_LINKER_FLAGS_STR ${SANITIZER_OPTIONS})
  else()
    list(TRANSFORM SANITIZER_LIBS PREPEND "-l"
         OUTPUT_VARIABLE SANITIZER_LIB_FLAGS)
    string(JOIN " " SANITIZER_EXE_LINKER_FLAGS_STR ${SANITIZER_LIB_FLAGS})
  endif()
  list(APPEND SANITIZER_OPTIONS "-fno-omit-frame-pointer")
  add_compile_options(${SANITIZER_OPTIONS})
  string(JOIN " " SANITIZER_FLAGS_STR ${SANITIZER_OPTIONS})
endif()


# Unfortunately the way CMake tests work, if they're given
# a pass/fail expression, they don't check for exit status.
# This was causing some false negatives with ASan (test was
# returning with 1, but CMake reported it as pass because
# the pass expression was present in output).
if(ENABLE_ASAN OR ENABLE_TSAN OR ENABLE_UBSAN OR ENABLE_LSAN)
  set(ENABLE_ANYSAN 1)
endif()
