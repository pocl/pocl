# Decide whether the CPU drivers link final kernel binaries in-process
# through lld's library API instead of invoking the Clang driver, which needs
# no external toolchain (no linker to exec, no startup files). This keeps
# kernel linking -- and with it poclbinary export -- working in deployments
# that ship no host toolchain at all (see also the JIT, which loads the
# kernel objects without linking them). On Windows (MSVC) the kernel
# binaries are additionally linked without the C runtime, against bundled
# helper objects, so they need no VS Build Tools at run time either.
#
# Input: POCL_LLD_FIND_MODE -- "library" resolves the lld component archives
#        with find_library() in LLVM_LIBDIR; "cmake" uses the targets the LLD
#        CMake package exports (the caller must find_package(LLD) first so
#        LLD_EXPORTED_TARGETS is set).
# Output: CPU_USE_LLD_LINK, and the libraries in POCL_LLD_LIBRARIES. The
#         caller must put them BEFORE the LLVM libraries on the link line,
#         since the lld archives reference LLVM (and lldCommon) symbols.

set(CPU_USE_LLD_LINK OFF)
set(POCL_LLD_LIBRARIES "")
set(POCL_LLD_COMPONENTS "")

if(ENABLE_HOST_CPU_DEVICES AND ENABLE_LLVM)
  # TODO MinGW is excluded: the C-runtime-free link needs the bundled helper
  # objects (libchkstk/libmemory), which are only built for MSVC (see
  # lib/kernel/host/CMakeLists.txt). The work-group function carries
  # dllexport on all Windows triples (Workgroup.cc), so the DLL export table
  # is not a blocker.
  if(MSVC AND STATIC_LLVM)
    set(POCL_LLD_COMPONENTS lldCOFF lldMinGW lldCommon)
  elseif(NOT WIN32)
    if(APPLE)
      set(POCL_LLD_COMPONENTS lldMachO lldCommon)
    else()
      set(POCL_LLD_COMPONENTS lldELF lldCommon)
    endif()
  endif()
endif()

if(POCL_LLD_COMPONENTS)
  set(CPU_USE_LLD_LINK ON)
  foreach(POCL_LLD_COMPONENT IN LISTS POCL_LLD_COMPONENTS)
    if(POCL_LLD_FIND_MODE STREQUAL "cmake")
      if(${POCL_LLD_COMPONENT} IN_LIST LLD_EXPORTED_TARGETS)
        list(APPEND POCL_LLD_LIBRARIES ${POCL_LLD_COMPONENT})
      else()
        set(CPU_USE_LLD_LINK OFF)
      endif()
    else()
      find_library(POCL_LLD_LIB_${POCL_LLD_COMPONENT}
                   NAMES "${POCL_LLD_COMPONENT}" HINTS "${LLVM_LIBDIR}")
      if(POCL_LLD_LIB_${POCL_LLD_COMPONENT})
        list(APPEND POCL_LLD_LIBRARIES ${POCL_LLD_LIB_${POCL_LLD_COMPONENT}})
      else()
        set(CPU_USE_LLD_LINK OFF)
      endif()
    endif()
  endforeach()
endif()

if(CPU_USE_LLD_LINK)
  message(STATUS "Using lld via library to link kernels for CPU devices")
else()
  set(POCL_LLD_LIBRARIES "")
endif()
