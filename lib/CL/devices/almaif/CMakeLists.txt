#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2019 Pekka Jääskeläinen / Tampere University
#                 2022 Topi Leppänen / Tampere University
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#   THE SOFTWARE.
#
#=============================================================================


set(ALMAIF_SOURCES "AlmaifShared.hh"
                  "almaif.cc"
                  "almaif.h"
                  "AlmaIFRegion.cc"
                  "AlmaIFRegion.hh"
                  "AlmaIFDevice.cc"
                  "AlmaIFDevice.hh"
                  "MMAPRegion.cc"
                  "MMAPRegion.hh"
                  "MMAPDevice.cc"
                  "MMAPDevice.hh"
                  "EmulationRegion.cc"
                  "EmulationRegion.hh"
                  "EmulationDevice.cc"
                  "EmulationDevice.hh"
                  "XilinxXrtDevice.hh"
                  "AlmaifCompile.cc"
                  "AlmaifCompile.hh"
                  "openasip/AlmaifCompileOpenasip.hh"
                  )

if(HAVE_XRT)
    add_compile_options(-I${XRT_INCLUDEDIR})
    set(ALMAIF_SOURCES ${ALMAIF_SOURCES}
                      "XilinxXrtDevice.cc"
                      "XilinxXrtRegion.cc"
                      "XilinxXrtRegion.hh"
                      "XilinxXrtExternalRegion.cc"
                      "XilinxXrtExternalRegion.hh"
                      "AlmaifDB/DBDevice.hh"
                      "AlmaifDB/DBDevice.cc"
                      "AlmaifDB/AlmaIFBitstreamDatabaseManager.cc"
                      "AlmaifDB/AlmaIFBitstreamDatabaseManager.hh"
                      "AlmaifDB/tiny-json.c"
                      "AlmaifDB/tiny-json.h"
                      )
endif()

if(ENABLE_TCE)
    set(ALMAIF_SOURCES ${ALMAIF_SOURCES}
                      "openasip/TTASimDevice.hh"
                      "openasip/TTASimDevice.cc"
                      "openasip/TTASimRegion.cc"
                      "openasip/TTASimRegion.hh"
                      "openasip/TTASimControlRegion.cc"
                      "openasip/TTASimControlRegion.hh"
                      "openasip/AlmaifCompileOpenasip.cc"
                      )
endif()

add_pocl_device_library(pocl-devices-almaif ${ALMAIF_SOURCES})

if(HAVE_XRT)
    target_link_libraries(pocl-devices-almaif PRIVATE "${XRT_LIBDIR}/libxrt_coreutil.so")
endif()

if(ENABLE_TCE)
    target_link_libraries(pocl-devices-almaif PRIVATE ${TCE_LIBS})

    target_compile_options(pocl-devices-almaif PRIVATE ${TCE_INCLUDES})
    target_compile_options(pocl-devices-almaif PRIVATE ${TCE_CXXFLAGS})
endif()

install(FILES "tce_builtins.cl"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}/almaif" COMPONENT "lib")
