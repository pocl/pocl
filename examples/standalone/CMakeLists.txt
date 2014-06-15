#EXTRA_DIST = standalone.cl

#noinst_DATA = $(EXTRA_DIST:.cl=.bc)

#.cl.bc:
# ../../scripts/pocl-standalone -h $(@:.bc=.h) -o $@ $<

#clean-local:
# rm -f $(EXTRA_DIST:.cl=.bc) $(EXTRA_DIST:.cl=.h)

add_test("pocl-standalone" "/bin/bash" "${CMAKE_BINARY_DIR}/scripts/pocl-standalone"
          -h "${CMAKE_BINARY_DIR}/standalone.h"
          -o "${CMAKE_BINARY_DIR}/standalone.bc"
          "${CMAKE_CURRENT_SOURCE_DIR}/standalone.cl")

