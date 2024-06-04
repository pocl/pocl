#!/usr/bin/env bash
# Copyright (c) 2023 Jan Solanti / Tampere University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# exit immediately on error, see
# https://www.davidpashley.com/articles/writing-robust-shell-scripts/
set -e

BUILD_DIR=$1
TEST_BINARY=$2
shift 2

# If POCLD_PORT is defined, use a prelaunched PoCL-D at that port.
if [ -z "$POCLD_PORT" ]; then
 PORT=12000
else
 PORT=$POCLD_PORT
fi

echo "Running in $BUILD_DIR with PORT: $PORT"

if [ ! -e "$BUILD_DIR/pocld/pocld" ]; then
  echo "Can't find server binary at $BUILD_DIR/pocld/pocld"
  exit 1
fi

if [ ! -e "$BUILD_DIR/$TEST_BINARY" ]; then
  echo "Can't find test binary at $BUILD_DIR/$TEST_BINARY"
  exit 1
fi

export OCL_ICD_VENDORS=$BUILD_DIR/ocl-vendors/pocl-tests.icd
export POCL_BUILDING=1
export POCL_DEVICES="cpu"
export POCL_DEBUG=

if [ -z "$POCLD_PORT" ]; then
    "$BUILD_DIR/pocld/pocld" -a 127.0.0.1 -p "$PORT" -v error &
    POCLD_PID=$!
    echo "PoCL-D launched with PID: $POCLD_PID"
fi

sleep 1

export POCL_DEVICES="remote"
export POCL_REMOTE0_PARAMETERS="127.0.0.1:$PORT/0"
export POCL_DEBUG="err"
unset POCL_ENABLE_UNINIT

echo "Running $BUILD_DIR/$TEST_BINARY"

"$BUILD_DIR/$TEST_BINARY" "$@" &
EXAMPLE_PID=$!

# kill returns nonzero on success
set +e

RESULT=3
WAIT=1
while [ $WAIT -le 1000 ]; do
  if [ ! -e "/proc/$EXAMPLE_PID" ]; then
    echo "..finished"
    wait $EXAMPLE_PID
    RESULT=$?
    break
  fi
  WAIT=$((WAIT + 1))
  sleep 0.1
done

echo "DONE"

if [ -e "/proc/$EXAMPLE_PID" ]; then
  kill $EXAMPLE_PID
fi

if [ -z "$POCLD_PORT" ]; then
  if [ -e "/proc/$POCLD_PID" ]; then
     kill "$POCLD_PID"
  fi
fi

kill -9 $EXAMPLE_PID 1>/dev/null 2>&1

if [ -z "$POCLD_PORT" ]; then
    kill -9 "$POCLD_PID" 1>/dev/null 2>&1
fi

wait -f

exit $RESULT
