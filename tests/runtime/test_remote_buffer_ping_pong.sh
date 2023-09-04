#!/bin/bash

BUILD_DIR=$1

PORT1=$((3000 + $RANDOM % 6000))
PORT2=$(($PORT1 + 10))

echo "Running in $BUILD_DIR with PORT: $PORT"

if [ ! -e "$BUILD_DIR/pocld/pocld" ]; then
  echo "Can't find server binary at $BUILD_DIR/pocld/pocld"
  exit 1
fi

if [ ! -e "$BUILD_DIR/tests/runtime/test_buffer_ping_pong" ]; then
  echo "Can't find test_buffer_ping_pong binary at $BUILD_DIR/tests/runtime/test_buffer_ping_pong"
  exit 1
fi

unset OCL_ICD_VENDORS
unset POCL_BUILDING
unset POCL_DEVICES

$BUILD_DIR/pocld/pocld -a 127.0.0.1 -p $PORT1 -v error &
POCLD_PID1=$!

echo "Pocld running with PID: $POCLD_PID1"

$BUILD_DIR/pocld/pocld -a 127.0.0.1 -p $PORT2 -v error &
POCLD_PID2=$!

echo "Pocld running with PID: $POCLD_PID2"

sleep 1

export POCL_DEVICES="remote remote"
export POCL_REMOTE0_PARAMETERS="127.0.0.1:$PORT1/0"
export POCL_REMOTE1_PARAMETERS="127.0.0.1:$PORT2/0"
export OCL_ICD_VENDORS=$BUILD_DIR/ocl-vendors/pocl-tests.icd
export POCL_BUILDING=1
export POCL_DEBUG=warn,err
unset POCL_ENABLE_UNINIT

echo "Running example1"

sleep 1

$BUILD_DIR/tests/runtime/test_buffer_ping_pong &
EXAMPLE_PID=$!

RESULT=3
WAIT=1
while [ $WAIT -le 10 ]; do
  echo "waiting for test_buffer_ping_pong.."
  if [ ! -e "/proc/$EXAMPLE_PID" ]; then
    echo "..finished"
    RESULT=$?
    break
  fi
  WAIT=$((WAIT + 1))
  sleep 1
done

echo "DONE"

if [ -e "/proc/$EXAMPLE_PID" ]; then
  kill $EXAMPLE_PID
fi

if [ -e "/proc/$POCLD_PID1" ]; then
  kill $POCLD_PID1
fi

if [ -e "/proc/$POCLD_PID2" ]; then
  kill $POCLD_PID2
fi

sleep 2

kill -9 $EXAMPLE_PID 1>/dev/null 2>&1
kill -9 $POCLD_PID 1>/dev/null 2>&1

exit $RESULT
