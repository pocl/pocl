#!/usr/bin/env bash
# Copyright (c) 2023-2024 Yashvardhan Agarwal / Tampere University
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

# https://www.davidpashley.com/articles/writing-robust-shell-scripts/
# Enable immediate exit on error
set -e

BUILD_DIR=$1
TEST_BINARY=$2
MODE=$3
RESULT=0  # Initialize RESULT
shift 3 # Shift for other arguments passed

# Check if binaries exist
if [ ! -e "$BUILD_DIR" ]; then
  echo "Server binary not found at $BUILD_DIR/pocld/pocld"
  exit 1
fi

if [ ! -e "$BUILD_DIR/$TEST_BINARY" ]; then
  echo "Test binary not found at $BUILD_DIR/$TEST_BINARY"
  exit 1
fi

# Remove log files if they already exist
rm -f "$BUILD_DIR/client_output.log" "$BUILD_DIR/server_output.log"

# Generate a random 32-char string for POCL_REMOTE_DHT_KEY to be used as common key
DHT_KEY=$(tr -dc 'a-zA-Z' < /dev/urandom | head -c 20)

# Set specific client environment variables based on mode
if [ "$MODE" == "mdns" ]; then
  echo "Running for mDNS."
  export POCL_DEVICES=""
  export POCL_DEBUG="none"
  export POCL_DISCOVERY=1
  SUCCESS_MESSAGE_CLIENT="Server IP: 127.0.0.1"

elif [ "$MODE" == "dht" ]; then
  echo "Running for DHT."
  export POCL_DEVICES=""
  export POCL_DEBUG="none"
  export POCL_DISCOVERY=1
  export POCL_REMOTE_DHT_PORT=4333
  export POCL_REMOTE_DHT_BOOTSTRAP="bootstrap.jami.net"
  export POCL_REMOTE_DHT_KEY="$DHT_KEY"
  SUCCESS_MESSAGE_CLIENT="Server IP: 127.0.0.1"
else
  echo "Unknown value: \"$MODE\". Use \"mdns\" or \"dht\""
  exit 1
fi

echo "Starting test app on client..."
"$BUILD_DIR/$TEST_BINARY" "$@" > "$BUILD_DIR/client_output.log" 2>&1 &
CLIENT_PID=$!
echo "Client PID: $CLIENT_PID"
sleep 1

# Set specific server environment variables based on mode
if [ "$MODE" == "mdns" ]; then
  unset POCL_DISCOVERY
  export POCL_DEVICES="cpu"
  export OCL_ICD_VENDORS=$BUILD_DIR/ocl-vendors/pocl-tests.icd
  SUCCESS_MESSAGE_SERVER="Avahi service '[a-fA-F0-9]\{32\}' successfully established"
elif [ "$MODE" == "dht" ]; then
  unset POCL_DISCOVERY
  export POCL_DEVICES="cpu"
  export OCL_ICD_VENDORS=$BUILD_DIR/ocl-vendors/pocl-tests.icd
  export POCL_REMOTE_DHT_PORT=12333
  export POCL_REMOTE_DHT_BOOTSTRAP="bootstrap.jami.net"
  export POCL_REMOTE_DHT_KEY="$DHT_KEY"
  SUCCESS_MESSAGE_SERVER="Publishing on DHT"
else
  echo "Unknown value: \"$MODE\". Use \"mdns\" or \"dht\""
  exit 1
fi

echo "Starting server..."
"$BUILD_DIR/pocld/pocld" -a 127.0.0.1 -p "12345" --log_filter error,general,warn,remote > "$BUILD_DIR/server_output.log" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Function to read log files with timeout
read_log_with_timeout() {
  local log_file="$1"
  local success_message="$2"
  local timeout_duration=100
  local elapsed_time=0
  local interval=0.1

  while [ $elapsed_time -lt $timeout_duration ]; do
    if grep -q "$success_message" "$log_file"; then
      return 0  # Success
    fi
    sleep "$interval"
    elapsed_time=$((elapsed_time + 1))  # Increment elapsed time
  done
  RESULT=$(( RESULT + 1 ))
  return 1  # Timeout
}

set +e

# Function to shut down server and client
goto_shutdown() {
  echo "Shutting down server..."
  if [ -e "/proc/$SERVER_PID" ]; then
    kill -9 "$SERVER_PID" 2>/dev/null 2>&1
  fi

  echo "Shutting down client..."
  if [ -e "/proc/$CLIENT_PID" ]; then
    kill -9 "$CLIENT_PID" 2>/dev/null 2>&1
  fi

  if [ $RESULT == 0 ]; then
    echo "OK"
  fi
  exit $RESULT
}

# Wait for server to output confirmation
echo "Reading server log to confirm advertisement..."
if ! read_log_with_timeout "$BUILD_DIR/server_output.log" "$SUCCESS_MESSAGE_SERVER"; then
  RESULT=$(( RESULT + 1 ))
  echo "Timeout waiting for server advertisement!"
  goto_shutdown
fi

# Wait for client to output the discovered server's IP
echo "Reading client log to confirm discovery..."
if ! read_log_with_timeout "$BUILD_DIR/client_output.log" "$SUCCESS_MESSAGE_CLIENT"; then
  RESULT=$(( RESULT + 1 ))
  echo "Timeout waiting for client to discover server!"
  goto_shutdown
fi

# Normal shutdown
goto_shutdown