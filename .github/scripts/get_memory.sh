#!/bin/sh

set -eu

if [ "${DEBUG:-}" = true ]; then
  set -x
fi

verbose() {
  if [ "${VERBOSE:-}" = true ]; then
    echo "VERBOSE(get_memory.sh):" "$@" >&2
  fi
}

warning() {
  echo "WARNING(get_memory.sh):" "$@" >&2
}

if [ -f /sys/fs/cgroup/cgroup.controllers ]; then
  verbose "cgroup v2 detected."
  if [ -f /sys/fs/cgroup/memory.max ]; then
    memory_bytes=$(cat /sys/fs/cgroup/memory.max)
    if [ "${memory_bytes}" = "max" ]; then
      verbose "No memory limits set."
      unset memory_bytes
    fi
  else
    warning "/sys/fs/cgroup/memory.max not found. Falling back to /proc/meminfo."
  fi
else
  verbose "cgroup v1 detected."

  if [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
    memory_bytes=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
    memory_kb=$((memory_bytes / 1024))
    proc_memory_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    if [ "${memory_kb}" -ge "${proc_memory_kb}" ]; then
      verbose "No memory limits set."
      unset memory_bytes
    fi
    unset memory_kb proc_memory_kb
  else
    warning "/sys/fs/cgroup/memory/memory.limit_in_bytes not found. Falling back to /proc/meminfo."
  fi
fi

if [ -n "${memory_bytes:-}" ]; then
  memory_mb=$((memory_bytes / 1024 / 1024))
else
  proc_memory_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
  memory_mb=$((proc_memory_kb / 1024))
  unset proc_memory_kb
fi

verbose "Memory (MB):"
printf '%s' "${memory_mb}"
