#!/bin/sh

set -eu

cpus=

if [ "${DEBUG:-}" = true ]; then
  set -x
fi

verbose() {
  if [ "${VERBOSE:-}" = true ]; then
    echo "VERBOSE(get_cpus.sh):" "$@" >&2
  fi
}

warning() {
  if [ "${VERBOSE:-}" = true ]; then
    echo "WARNING(get_cpus.sh):" "$@" >&2
  fi
}

if [ -f /sys/fs/cgroup/cgroup.controllers ]; then
  verbose "cgroup v2 detected."
  if [ -f /proc/self/cgroup ]; then
    CGROUP=$(grep '0::' /proc/self/cgroup | cut -d ':' -f 3)
    verbose "Using CGROUP: $CGROUP"
    CGROUPFILE="/sys/fs/cgroup$CGROUP/cpu.max"
    if [ -r "$CGROUPFILE" ]; then
      read -r quota period <$CGROUPFILE
      if [ "${quota}" = "max" ]; then
        verbose "quota=max; no CPU limits set."
        unset quota period
      fi
    else
      warning "CGROUP file $CGROUPFILE not found. Falling back to nproc."
    fi
  else
    warning "CGROUP cpu.max not found. Falling back to nproc."
  fi
else
  verbose "cgroup v1 detected."

  if [ -f /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] && [ -f /sys/fs/cgroup/cpu/cpu.cfs_period_us ]; then
    quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
    period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)

    if [ "${quota}" = "-1" ]; then
      verbose "No CPU limits set."
      unset quota period
    fi
  else
    warning "/sys/fs/cgroup/cpu/cpu.cfs_quota_us or /sys/fs/cgroup/cpu/cpu.cfs_period_us not found. Falling back to nproc."
  fi
fi

# This is theoretically not possible, but:
# https://github.com/blakeblackshear/frigate/discussions/11755#discussioncomment-10304356
if [ -n "${period:-}" ] && [ "${period:-}" -eq 0 ]; then
  warning "CPU period is 0. Falling back to nproc."
  unset quota period
fi

if [ -n "${quota:-}" ] && [ -n "${period:-}" ]; then
  cpus=$((quota / period))
  if [ "${cpus}" -eq 0 ]; then
    cpus=1
  fi
else
  if which nproc 1>/dev/null 2>/dev/null; then
    verbose "using nproc"
    cpus=$(nproc)
  elif [ -r /proc/cpuinfo ];  then
    verbose "using proc/cpuinfo"
    cpus=$(grep -c ^processor /proc/cpuinfo)
  fi
fi

if [ -z "$cpus" ]; then
  cpus=1
fi

verbose "CPUs:"
printf '%s' "${cpus}"
