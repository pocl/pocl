/* (C) Copyright 2019 Robert Sauter
 * SPDX-License-Identifier: MIT
 */

/** Pthread-barrier implementation for macOS using a pthread mutex and
 * condition variable */

#include "pthread_barrier.h"
#include <errno.h>

#ifdef __APPLE__

int
pthread_barrier_init (pthread_barrier_t *__restrict barrier,
                      const pthread_barrierattr_t *__restrict attr,
                      unsigned count)
{
  if (count == 0)
    {
      return EINVAL;
    }

  int ret;

  pthread_condattr_t condattr;
  pthread_condattr_init (&condattr);
  if (attr)
    {
      int pshared;
      ret = pthread_barrierattr_getpshared (attr, &pshared);
      if (ret)
        {
          return ret;
        }
      ret = pthread_condattr_setpshared (&condattr, pshared);
      if (ret)
        {
          return ret;
        }
    }

  ret = pthread_mutex_init (&barrier->mutex, attr);
  if (ret)
    {
      return ret;
    }

  ret = pthread_cond_init (&barrier->cond, &condattr);
  if (ret)
    {
      pthread_mutex_destroy (&barrier->mutex);
      return ret;
    }

  barrier->count = count;
  barrier->left = count;
  barrier->round = 0;

  return 0;
}

int
pthread_barrier_destroy (pthread_barrier_t *barrier)
{
  if (barrier->count == 0)
    {
      return EINVAL;
    }

  barrier->count = 0;
  int rm = pthread_mutex_destroy (&barrier->mutex);
  int rc = pthread_cond_destroy (&barrier->cond);
  return rm ? rm : rc;
}

int
pthread_barrier_wait (pthread_barrier_t *barrier)
{
  pthread_mutex_lock (&barrier->mutex);
  if (--barrier->left)
    {
      unsigned round = barrier->round;
      do
        {
          pthread_cond_wait (&barrier->cond, &barrier->mutex);
        }
      while (round == barrier->round);
      pthread_mutex_unlock (&barrier->mutex);
      return 0;
    }
  else
    {
      barrier->round += 1;
      barrier->left = barrier->count;
      pthread_cond_broadcast (&barrier->cond);
      pthread_mutex_unlock (&barrier->mutex);
      return PTHREAD_BARRIER_SERIAL_THREAD;
    }
}

int
pthread_barrierattr_init (pthread_barrierattr_t *attr)
{
  return pthread_mutexattr_init (attr);
}

int
pthread_barrierattr_destroy (pthread_barrierattr_t *attr)
{
  return pthread_mutexattr_destroy (attr);
}

int
pthread_barrierattr_getpshared (const pthread_barrierattr_t *__restrict attr,
                                int *__restrict pshared)
{
  return pthread_mutexattr_getpshared (attr, pshared);
}

int
pthread_barrierattr_setpshared (pthread_barrierattr_t *attr, int pshared)
{
  return pthread_mutexattr_setpshared (attr, pshared);
}

#endif /* __APPLE */
