/* (C) Copyright 2019 Robert Sauter
 * SPDX-License-Identifier: MIT
 */

/** Pthread-barrier implementation for macOS */

#ifndef PTHREAD_BARRIER_H
#define PTHREAD_BARRIER_H

#include <pthread.h>

#ifdef __APPLE__

#ifdef __cplusplus
extern "C"
{
#endif

#ifndef PTHREAD_BARRIER_SERIAL_THREAD
#define PTHREAD_BARRIER_SERIAL_THREAD -1
#endif

  typedef pthread_mutexattr_t pthread_barrierattr_t;

  /* structure for internal use that should be considered opaque */
  typedef struct
  {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    unsigned count;
    unsigned left;
    unsigned round;
  } pthread_barrier_t;

  int pthread_barrier_init (pthread_barrier_t *__restrict barrier,
                            const pthread_barrierattr_t *__restrict attr,
                            unsigned count);
  int pthread_barrier_destroy (pthread_barrier_t *barrier);

  int pthread_barrier_wait (pthread_barrier_t *barrier);

  int pthread_barrierattr_init (pthread_barrierattr_t *attr);
  int pthread_barrierattr_destroy (pthread_barrierattr_t *attr);
  int
  pthread_barrierattr_getpshared (const pthread_barrierattr_t *__restrict attr,
                                  int *__restrict pshared);
  int pthread_barrierattr_setpshared (pthread_barrierattr_t *attr,
                                      int pshared);

#ifdef __cplusplus
}
#endif

#endif /* __APPLE__ */

#endif /* PTHREAD_BARRIER_H */
