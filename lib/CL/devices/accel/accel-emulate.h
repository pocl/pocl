
#ifndef POCL_ACCELEMULATE_H
#define POCL_ACCELEMULATE_H

#define EMULATING_ADDRESS 0xE
#define EMULATING_MAX_SIZE 4194304
//#define EMULATING_MAX_SIZE 4 * 4096

struct emulation_data_t
{
  int Emulating;
  pthread_t emulate_thread;
  void *emulating_address;
  volatile int emulate_exit_called;
  volatile int emulate_init_done;
};

#ifdef __cplusplus
extern "C"
{
#endif

  void *emulate_accel (void *E_void);

#ifdef __cplusplus
}
#endif

#endif
