void
subfunction ()
{
  printf ("Aborting from subfunction!\n");
#ifdef cl_ext_device_side_abort
  abort_platform ();
#else

#error cl_ext_device_side_abort not supported by the device!

#endif
}

kernel void
test_abort_platform ()
{
  subfunction ();
}
