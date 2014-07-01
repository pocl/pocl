#ifndef POCL_TESTS_H
#define POCL_TESTS_H

#define CHECK_OPENCL_ERROR_IN(func_name)                                \
do {                                                                    \
   if(check_cl_error(err, __LINE__, func_name))                         \
     return EXIT_FAILURE;                                               \
} while (0)

#define TEST_ASSERT(EXP)                                                \
do {                                                                    \
  if (!(EXP)) {                                                         \
    fprintf(stderr, "Assertion: \n" #EXP "\nfailed on %s:%i\n",         \
        __FILE__, __LINE__);                                            \
    return EXIT_FAILURE;                                                \
  }                                                                     \
} while (0)

#endif
