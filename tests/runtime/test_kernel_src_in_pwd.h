#ifndef __FUNC__
#error __FUNC__ macro did not propagate to the Program
#endif

void __FUNC__();

kernel
void test_kernel() {
    __FUNC__();
}
