#define HILO_TO_ULONG(v) ((ulong)(v).x | ((ulong)(v).y << 32))

kernel void test_kernel_clock() {
    ulong dev0 = clock_read_device();
    ulong dev1 = clock_read_device();
    ulong wg0 = clock_read_work_group();
    ulong wg1 = clock_read_work_group();
    ulong sg0 = clock_read_sub_group();
    ulong sg1 = clock_read_sub_group();

    ulong dev_hilo0 = HILO_TO_ULONG(clock_read_hilo_device());
    ulong dev_hilo1 = HILO_TO_ULONG(clock_read_hilo_device());
    ulong wg_hilo0 = HILO_TO_ULONG(clock_read_hilo_work_group());
    ulong wg_hilo1 = HILO_TO_ULONG(clock_read_hilo_work_group());
    ulong sg_hilo0 = HILO_TO_ULONG(clock_read_hilo_sub_group());
    ulong sg_hilo1 = HILO_TO_ULONG(clock_read_hilo_sub_group());

    if (dev0 > dev1) {
        printf("Device clocks: FAIL (%lu > %lu)\n", dev0, dev1);
    }
    if (wg0 > wg1) {
        printf("Work-group clocks: FAIL (%lu > %lu)\n", wg0, wg1);
    }
    if (sg0 > sg1) {
        printf("Sub-group clocks: FAIL (%lu > %lu)\n", sg0, sg1);
    }

    if (dev_hilo0 > dev_hilo1) {
        printf("Device hilo clocks: FAIL (%lu > %lu)\n",
               dev_hilo0, dev_hilo1);
    }
    if (wg_hilo0 > wg_hilo1) {
        printf("Work-group hilo clocks: FAIL (%lu > %lu)\n",
               wg_hilo0, wg_hilo1);
    }
    if (sg_hilo0 > sg_hilo1) {
        printf("Sub-group hilo clocks: FAIL (%lu > %lu)\n",
               sg_hilo0, sg_hilo1);
    }
}
