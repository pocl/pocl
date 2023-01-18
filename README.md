# Portable Computing Language (PoCL) for Ventus GPGPU

See README-pocl.md for original README.

See https://github.com/THU-DSP-LAB/llvm-project for detailed build guide with
Ventus llvm based OpenCL Compiler and icd loader.

Original PoCL build instruction doesn't work for Ventus GPGPU.


## TODOs

TODOs are divided into 2 parts, first part contains jobs required to make sure
OpenCL tests can be tested with pocl+spike, second part contains jobs required
to enable real Ventus GPGPU work flow.

### TODOs(Part 1)

  * Correctly report Ventus GPGPU virtual device after pocl is statically/dynamally
    linked with spike(Currently ventus device is hardcoded).
  * Determine how we implement workitem builtins, if using workitem.S implementation
    from ventus-llvm, some extra work should to be done in ventus pocl driver,
    otherwise we need to redesign how workitem builtins are implemented.
  * Make sure pocl ventus driver is correctly initialized, such as `pocl_ventus_init`
    is double checked.
  * Make sure there is no native(host) device related code in ventus pocl driver,
    such as `kernel.so` in function `llvm_codegen` should be renamed to `kernel.elf`
    or something else(a static elf file). Also the native kernel execution
    `dlopen(kernel.so)` should be replaced by an elf loader(Should already be done
    by spike fesrv elf loader).
  * Kernel metadata buffer and kernel args buffer should be prepared in pocl, then
    passed to spike by asking spike to store those buffer into the lowerest address of
    physical memory of Ventus GPGPU, related CSRs should be initialized with the kernel
    metadata buffer base address.
  * Check the behavior of all the OpenCL APIs for Ventus GPGPU are correctly coded(Some
    API implementations in common.c may not work for Ventus GPGPU), make sure all the
    examples and tests can be kicked off(may not run through) to spike(via spike elf
    loader).

### TODOs(Part 2)

  * Ventus GPGPU kernel mode driver(kmd) should be provided and bridged with pocl.
  * Correctly report Ventus GPGPU device after pocl(umd) can locate ventus kmd and
    read necessory information from the kmd(such as supported extensions etc).