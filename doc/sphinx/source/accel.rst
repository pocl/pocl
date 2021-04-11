===========================
Fixed-Function Accelerators
===========================

The ``accel`` driver can be used for easy integration of custom fixed-function
accelerators through a standardized hardware interface and a standardized
procedure for enqueuing commands.


Interface
---------

The control register interface for the fixed-function accelerators is quite
simple. The address space of the device is split into four regions, the size of
which is determined by the largest of the memories in these regions.
Therefore, the region is selected with the highest bits of the address space of
the accelerator:

+-------------+--------------------+
| High bits   | Address Space      |
|             |                    |
+=============+====================+
| 00          | Control registers  |
+-------------+--------------------+
| 01          | Instruction memory |
+-------------+--------------------+
| 10          | Data memory        |
+-------------+--------------------+
| 11          | Parameter memory   |
+-------------+--------------------+

The size of the memories is read from the control registers, which is sufficient
to determine the size of the address space of the accelerator as well as the
offsets of each memory. The control registers are also used to control the
execution of the accelerator:

.. list-table::
  :widths: 20 25 55
  :header-rows: 1

  * - Offset
    - Name
    - Description
  * - 0x000
    - STATUS
    - Status of the accelerator. Bit 0 is high when the execution is stalled
      due to any reason, bit 1 is high when the external stall signal is active,
      and bit 2 is high when the accelerator reset is active.
  * - 0x100
    - AQL_READ_IDX_LOW
    - Read index of the AQL queue (low 32 bits). Read only.
  * - 0x104
    - AQL_READ_IDX_HIGH
    - Read index of the AQL queue (high 32 bits). Read only.
  * - 0x108
    - AQL_WRITE_IDX_LOW
    - Write index of the AQL queue (low 32 bits). Writing to this register
      increments the 64-bit value.
  * - 0x10C
    - AQL_WRITE_IDX_HIGH
    - Write index of the AQL queue (high 32 bits). Read only.
  * - 0x200
    - COMMAND
    - Command register to control execution. Writing 1 to this register resets
      the accelerator, writing 2 lifts reset and external stall, and writing 4
      enables the external stall signal, pausing execution.
  * - 0x300
    - DEVICE_CLASS
    - Device class (vendor ID) of the accelerator. Currently unused by the
      driver.
  * - 0x304
    - DEVICE_ID
    - Device ID of the accelerator. Currently unused by the driver.
  * - 0x308
    - INTERFACE_TYPE
    - Version number of the interface. This describes interface
      version 2.
  * - 0x30C
    - CORE_COUNT
    - Core count of the accelerator. Multicore devices are currently not
      supported.
  * - 0x310
    - CTRL_SIZE
    - Size of control memory (this register space) in bytes.
      Must be at least 1024.
  * - 0x314
    - DMEM_SIZE
    - Size of the data memory in bytes
  * - 0x318
    - IMEM_SIZE
    - Size of the instruction memory in bytes
  * - 0x31c
    - PMEM_SIZE
    - Size of the parameter memory in bytes.

The instruction memory can be used to configure the accelerator. However, it
currently has to be done manually, and is not managed by pocl. The data memory
is used to store an AQL Queue, as defined by the `HSA Runtime Programmer’s
Reference Manual <http://www.hsafoundation.com/standards/>`_, the write and read
indexes of which are exposed by the control registers. The size of the queue is
such that it uses all of the data memory. Finally, the parameter memory is used
to store data and argument buffers as well as completion signals for the
kernels.

As a practical example, enqueuing a kernel dispatch packet proceeds as follows:

  - The driver allocates and populates the OpenCL buffers and the argument
    buffer for the kernel, as well as space for a 32-bit completion signal.
  - The driver writes the kernel packet, excluding the header, to the device.
    Its position depends on the value of the write index. The completion signal
    address as well as the argument buffer address and pointers to buffer
    arguments are given as physical addresses in the accelerator's address
    space. The kernel object simply corresponds to the kernel IDs shown in the
    table below.
  - The driver sets the packet header and increments the queue write index.
  - The device executes the kernel and writes a 1 in case of a success or a 2
    in case of a failure to the completion signal address, if it is not 0.
  - The driver sees the completion signal change, and can consider the command
    completed.

Usage
-----

To enable this driver, simply add ``-DENABLE_ACCEL_DEVICE=1`` to the cmake
arguments. On small FPGA SoCs and other relatively low performance hosts, you
may wish to follow the instructions in :ref:`pocl-without-llvm`.

The fixed-function accelerators need to be told what kernel to execute. For
this, the accel driver has a list of builtin kernels that can be referred to
in the ``clCreateProgramWithBuiltInKernels`` call:

.. list-table::
  :widths: 20 20 60
  :header-rows: 1

  * - Kernel name
    - Kernel ID
    - Function
  * - pocl.copy.i8
    - 0
    - Copies from argument 0 to argument 1 as many bytes as there are work items
  * - pocl.add.i32
    - 1
    - 32-bit element-wise addition on arrays pointed to by arguments 0 and 1,
      stored in an array pointed to by argument 3
  * - pocl.mul.i32
    - 2
    - As pocl.add.i32, but with 32-bit multiplication

This list will be expanded in the future.

There is an example program using the accel driver in ``examples/accel`` which
also includes the VHDL code for synthesizing the accelerator. The accelerator
has been developed with the `TCE toolset <http://openasip.org/>`_. In order to
synthesize the accelerator for a Xilinx FPGA SoC, you can follow the
instructions in the `TCE manual <http://openasip.org/user_manual/TCE.pdf>`_,
in the section titled System-on-a-Chip design with AlmaIF Integrator. Make sure
to check the accelerator base address from Vivado.

Driver arguments are used to tell pocl where the accelerator is and what
functions it supports. To run this example manually, execute::

  POCL_DEVICES=accel POCL_ACCEL0_PARAMETERS=0x43C00000,1,2 ./accel_example

The environment variables define an accelerator with base physical address of
0x43C0_0000 that can execute pocl.add.i32 and pocl.mul.i32. When running the
example, verify that the address given in the parameter matches the base address
of the accelerator.

There's an alternative way to emulate the accelerator in software by
setting the base physical address to 0xE. This directs the driver to instead
use a software emulating function from accel.cc. No changes to accel_example.cpp
are needed to run the emulation.

Note that as the driver requires write access to ``/dev/mem`` for memory
mapping, you may need to execute the application with elevated privileges. In
this case, note that ``sudo`` by default overrides your environment variables.
You can either assign them in the same command, or use ``sudo`` with the
``--preserve-env`` switch.
