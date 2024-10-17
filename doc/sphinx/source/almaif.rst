===========================
Fixed-Function Accelerators
===========================

The ``almaif`` driver can be used for easy integration of custom fixed-function
accelerators through a standardized hardware interface and a standardized
procedure for enqueuing commands. More information about the interface can
be found from the publications at the end of this page.


Interface
---------

The control register interface for the fixed-function accelerators is quite
simple. The address space of the device is split into four regions. The sizes
and starting addresses of the regions are advertised in the control region
by the accelerator.

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

The control registers are also used to control the
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
      version 3.
  * - 0x30C
    - CORE_COUNT
    - Core count of the accelerator. Multicore devices are currently not
      supported.
  * - 0x310
    - CTRL_SIZE
    - Size of control memory (this register space) in bytes.
      Must be at least 1024.
  * - 0x314
    - IMEM_SIZE
    - Size of the instruction memory in bytes
  * - 0x318
    - IMEM_STARTING_ADDRESS (64b)
    - Starting address of the instruction memory
  * - 0x320
    - CQMEM_SIZE (64b)
    - Size of the command queue memory in bytes. The cq region includes
      a ring buffer of 64B packets and the 64B queue header. Therefore
      the CQMEM_SIZE is (queue_length + 1) * 64.
  * - 0x328
    - CQMEM_STARTING_ADDRESS (64b)
    - Starting address of the command queue memory
  * - 0x330
    - BUFFERMEM_SIZE
    - Size of the data memory reserved for on-chip buffers.
  * - 0x338
    - BUFFERMEM_STARTING_ADDRESS
    - Starting address of the buffer memory.
  * - 0x340
    - FEATURE_FLAGS (64b)
    - Bitmap of various features.

      Bit 0: HAS_MASTER_INTERFACE. If set to 1, the accelerator can access outside
      of its AlmaIF address space with a master interface. This also means that the
      above X_STARTING_ADDRESS values are absolute values. Also, any pointers to data buffers,
      completion signals etc. are given to the accelerator as absolute addresses and
      the accelerator needs to decode the address to determine whether the pointer
      points to its own buffermem region or external devices or memories.
      If set to 0, the data buffers, completion signals etc. are given as pointers
      relative to the beginning of the buffer mem region, and the device is assumed
      to not being able to access external devices or memories.

      Bits: 1-63: Reserved, should be set to 0

The other three regions are used for the following:
The instruction memory can be used to configure the accelerator. PoCL looks for
<device_name>.img-binary file and if it exists, writes it to the region at the initialization time.
In case of compiled kernels, PoCL will overwrite this region with the new program.
The command queue memory is used to store an AQL Queue, as defined by the `HSA Runtime Programmer’s
Reference Manual <http://www.hsafoundation.com/standards/>`_, the write and read
indexes of which are exposed in a 64B header at the beginnig of the region.
The size of the queue is such that it uses all of the region remaining after the header.
Finally, the buffer memory is used to store data and argument buffers as well as
completion signals for the kernels.

As a practical example, enqueuing a kernel dispatch packet proceeds as follows:

  - The driver allocates and populates the OpenCL buffers and the argument
    buffer for the kernel, as well as space for a 32-bit completion signal.
  - The driver writes the kernel packet to the device.
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

.. _almaif_usage:

Usage
-----

To enable this driver, simply add ``-DENABLE_ALMAIF_DEVICE=1`` to the cmake
arguments. On small FPGA SoCs and other relatively low performance hosts, you
may wish to follow the instructions in :ref:`pocl-without-llvm`.
(Recommended for Zynq-7020 SoC).

The fixed-function accelerators need to be told what kernel to execute. For
this, the almaif driver has a list of builtin kernels that can be referred to
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
  * - Online compiler available.
    - 65535
    - Special flag to communicate that device supports compiled kernels.

The full list of currently supported built-in kernels is maintained in
lib/CL/devices/builtin_kernels.{cc,hh}

To execute tests in examples/accel that generate both TTA and High-level synthesis
(HLS) based accelerators for PYNQ-Z1 device you need to enable few variables
in the CMAKE configuration.
First, set CMAKE variable VIVADO_PATH to point to the directory with the
'vivado' executable. (E.g. at Xilinx/Vivado/2021.2/bin/)

1. If you have the `OpenASIP <http://openasip.org/>`_ toolset installed,
you can set ENABLE_TCE to 1 to enable
RTL and firmware generation of various OpenASIP TTA cores with different memory configurations.
Then, you can simulate them with ttasim instruction set simulator by running
``../tools/scripts/run_almaif_tests`` from the build directory.

2. If you have Vitis HLS installed, set VITIS_HLS_PATH to point to the directory
with the vitis_hls executable.
This enables the generation of fixed-function accelerator from C description.

The bitstreams themselves are not automatically built with PoCL build process, but rather
with a separate 'make bitstreams' command. This generates the bitstreams to
build/examples/accel/bitstreams and build/examples/accel/hls/bitstreams directories. 
Once bitstreams have been built, build PoCL on the PYNQ-Z1 device.
(You don't need to set ENABLE_TCE or VIVADO/VITIS_HLS_PATH).
Copy the bistreams directories (and in case of TTA, also the firmware_imgs
directory and example0_*.poclbins)
to their correct PoCL build directories on PYNQ.
Finally, run ``../tools/scripts/run_almaif_tests`` to run the test program
on the FPGA device.




Driver arguments are used to tell pocl where the accelerator is and what
functions it supports. To run examples manually, after programming the
FPGA, execute::

  POCL_DEVICES=almaif POCL_ALMAIF0_PARAMETERS=0x40000000,<device_name>,1,2 ./accel_example

The environment variables define an accelerator with base physical address of
0x4000_0000 that can execute pocl.add.i32 and pocl.mul.i32. If the device requires
firmware to be loaded in, pocl will attempt to load it from <device_name>.img.
When running the example, verify that the address given in the parameter matches
the base address of the accelerator.


Note that as the driver requires write access to ``/dev/mem`` for memory
mapping, you may need to execute the application with elevated privileges. In
this case, note that ``sudo`` by default overrides your environment variables.
You can either assign them in the same command, or use ``sudo`` with the
``--preserve-env`` switch.



The driver supports instruction-set simulation for TTA devices. To enable it,
set the base address to 0xB, and set the <device_name> to point to a TTA
device's .adf-file and compiled firmware binary (.tpef-file). PoCL will then
start up the simulation with <device_name>.adf and, if it exists, <device_name>.tpef.



There's an alternative way to emulate the accelerator in software by
setting the base physical address to 0xE. This directs the driver to instead
use a software emulating function from almaif/EmulationDevice.cc.
No changes to the source OpenCL host program (e.g. accel_example.cpp)
when switching between emulation, instruction-set simulation or FPGA execution.


Standalone mode
^^^^^^^^^^^^^^^

Almaif driver's OpenASIP backend supports a standalone mode meant for executing
kernels without the host runtime. The standalone mode generates a C program
that contains the input data and pre-initialized command structures to run
a single kernel with either ttasim or RTL simulation. The mode is enabled
with an environment variable POCL_ALMAIF_STANDALONE=1. The mode generates
helper scripts to the working directory, while outputting the standalone C program
to the kernel CACHE directory.

Example usage of the mode can be found in examples/accel/CMakelists.txt, which
generates standalone tests using both ttasim and RTL simulator (ghdl) to run the
example0 kernel on various TTA configurations.


Using a bitstream database
^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use the AlmaIF-driver with the cross-vendor bitstream databases generated
with the `AFOCL-project <http://github.com/cpc/AFOCL>`_.
That project generates a directory-based database with a json-based metadata.
The database contains the bitstreams and firmware-files necessary to implement
the set of built-in kernels defined in the json-file.

The bistream database device will report all the built-in kernel implementations it can
find from the database in clGetDeviceInfo's CL_DEVICE_BUILT_IN_KERNELS-query.
The bitstream database device ("0xF") will automatically fetch bitstream from the database
and reconfigure the FPGA when user enqueues a built-in kernel for execution.
Therefore, the user does not need to handle the bitstream binaries themselves,
since the OpenCL implementation reconfigures the FPGA behind-the-scenes.

To use AFOCL-databases in PoCL, it is enough to point the Almaif-driver to the database
with the env variable::

  POCL_DEVICES=almaif POCL_ALMAIF0_PARAMETERS=0xF,<path/to/afocl-db> ./accel_example

At the moment, the public AlmaIF-driver and AFOCL include support only for
Xilinx Alveo U280 device, but adding support for other Alveo devices should be easy.
In the AFOCL publication the methodology was also demonstrated with Intel Arria 10,
but the code for that is not yet upstreamed. The driver is built to hide the
vendor-specific details from the end user, with different AlmaIFDevice backends
taking care of vendor-specific details.
For more information about the bitstream database,
see our :ref:`AFOCL-publication (2023) <publications>`.


Wrapping a new hardware component
---------------------------------

This section will walk through the addition of new implementation for an existing
built-in kernel.
The component can be any hardware component, as long as it supports the AlmaIF
interface specification described above.
The following section presents an example method of generating the accelerator
with HLS. However, other methods of generating the accelerator exists, the only
requirement is that it implements the AlmaIF specification as described above.


High-level synthesis
^^^^^^^^^^^^^^^^^^^^
Template for HLS-accelerator is in examples/accel/hls/poclAccel.cpp-file.
It can be generated with 'make hls_vecadd_bs', which generates the biststream
file to examples/accel/hls/bitstreams/. To enable the target, you need to add
VITIS_HLS_PATH and VIVADO_PATH as CMAKE variables that point to the directory
containing the 'vitis_hls' and 'vivado' binaries.

The build process of HLS accelerator consists of two parts:

1. Generating accelerator RTL from C++ input (With Vitis HLS using script
generate_hls_core.tcl)

2. Generating block design with the accelerator and block memory for AlmaIF
regions (With Vivado using script generate_hls_project.tcl)

To run the vector addition on HLS generated core, the bitstream needs to
be copied to PYNQ-Z1.
The generate_hls_project.tcl file sets the base address of the accelerator
to a physical address 0x40000000. This base address is given to PoCL through
an environment variable::

    export POCL_DEVICES=almaif
    export POCL_ALMAIF0_PARAMETERS="0x40000000,dummy,1,2"

The bitstream can be loaded on the FPGA with various ways. PYNQ-Z1 image
includes a python library to do it, which can be used with a following one-liner::

    sudo -E python -c "from pynq import Overlay;Overlay('examples/accel/hls/bitstreams/vecadd_1.bit')"

After that, it's possible to run the examples/accel/accel_example program.

OpenASIP and AlmaIF
-------------------

This section outlines the use of OpenASIP toolset in AlmaIF-driver.
The OpenASIP is used for two different roles in AlmaIF-based devices:

1. As a command processor to manage the separate accelerator IP components (built-in kernel implementations)
2. As a soft processor that OpenCL C kernels can be compiled for and executed on.

For more details, see `Efficient OpenCL system integration of non-blocking FPGA accelerators
<https://doi.org/10.1016/j.micpro.2023.104772>`_.

OpenASIP used with built-in kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this role, the OpenASIP core is used as a small control processor.
In principle, any soft processor could be used, but we chose to use OpenASIP for familiarity.
Replacing this component with a simpler RISC-V processor would be a very reasonable future change.

The command processor does not execute the kernel computation,
but just controls external accelerator that perform the kernel's functionality.
Since the built-in kernel functionality is fixed, the kernel firmware can also be fixed.
The firmware configures external accelerators (and DMA units), and then waits for their execution to complete.
Once done, it will mark the packet as completed.
In this role, the OpenASIP core is still responsible for managing the computation,
but due to the specialization afforded to it by only supporting a set of built-in kernels,
it is free to configure external accelerators to perform the computation.
For examples of such designs, see the `AFOCL repository <https://github.com/CPC/AFOCL/>`_.


OpenASIP used with compiled kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OpenASIP enables the design of specialized processors.
The instruction-set can be extended and specialized at a very fine granularity,
which allows the description of highly efficient ASIPs.
These processors are exposed as OpenCL Compute Devices via the AlmaIF-driver.
The same AlmaIF interface is used as for the command processor-role,
the difference to it is that in this mode, the kernel function itself is compiled for the
instruction set of the ASIP.
The AlmaIF-driver knows about the compiler support from the magic kernel ID *65535*

In OpenASIP's kernel compilation flow, the kernel is first compiled with LLVM's
backend-agnostic flow, using the generic *pocl_driver_build_source*.
The prebuilt built-in function library is linked in from *kernel/tce*
(found via the variable *device->kernellib_name*).

Then, a work-group function is generated that executes a single workgroup.
Similarly to the CPU target, "work-item loops" are generated that iterate over the work-items of a work-group while respecting the work-group barrier semantics.
This uses the same function *pocl_llvm_generate_workgroup_function* as the CPU backend,
but now just calling it from *lib/CL/devices/almaif/openasip/AlmaifCompileOpenasip.cc*.
The kernel arguments are loaded from the argument buffer, meaning that the kernel interface is always the same:
(pointer_to_argument_buffer, pointer_to_poclContext32b, global_id_x, global_id_y, global_id_z)
Output of this stage is still LLVM IR.

Finally, and as the only completely OpenASIP-specific step,
in the *lib/CL/devices/almaif/openasip/AlmaifCompileOpenasip.cc*-file,
the main-file (*tta_device_main.c*) is compiled by calling OpenASIP compiler *oacc*'s CLI.
The workgroup-function is statically linked in, such that the generic workgroup-function call in the main-file
correctly calls the workgroup function generated by the above steps.
The main-file is responsible for communicating with the AlmaIF-driver, and iterating over the workgroups
in the kernel dispatch packet.


Adding new co-processor types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The AlmaIF-driver is designed to generalize to different co-processors/accelerators than just those generated with OpenASIP
(however, no other processor types are yet tested to our knowledge).
To plug a different processor with different compilation flow to the driver,
only the compilation calls need to be redirected in the *lib/CL/devices/AlmaifCompile.cc*,
and the kernel function library may need to be regenerated in *lib/kernel*.

During runtime, the AlmaIF interface abstraction layer hides anything processor-specific,
as long as the new processor type implements the AlmaIF memory map as described at the start of this document.

Using this work
---------------

If you are utilizing, further developing or comparing to the AlmaIF driver of PoCL
in your academic work, please cite the following publication::

    @ARTICLE{leppanen2023,
      TITLE = {Efficient {OpenCL} system integration of non-blocking {FPGA} accelerators},
      JOURNAL = {Microprocessors and Microsystems},
      VOLUME = {97},
      PAGES = {104772},
      YEAR = {2023},
      ISSN = {0141-9331},
      DOI = {https://doi.org/10.1016/j.micpro.2023.104772},
      AUTHOR = {Topi Leppänen and Atro Lotvonen and Panagiotis Mousouliotis and Joonas Multanen and Georgios Keramidas and Pekka Jääskeläinen},
    }

.. _publications:

The other relevant publications::

    @INPROCEEDINGS{leppanen2024,
      AUTHOR={Leppänen, Topi and Multanen, Joonas and Leppänen, Leevi and Jääskeläinen, Pekka},
      TITLE = {Towards Efficient {OpenCL} Pipe Specification for Hardware Accelerators},
      YEAR = {2024},
      ISBN = {9798400717901},
      PUBLISHED = {Association for Computing Machinery},
      ADDRESS = {New York, NY, USA},
      URL = {https://doi.org/10.1145/3648115.3648128},
      DOI = {10.1145/3648115.3648128},
      BOOKTITLE = {Proceedings of the 12th International Workshop on OpenCL and SYCL},
      ARTICLENO = {2},
      NUMPAGES = {8},
      LOCATION = {Chicago, IL, USA},
      SERIES = {IWOCL '24}
    }

    @ARTICLE{afocl2023,
      AUTHOR={Leppänen, Topi and Multanen, Joonas and Leppänen, Leevi and Jääskeläinen, Pekka},
      TITLE={{AFOCL}: Portable {OpenCL} Programming of {FPGAs} via Automated Built-in Kernel Management},
      BOOKTITLE={2023 IEEE Nordic Circuits and Systems Conference ({NorCAS})},
      YEAR={2023},
      PAGES={1-7},
      DOI={10.1109/NorCAS58970.2023.10305457}
    }

    @ARTICLE{leppanen2022,
      AUTHOR={Leppänen, Topi and Lotvonen, Atro and Jääskeläinen, Pekka},
      TITLE={Cross-vendor programming abstraction for diverse heterogeneous platforms},
      JOURNAL={Frontiers in Computer Science},
      VOLUME={4},
      YEAR={2022},
      URL={https://www.frontiersin.org/articles/10.3389/fcomp.2022.945652},
      DOI={10.3389/fcomp.2022.945652},
      ISSN={2624-9898},
    }

    @INPROCEEDINGS{leppanen2021,
      AUTHOR={Leppänen, Topi and Mousouliotis, Panagiotis and Keramidas, Georgios and Multanen, Joonas and Jääskeläinen, Pekka},
      BOOKTITLE={2021 IEEE Nordic Circuits and Systems Conference (NorCAS)},
      TITLE={Unified OpenCL Integration Methodology for FPGA Designs},
      YEAR={2021},
      PAGES={1-7},
      DOI={10.1109/NorCAS53631.2021.9599861}
    }
