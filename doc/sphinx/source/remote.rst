.. _remote-label:

=============
Remote Driver
=============

Background
----------

PoCL-Remote is an OpenCL driver which forwards OpenCL commands to
a remote server over network. The remote OpenCL devices are listed in
the local OpenCL platform device list and each each remote device can
be used like it was a local OpenCL device. The properties of a remote
device as queried via clGetDeviceInfo() mirror the remote physical
device's properties etc.

The key differentiating aim of PoCL-Remote to other similar remote
forwarding OpenCL implementations is that the server side component
is capable of autonomously scheduling commands based on event dependencies
as well as communicating in peer-to-peer fashion with other servers in order
to not tax the client's network connection with signaling and buffer
migrations between servers. The overall design focuses also on execution
latency, not only throughput, and is still targeted to also support
interactive low latency applications such as mixed reality.

On the client side PoCL-Remote is a PoCL driver backend that controls
remote devices via a control protocol and transparently exposes them to
applications as if they were local devices. On the server side, there is the
``pocld`` daemon which forwards requests to remote OpenCL implementations,
which can be either PoCL-based or proprietary.

The PoCL-Remote client has successfully been run on PC and Android devices
and with some tweaks (dubbed "Nano-PoCL") it has proven usable even in bare
metal firmware of low end embedded devices with limited toolchain support.
This kind of portability is a big reason why the PoCL-Remote client and core
PoCL components are written in plain C rather than C++. As a concrete example
of this embedded support is this `demo <https://doi.org/10.1145/3585341.3585376>`__
of the PoCL-Remote client running on the "AI Deck" add-on board of a Crazyflie
nano drone.

Overall execution latency is optimized by leaving server side command ordering
up to the underlying OpenCL driver via translated OpenCL event dependencies.
In multi-node scenarios, communication is optimized to avoid overloading the
client link by using peer-to-peer connections for inter-server communication.
The peer-to-peer connections can optionally be made more efficient by building
the daemon with RDMA support, which then gets used when migrating buffers
between nodes. Applications that deal with variable sized data can addionally
be written to make use of the ``cl_pocl_content_size`` extension that lets
the runtime know that buffers don't need to be transferred in full, which
PoCL-Remote can take advantage of both in client-server and server-to-server
communication.

More Information
----------------

PoCL-Remote has previously been showcased at
`IWOCL '20 <http://doi.org/10.1145/3388333.3388642>`__,
`SAMOS 2021 <https://doi.org/10.1007/978-3-031-04580-6_6>`__ and
`IWOCL '23 <https://doi.org/10.1145/3388333.3388642>`__.
There is also a full length journal article under review which describes the
published version (for example its RDMA support). A preprint of the article is
available in `arXiv <https://doi.org/10.48550/arXiv.2309.00407>`__.

In publications, when referring to PoCL-R, please cite the introductory paper with the following format::

  @InProceedings{10.1007/978-3-031-04580-6_6,
  author="Solanti, Jan and Babej, Michal and Ikkala, Julius and Malamal Vadakital, Vinod Kumar and J{\"a}{\"a}skel{\"a}inen, Pekka",
  title="PoCL-R: A Scalable Low Latency Distributed OpenCL Runtime",
  booktitle="Embedded Computer Systems: Architectures, Modeling, and Simulation",
  year="2022",
  pages="78--94",
  isbn="978-3-031-04580-6"
  }



Prerequisites
--------------

A machine with a working OpenCL implementation (can be any proprietary or
pocl-based) to act as a server, and another machine as a client which runs the
OpenCL application. Naturally, it's possible to use the same machine for both
for testing purposes.

The remote driver has been tested so far with AMD, NVidia and Intel OpenCL
implementations as well as with PoCL's CPU backend.

Current Status
--------------

The current version has been tested with various programs, such as:

  * the builtin tests of pocl
  * `FluidX3D <https://github.com/ProjectPhysX/FluidX3D/>`
  * `Assorted research demos <https://www.youtube.com/playlist?list=PLS9btrvnAmoKR7D_O6S6zetc7np0wYDAK>`

The image support in particular is quite new and very lightly tested.
The same applies for multi-device setup.

printf() support exists, but please note that the "standard output" (stdout) of the server-side OpenCL will be printed to the PoCL-D stdout. Furthermore, the stdout is shared with all client-server connection, so if multiple client
devices launch kernels with printf() simultaneously, the output order is
undefined.

Known Bugs/Issues/WiP
---------------------

* The "-I" option to clBuildProgram does not work
* clGetKernelWorkGroupInfo() can return incorrect information
* clCompileProgram() and clLinkProgram() API calls are broken
* clSetKernelArg() will not return a CL_INVALID_ARG_SIZE error if arg_size does not
  match the size of the data type for an argument that is not a memory object.
  Fixing this would require involving LLVM on the client side, as argument size
  information cannot be retrieved from OpenCL API runtime calls.
* There are some hardcoded limits (max devices per server)

.. _remote-issues-label:

Known Bugs/Issues in OpenCL Implementations
--------------------------------------------

* Nvidia's OpenCL implementation has a bug where clCreateKernelsInProgram()
  sometimes fails. This affects pocld since it uses clCreateKernelsInProgram
  in background.
  Workaround: always build programs for all devices. More details:
  https://devtalk.nvidia.com/default/topic/995251/clcreatekernelsinprogram-strangely-returns-cl_invalid_kernel_definition/

* ARM Mali OpenCL SDK (on Linux) and some Android OpenCL implementations fail
  to return anything for clGetKernelArgInfo calls,
  so it's unusable by itself (as backend for the proxy driver or for pocld).
  It is usable if pocl is built with at least two drivers (proxy/remote/pthread
  etc) of which at least one provides the build information.
  For OpenCL (pocl) users, it means all CL programs (to be used with Mali) must
  be built for at least two devices from two drivers,
  so the other driver provides the build information Mali does not.
  This Mali bug is only fixable by using another driver, or writing some extra
  compilation/parsing step, because without argument metadata
  it's impossible to tell if an argument to clSetKernelArg is a pointer or an integer.

.. _remote-how-to-build-label:

How to Build
-------------

First you need to install the build dependencies.
These are listed in :ref:`pocl-install` or alternatively you can
also take a look at the Dockerfiles in ``tools/docker``.

Note that you do not need LLVM if you want to only use the PoCL-Remote driver
to control server side devices for which a separate OpenCL driver exists.

These steps build pocl **without** the CPU driver (= with remote driver only).

If you want to use event tracing/profiling, scroll below as it requires
installing some extra packages before building pocl.

To build the remote *client*::

    mkdir build; cd build;
    cmake -DENABLE_HOST_CPU_DEVICES=0 -DENABLE_LLVM=0 -DENABLE_ICD=1 -DENABLE_REMOTE_CLIENT=1 ..
    make -j$(nproc)

This should produce **lib/CL/libpocl.so** (the client library that implements
the OpenCL runtime API).

To build the remote *server*::

    mkdir build; cd build;
    cmake ../pocld

This should produce **pocld** (the server executable). If you need both the
client library and server binary on the same machine you can alternatively add
``-DENABLE_REMOTE_SERVER=1`` to the cmake flags in the client build to get
**pocld/pocld** generated in the same build directory.

On the server, make sure that "clinfo" lists at least one OpenCL device, then
run the server command::

    ./pocld -a <IP ADDRESS> -p <PORT>

Run ``pocld --help`` to list all options.
Note that pocld will listen on three ports, ``PORT``, ``PORT+1`` and ``PORT+2``.
You can tune the amount of messages produced with the environment variable
"POCLD_LOGLEVEL" before running pocld or the ``--log_filter`` command line option.
The default log level is "err".
Accepted values are: debug, info, warn, err, critical, off.

On the client, export these environment variables (the first one must be done
in the pocl remote-client build directory) ::

    export OCL_ICD_VENDORS=$PWD/ocl-vendors/pocl-tests.icd
    export POCL_DEVICES=remote
    export POCL_REMOTE0_PARAMETERS='<IP ADDRESS>:<PORT>/<DEVICE ID>#<PEER ADDRESS>'

``IP ADDRESS`` and ``PORT`` are self-explanatory, ``DEVICE ID`` is the index of
the device on the server. ``PORT`` is the lower port number assigned to the server.
Indices are from zero to N-1 where N is the total number of devices across
all platforms on the server.
The index is the order in which pocld lists the devices in the OpenCL platform it uses.
This is the same order than can be displayed  by "clinfo".

``PEER ADDRESS`` (and the preceding '#' sign) is optional and is used for server-server
communication when there are multiple remote servers that have a public IP and
a private IP on a fast internal network. If a separate peer address is not given,
server-server communication will use ``IP ADDRESS`` just like client-server communications.

To "smoke test" that the distributed setup works, you can use the clinfo
tool, which should now list the remote devices also::

  $ clinfo | grep PoCL-Remote
  Device Version OpenCL 1.2 CUDA HSTR: PoCL-Remote 123.456.789.123:1000/0

Then you can run the simple dot product in example1::

  $ cd examples/example1
  $ ./example1
  (0.000000, 0.000000, 0.000000, 0.000000) . (0.000000, 0.000000, 0.000000, 0.000000) = 0.000000
  (1.000000, 1.000000, 1.000000, 1.000000) . (1.000000, 1.000000, 1.000000, 1.000000) = 4.000000
  (2.000000, 2.000000, 2.000000, 2.000000) . (2.000000, 2.000000, 2.000000, 2.000000) = 16.000000
  (3.000000, 3.000000, 3.000000, 3.000000) . (3.000000, 3.000000, 3.000000, 3.000000) = 36.000000
  OK

By default pocld will destroy client contexts when the client connection is lost to avoid
accumulating stale resources, as there is currently no way for clients to signal a controlled
shutdown. Setting ``POCLD_ALLOW_CLIENT_RECONNECT=1`` in pocld's environment disables this behavior
and allows clients to reconnect to their existing session.

Android Build (Client Only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :ref:`android-label` on how to do this.

Windows build (server only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only the remote server can be built on Windows.

Note that later versions of the Khronos ICD have problems compiling with MSYS's GCC.
Git version from before Oct 3 2017 is known to compile. This is because apparently
MS decided to forbid the graphics drivers from writing to some Windows registry entries in
some later versions of Windows 10, and the Khronos ICD couldn't find the list of OpenCL
implementations in the same Windows registry entries as before, so the Khronos ICD
gained some new code which uses new & awesome way to enumerate the OpenCL implementations
- but this code does not compile under MSYS.

Possible workaround (untested): manually add OpenCL implementations to old registry paths
(example is in https://github.com/KhronosGroup/OpenCL-ICD-Loader/blob/master/README.txt).

First, install MSYS2 from https://www.msys2.org/; only tested the x86_64
version has been tested. Follow the instructions to update all MSYS2 packages.
Then install CMake, GCC and friends::

    pacman -S cmake make gcc patch

Download Khronos ICD loader from https://github.com/KhronosGroup/OpenCL-ICD-Loader
and Khronos OpenCL headers from https://github.com/KhronosGroup/OpenCL-Headers

* put the PoCL-Remote and ICD loader sources in ``$HOME/pocl`` and ``$HOME/ICD``

* patch the ICD loader with ``<pocl_sources>/tools/patches/windows_khronos_icd.patch``

* from the OpenCL headers, take the "CL" directory and put it into ``$HOME/ICD/inc/``

To build the ICD loader library::

    cd $HOME/ICD
    mkdir b
    cd b
    cmake -DCMAKE_SYSTEM_NAME=Windows -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
    make -j4

To build pocl::

    cd $HOME/pocl
    mkdir b
    cd b
    cmake -DCMAKE_SYSTEM_NAME=MSYS -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLIBOPENCL=$HOME/ICD/b ../pocld
    make -j8

This will result in pocld.exe in ``$HOME/pocl/b`` directory. This requires a few DLLs:

*  ``msys-2.0.dll  msys-gcc_s-seh-1.dll  msys-stdc++-6.dll`` from ``$MSYS/usr/bin``
*  ``libOpenCL.dll``  from ``$HOME/ICD/b/bin``

These can be copied into the same directory as pocld.exe. The binary is quite large;
if debugging symbols are not needed, ``strip.exe`` command works as on Linux.


Event Tracing
-------------

It's possible to use LTTNG to trace both the server and the client library.
To install lttng on Ubuntu, run this as root / sudo::

     apt install lttng-tools lttng-modules-dkms liblttng-ust-dev liblttng-ctl-dev

You must now run cmake with ``-DENABLE_LTTNG=YES`` and rebuild pocl.


Using LTTNG
~~~~~~~~~~~

First, check that lttng-sessiond is running. If it's not, start it::

    lttng-sessiond --daemonize

Then launch the pocld server / OpenCL client app / both. Note that LTTNG sessiond
registers userspace events only AFTER the program has started and loaded the lttng
library. This is OK for the server, but a problem for the client, since many OpenCL
clients immediately start execution. There is an environment variable to help
with this: ``POCL_STARTUP_DELAY=<N>`` where N is the delay in seconds. You'll also
need to enable tracing for the client application with ``POCL_TRACING`` so::

    POCL_STARTUP_DELAY=<N> POCL_TRACING=lttng application [arguments]

To create a LTTNG session::

    lttng create <session-name>

Now list the userspace events::

    lttng list --userspace

You should see "pocl_trace:" events for the client, and "pocld_trace:" events
for the server. Enable the ones you care about, or all::

    lttng enable-event --userspace pocl_trace:*
    lttng enable-event --userspace pocld_trace:*

You can trace any number/combination of events, and also kernelspace events
(probably requires root).

To start tracing::

    lttng start

To stop tracing::

    lttng stop

To destroy session (this merely destroys session in the daemon, does not delete data)::

    lttng destroy

Now you have tracing data in ``$HOME/lttng-traces/<session-name>-<date>-<time>``
directory. You can view them using "babeltrace" tool, or eclipse-based "trace compass",
or possibly other tools.

Viewing Traces
~~~~~~~~~~~~~~
For this you'll need chrome/chromium, python and babeltrace installed.
START_TIME and END_TIME are optional - they define a time
slice to pick from the log. If not defined, the entire
trace log will be converted to JSON (Warning : large logs can be a problem).

To convert binary LTTNG trace format to text, then to JSON, run::

    cd $HOME/lttng-traces/<session-name>-<date>-<time>
    babeltrace --clock-seconds . >/tmp/trace.text
    ./<pocl_source>/tools/scripts/babel_parse.py -o OUTPUT_FILE /tmp/trace.txt

To view the JSON trace, open Google Chrome/Chromium, type ``chrome://tracing``,
click Load, and find ``/tmp/trace.json``.

Remote and Local Traces
~~~~~~~~~~~~~~~~~~~~~~~

It's possible to combine local and remote tracing outputs to get a full view
of what's happening over network. Note that this requires LTTNG installed on
servers as well, plus it requires very precisely synchronized time between all
involved machines (1 microsecond or so should be good enough). The simplest
way to achieve that seems to have all machines are equipped with an Intel NIC,
and then setup PTP (Precision time protocol). Note that PTP requires hardware
support from *every* network device in path to achieve sub-microsecond precision.

Distributed SYCL Execution Using PoCL-R
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

oneAPI DPC++ can be used to distribute SYCL applications using PoCL-R. Notably,
only buffer-based memory management (not USM) works currently. To test it out,
build DPC++ as instructed in the `Getting Started with oneAPI DPC++ <https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md>`_ document.

Then, build a SYCL program of your choice as instructed in the DPC++ documentation and launch it like any OpenCL program with the PoCL-D running. Just ensure you use an OpenCL device in DPC++ and it points to a PoCL-R device::

  export POCL_DEVICES=remote
  export POCL_REMOTE0_PARAMETERS=localhost:7777/0
  export ONEAPI_DEVICE_SELECTOR=opencl:0
  ./simple-sycl-app
  The results are correct!

.. _remote-discovery-label:

Dynamic Device Management and Network Discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Dynamic Device Management and Network Discovery adds the ability to find and
manage remote servers in both Local Area Networks (LAN) and Wide Area Networks
(WAN). Through discovery remote servers can be discovered and their OpenCL
devices can be dynamically added to the platform.

**Network Discovery Mechanisms**

* mDNS (Multicast-DNS): Utilizes the Avahi library for local network discovery.

* Unicast-DNS-SD: For server discovery in specified domains, also using Avahi.

* DHT (Distributed Hash Table): uses the OpenDHT library to establish or join a
  DHT network, where servers can publish their information and clients can
  listen for server announcements.

**Enumerations for Device Info**

The following new enumerations have been added to the clGetDeviceInfo():

* ``CL_DEVICE_REMOTE_SERVER_IP_POCL``: Retrieves the IP address of the remote server.

* ``CL_DEVICE_REMOTE_SERVER_PORT_POCL``: Retrieves the port of the remote server.

**Discovery Build and Environment Variables**

To build the remote server or client with discovery follow the steps in
:ref:`remote-how-to-build`, in addition to those, follow the following build
options:

Remote Client:

* ``-DENABLE_REMOTE_DISCOVERY_AVAHI``: Enable mDNS and unicast-DNS-SD discovery
  via Avahi.

* ``-DENABLE_REMOTE_DISCOVERY_DHT``: Enable DHT-based discovery.

* ``-DENABLE_REMOTE_DISCOVERY_ANDROID``: Enables discovery for Android. The helper
  function ``pocl_remote_discovery_add_server()`` in include/pocl_remote.h can be
  used to dynamically add remote servers and its devices through the remote driver.

Remote Server:

* ``-DENABLE_REMOTE_ADVERTISEMENT_AVAHI``: Enable mDNS advertisement via Avahi.

* ``-DENABLE_REMOTE_ADVERTISEMENT_DHT``: Enable DHT-based advertisement.


The following environment variables are introduced to control and customize
discovery and advertisement:

* ``POCL_DISCOVERY`` (bool): Located in devices.h and utilized in devices.c.
  This variable allows users to enable or disable device discovery entirely.

* ``POCL_REMOTE_SEARCH_DOMAINS`` (string): Introduced in network_discovery.h
  and used in network_discovery.c. This variable specifies DNS domains for
  unicast-DNS-SD queries. If set to ``NULL``, discovery is limited to the
  local network. For example, setting
  ``POCL_REMOTE_SEARCH_DOMAINS="example.com"`` enables discovery in both the
  LAN and the specified domain.

* ``POCL_REMOTE_DHT_PORT`` (int): Introduced in network_discovery.h and
  dht_advertise.h, and used in network_discovery.c and
  dht_advertise.c. This variable allows users to specify a port on which the
  DHT node will operate. Set to 4222 by default.

* ``POCL_REMOTE_DHT_BOOTSTRAP`` (string): Introduced in network_discovery.h and
  dht_advertise.h, and utilized in network_discovery.c and
  dht_advertise.c. This variable specifies a bootstrap node to connect to
  an existing DHT network. Set to ``NULL`` by default.

* ``POCL_REMOTE_DHT_KEY`` (string): Introduced in network_discovery.h and
  dht_advertise.h, and used in network_discovery.c and
  dht_advertise.c. This variable provides a common key for server and
  client nodes to use when publishing or listening for server details within
  the DHT network. Set to "poclremoteservernetwork" by default.

Run remote client with these environment variables in addition to the ones
specified in :ref:`remote-how-to-build`.

**Examples**

Discovery in client using mDNS::

    mkdir build; cd build;
    cmake -DENABLE_HOST_CPU_DEVICES=0 -DENABLE_LLVM=0 -DENABLE_ICD=1 -DENABLE_REMOTE_CLIENT=1 -DENABLE_REMOTE_DISCOVERY_AVAHI=1 ..
    make -j$(nproc)

Environment variables::

    export POCL_DISCOVERY=1
    export OCL_ICD_VENDORS=$PWD/ocl-vendors/pocl-tests.icd
    export POCL_DEVICES=remote
    export POCL_REMOTE0_PARAMETERS='<IP ADDRESS>:<PORT>/<DEVICE ID>#<PEER ADDRESS>'

Discovery in client using unicast-DNS-SD, build is same as mDNS::

    export POCL_DISCOVERY=1
    export POCL_REMOTE_SEARCH_DOMAINS="domain_name.com"
    export OCL_ICD_VENDORS=$PWD/ocl-vendors/pocl-tests.icd
    export POCL_DEVICES=remote
    export POCL_REMOTE0_PARAMETERS='<IP ADDRESS>:<PORT>/<DEVICE ID>#<PEER ADDRESS>'

Discovery in client using DHT::

    mkdir build; cd build;
    cmake -DENABLE_HOST_CPU_DEVICES=0 -DENABLE_LLVM=0 -DENABLE_ICD=1 -DENABLE_REMOTE_CLIENT=1 -DENABLE_REMOTE_DISCOVERY_DHT=1 ..
    make -j$(nproc)

Environment variables::

    export POCL_DISCOVERY=1
    export POCL_REMOTE_DHT_PORT=<DHT PORT>
    export POCL_REMOTE_DHT_BOOTSTRAP=<BOOTSTRAP IP>
    export POCL_REMOTE_DHT_KEY=<COMMON KEY>
    export OCL_ICD_VENDORS=$PWD/ocl-vendors/pocl-tests.icd
    export POCL_DEVICES=remote
    export POCL_REMOTE0_PARAMETERS='<IP ADDRESS>:<PORT>/<DEVICE ID>#<PEER ADDRESS>'

For remote server advertisement using mDNS use ``-DENABLE_REMOTE_ADVERTISEMENT_AVAHI=YES``

To advertise using unicast-DNS-SD, you would need to add DNS records for the
remote-server in the name server of the domain which you are using. The
following example shows how that can be done for a domain "example.com".

Add PTR records:

* ``lb._dns-sd._udp.example.com`` points to ``example.com`` - This sets legacy
  browse domain to "example.com".

* ``b._dns-sd._udp.example.com`` points to ``example.com`` - Sets browse domain
  to "example.com".

* ``_services._dns-sd._udp.example.com`` points to ``pocl._tcp.example.com`` -
  Adds the service type "pocl" in the seacrh domain.

* ``_pocl._tcp.example.com`` points to ``unique_32_char_string._pocl._tcp.example.com``
  - Adds the service instance called "unique_32_char_string" for service type "pocl".

Add SRV records:

* ``unique_32_char_string._pocl._tcp.example.com`` with your choice of priority,
  weight, TTL, port, points to ``unique_32_char_string.example.com``

Add A/AAAA records:

* ``unique_32_char_string.example.com`` points to IP

Add TXT records:

* ``unique_32_char_string._pocl._tcp.example.com`` has content "11" (The text
  record is used by the client to determine the type of the device and total
  number of devices. 0 - CPU, 1 - GPU, 2 - Accelerator, 4 - Custom. Hence, "11"
  denotes "gpu gpu")


For remote server advertisement using DHT use ``-DENABLE_REMOTE_ADVERTISEMENT_DHT=YES``
Environment variables::

    export POCL_REMOTE_DHT_PORT=<DHT PORT>
    export POCL_REMOTE_DHT_BOOTSTRAP=<BOOTSTRAP IP>
    export POCL_REMOTE_DHT_KEY=<COMMON KEY>


Implementation Notes
--------------------

Although PoCL-R has been in development for several years it has only seen
limited testing outside the original lab since it has not been publicly available.

* The current implementation is asynchronous across multiple command queues, but
  blocking within a command queue. In other words, multiple CQs progress
  in parallel, but each enqueued command has an implicit clFinish() and
  there is network communication before the next command is launched.
  This is a key bottleneck that will be resolved in a future version.

* The old SPIR 1.2/2.0 are not supported and the respective extension is masked out from
  remote devices' extension lists by pocld.

* There is no authentication or encryption whatsoever of network traffic. Don't
  use PoCL-Remote outside of closed private networks.

* Synchronous commands (like clCreate* / clBuildProgram etc) are run in a separate thread.

* There are two separate network connections (TCP ports) used by the driver;
  one is for large transfers (like buffer transfers) and the other for fast / small
  transfers (clEnqueueNDRange).

* The client and server CPUs must be both little-endian, but may differ in
  pointer size, although things may break in unexpected ways if using images
  or buffers larger than 4G.
