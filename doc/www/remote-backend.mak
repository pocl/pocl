<%!
        sub_page = "Networked Offloading and Distribution"
%>

<%inherit file="basic_page.makt" />

<h1>August 2023: Networked Offloading and Distribution Driver Added</h1>

<p>
PoCL now has support for offloading and distributing OpenCL tasks across a
network. The work was primarily carried out by Michal Babej, Jan Solanti and
Pekka Jääskeläinen from the Customized Parallel Computing group at Tampere
University. This functionality is comprised of two parts: the new <code>remote</code>
backend in the PoCL library and the <code>pocld</code> daemon.
</p>

<p>
The daemon can be run on any machine that is reachable via TCP/IP
and has an OpenCL implementation available. The client library connects to the
configured daemon(s) and lists the OpenCL devices available on them in
clGetDeviceIDs as if they were local to the client.
</p>

<img src="img/pocl-remote.svg" border="0" style="width: 80%; vertical-align: middle;" />

<p>
In contrast to existing networked offloading solutions for OpenCL, PoCL-Remote
makes use of PoCL's memory management infrastructure to keep track of memory
objects and only copy them around when actually necessary. When a memory
object migration is needed, the most efficient route for the transfer is
automatically chosen:
<ul>
<li>if both the source and destination device are part of the same native
OpenCL context on the machine running the daemon, the migration is delegated
to the underlying native driver.</li>
<li>if the devices are part of separate OpenCL platforms, the memory contents
are manually copied through host memory within the daemon.</li>
<li>for copies between two daemons, direct peer-to-peer connections are utilised
in order to minimize the amount of traffic to and from the client.</li>
<li>should all else fail, the memory contents are downloaded into the client's
memory and uploaded to the destination from there.</li>
</ul>
Similarly, synchronisation between commands is done with OpenCL events within
the daemon and OpenCL user events that are signaled in peer-to-peer fashion
for dependencies between daemons.
</p>

<p>The backend has been previously showcased at
<a href="https://doi.org/10.1145/3388333.3388642">IWOCL '20</a>,
<a href="https://doi.org/10.1007/978-3-031-04580-6_6">SAMOS 2021</a> and
<a href="http://doi.org/10.1145/3585341.3585376">IWOCL '23</a>.
There is also a full length journal article under review which describes the published
version (for example its RDMA support). A preprint of it is available in
<a href="https://doi.org/10.48550/arXiv.2309.00407">arXiv</a>.
</p>

<p>The full documentation can be found at
<a href="http://portablecl.org/docs/html/remote.html">http://portablecl.org/docs/html/remote.html</a>.
</p>

<img src="img/pocl-remote-screenshot-august2023.png" border="0" style="vertical-align: middle;" />

<h2>Status</h2>

<p>
Instructions for building and using the PoCL-Remote backend can be found in the
<a href="http://portablecl.org/docs/html/remote.html">user manual</a>.
</p>

<p>
This backend passes most of PoCL's builtin tests and has been successfully
used to run applications such as
<a href="https://github.com/ProjectPhysX/FluidX3D">FluidX3D</a> and various
computer vision and machine learning demos. The actual usable set of features
is naturally also dependent on the native driver used by the daemon.
</p>

<p>
In terms of performance there is of course a penalty from having to transfer
data across the network. However this is mostly noticeable in buffer transfers
(Write-/Copy-/ReadBuffer and migrating a buffer from one device to another).
These can easily become a bottleneck for other drivers as well so designing
applications with that in mind is advisable in general.

It is worth noting that PoCL-R will leave buffers resident on devices after
use, so unchanged buffers do not need to be transferred again on next use.
This means that static buffers such as neural network coefficients only need
to be uploaded once during launch and afterwards inference can be performed
repeatedly without this initial buffer transfer cost.

In multi-server setups the effects of server to server transfers can be
mitigated somewhat by building PoCL with RDMA support enabled, if RDMA is
supported by the networking hardware.
</p>

<h4>Known Limitations (At The Time of Writing)</h4>
<ul>
<li>There is no traffic encryption or authentication, making PoCL-R unsuitable for use outside of closed private networks</li>
<li>SPIR-V is not supported</li>
<li>PoCL must be built with LOADABLE_DRIVERS=OFF, else initialisation of the remote backend fails</li>
<li>While printf does somewhat work, it will likely behave differently from what applications expect</li>
</ul>

<h2>Contributing</h2>

<p>
We welcome any contributions in the form of bug reports and pull requests.
In particular, we are keen to see contributions that fill in the remaining
functionality, as well as performance improvements.
If you're interested in helping out but aren't sure what to work on, ask on the
PoCL mailing list or the PoCL discussions forum on GitHub for more information.
</p>
