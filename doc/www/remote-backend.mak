<%!
        sub_page = "No-MPI OpenCL-Only Distributed Computing With PoCL-Remote"
%>

<%inherit file="basic_page.makt" />

<h1>September 4th, 2023: No-MPI OpenCL-Only Distributed Computing With PoCL-Remote</h1>

<p>PoCL now has a new backend that allows transparently
offloading OpenCL tasks to other nodes on the network, thus enabling
distributing compute without using MPI or similar APIs. Since the
standard OpenCL API suffices, compute offloading can be performed
identically whether using local or remote devices, which makes it
useful for selective/adaptive edge offloading and other use cases.</p>

<p>The driver is now considered ready for out-of-lab testing and has been
integrated to the <a href="http://code.portablecl.org">main</a> branch for the
upcoming v5.0 release. The work was primarily carried out by Michal Babej,
Jan Solanti and Pekka Jääskeläinen in the <a href="http://tuni.fi/cpc">Customized Parallel Computing</a>
group at <a href="http://tuni.fi/en">Tampere University</a> within multiple
European and national research projects.</p>

<p>PoCL-Remote follows a client-server architecture and is comprised of two
parts: The <code>remote</code> driver in the PoCL library and
<code>pocld</code> daemon.</p>

<p>The daemon can be run on any machine that is reachable via TCP/IP
and has an OpenCL implementation available. Any OpenCL implementation/driver/device
works on the server-side, not only PoCL-based drivers. The client library connects to the
configured daemon(s) and lists the OpenCL devices available on them in
<code>clGetDeviceIDs</code> just as if they were local to the client.</p>

<div style="text-align: center;">
<img src="img/pocl-remote.svg" border="0" style="width: 60%;" />
</div>

<p>
In contrast to existing networked offloading solutions for OpenCL, PoCL-Remote
makes use of PoCL's memory management infrastructure to keep track of memory
objects and only copy them around when actually necessary. When a memory
object migration is needed, the most efficient route for the transfer is
automatically chosen:

<ul>
<li>If both the source and destination device are part of the same native
OpenCL context on the machine running the daemon, the migration is delegated
to the underlying native driver.</li>
<li>If the devices are part of separate OpenCL platforms, the memory contents
are manually copied through host memory within the daemon.</li>
<li>For copies between two daemons, direct peer-to-peer connections are utilised
in order to minimize the amount of traffic to and from the client.</li>
<li>Should all else fail, the memory contents are downloaded into the client's
memory and uploaded to the destination from there.</li>
</ul>

Similarly, synchronisation between commands is done with OpenCL events within
the daemon and OpenCL user events that are signaled in peer-to-peer fashion
for dependencies between daemons.
</p>

<p>Early versions of PoCL-R or experiments using it have been presented at
<a href="https://doi.org/10.1145/3388333.3388642">IWOCL '20</a>,
<a href="https://doi.org/10.1007/978-3-031-04580-6_6">SAMOS 2021</a> and
<a href="http://doi.org/10.1145/3585341.3585376">IWOCL '23</a>.
There is also a full length journal article under review which describes the published
version (for example its RDMA support). A preprint of it is available in
<a href="https://doi.org/10.48550/arXiv.2309.00407">arXiv</a>.
</p>

<p>
More information, instructions for building and using the PoCL-Remote backend can be found in the
<a href="http://portablecl.org/docs/html/remote.html">user manual</a>.
</p>

<div style="text-align: center;">
<a href="img/pocl-remote-screenshot-august2023.png">
<img src="img/pocl-remote-screenshot-august2023.png" border="0" width="80%" style="vertical-align: middle; " />
</a>
</div>

<h2>Status and Maturity</h2>

<p>
The backend passes most of PoCL's basic test suite and has been successfully
used to run complex applications such as
<a href="https://github.com/ProjectPhysX/FluidX3D">FluidX3D</a> and various
computer vision and machine learning demos. The actual usable set of features
is naturally also dependent on the native driver controlled by the daemon.
</p>

<p>PoCL-R has been mainly tested within the research group but integrated to proper
demonstrators, thus can be considered TRL4-TRL5 in the EU scale.
</p>

<p>
In terms of performance there is of course a major penalty when having to transfer
buffers across the network. However this is mostly noticeable in buffer transfers
(Write-/Copy-/ReadBuffer and migrating a buffer from one device to another).
These can easily become a bottleneck for other drivers as well so designing
applications with that in mind is advisable in general.</p>
<p>
It is worth noting that PoCL-R will leave buffers resident on devices after
use, so unchanged buffers do not need to be transferred again on next use.
This means that static buffers such as neural network coefficients only need
to be uploaded once during launch and afterwards inference can be performed
repeatedly without this initial buffer transfer cost.
</p>
<p>
In multi-server setups the effects of server to server transfers can be
mitigated somewhat by building PoCL with RDMA support enabled, if RDMA is
supported by the networking hardware.
</p>

<h4>Known Limitations</h4>
<ul>
<li>There is no traffic encryption or user authentication on the daemon side, making PoCL-R currently
not suitable for use outside of closed private networks/clusters.</li>
<li>SPIR-V is not yet supported.</li>
<li>PoCL must be built with LOADABLE_DRIVERS=OFF, else initialisation of the remote backend fails.</li>
<li>While printf does somewhat work, it will likely behave differently from what applications expect.</li>
</ul>

<h2>Contributing</h2>

<p>We welcome any contributions in the form of good quality bug reports and pull requests,
but cannot commit to rapid support if the issue does not affect us due to a limited
number of developers working on the project. If you're interested in improving PoCL-R,
but aren't sure what to work on, please ask in the
<a href="https://lists.sourceforge.net/lists/listinfo/pocl-devel">mailing list</a> or the
<a href="https://github.com/pocl/pocl/discussions">discussion forum</a> for more information.
</p>
