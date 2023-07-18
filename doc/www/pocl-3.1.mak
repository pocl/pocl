<%!
        sub_page = "Portable Computing Language (pocl) v3.1 released"
%>
<%inherit file="basic_page.makt" />

<h1>December 5, 2022: pocl v3.1 released</h1>

<h2>Release Highlights</h2>

<ul>
<li>Support for Clang/LLVM 15.0
<li>Much improved <a href="http://portablecl.org/docs/html/opencl_status.html#spir-and-spir-v-support">SPIR-V support</a>
for CPU and CUDA drivers
<li>Major <a href="/almaif.html">rework of the custom device driver</a>. The driver was formerly named Accel, now AlmaIF.
<li>Various improvements to the work-in-progress <a href="http://portablecl.org/docs/html/vulkan.html">Vulkan driver</a>
<li><a href="http://portablecl.org/docs/html/extensions.html#cl-khr-command-buffer">Basic implementation</a> of cl_khr_command_buffer
</ul>

<p>You can <a href="download.html">download it from here</a>.</p>

<h2>Notes</h2>

<ul>

<li>Please note that there's an official PoCL
<a href="http://portablecl.org/docs/html/development.html#maintenance-policy">maintenance policy</a> in
place. This text describes the policy and how you can get your favourite
project that uses OpenCL to remain regression free in the future PoCL
releases.

<li>We are looking for ARM CPU and RISC-V CPU maintainers (and any other
target of your interest). If you are interested in ensuring PoCL stays
stable for these processor architectures in the future, please contact us.

<li>The dedicated TTASim driver will be deprecated once the AlmaIF driver
reaches feature parity with it.

<li>Support for LLVM versions below 10 will be removed in the next
release. If you rely on these versions, please let us know.

</ul>

<h2>Acknowledgments</h2>

<p>
<a href="http://tuni.fi/cpc">Customized Parallel Computing (CPC)</a> research group of
<a href="http://tuni.fi">Tampere University</a>,
Finland leads the development of PoCL on the side and for the needs of
their research projects. This project has received funding from European
Union's Horizon 2020 research and innovation programme under Grant
Agreement No 871738 (CPSoSaware), Academy of Finland (decision #331344) and
Business Finland's AISA project. The financial support is very much
appreciated -- it keeps this open source project going!</p>

