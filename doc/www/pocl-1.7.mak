<%!
        sub_page = "Portable Computing Language (pocl) v1.7 released"
%>
<%inherit file="basic_page.makt" />

<h1>April NN, 2021: pocl v1.7 released</h2>

<h2>Release Highlights</h2>

<h3>Support for Clang/LLVM 12</h3>

<p>LLVM 11 remains to be officially supported, but versions down to 6.0 might
still work.</p>

<h3> Improved support for cross-compiling </h3>

<h3> Improved support for SPIR-V binaries when using CPU device </h3>

<h3> Changed default target OpenCL version to 3.0 </h3>

<p>
Note: PoCL still lacks support for most new features in OpenCL 3.0 and calling
such features might result in undefined behaviour.
</p>

<p>
Implemented OpenCL 3.0 features: <b>clGetDeviceInfo queries</b>
<ul>
<li> CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES (Minimal implementation) </li>
<li> CL_DEVICE_ATOMIC_FENCE_CAPABILITIES (Minimal implementation) </li>
</ul>
</p>

<h2>Acknowledgments</h2>

<p>
The CUDA improvements, PowerPC support and packaging support described in this post were made by
Isuru Fernando and Matt Wala with assistance from Nick Christensen, and Andreas Klöckner,
all part of the Department of Computer Science at the University of Illinois at Urbana-Champaign.
The work was partially supported through awards OAC-1931577 and SHF-1911019 from the
US National Science Foundation, as well as award DE-NA0003963 from the US Department of Energy.
</p>

<p>
<a href="http://tuni.fi/cpc">Customized Parallel Computing (CPC)</a> research group of
<a href="http://tuni.fi">Tampere University</a>,
Finland leads the development of PoCL on the side and for the needs of
their research projects. This project has received funding from the ECSEL
Joint Undertaking (JU) under grant agreement No 783162 (FitOptiVis).
The JU receives support from the European Union’s Horizon 2020 research and
innovation programme and Netherlands, Czech Republic, Finland, Spain, Italy.
It was also supported by European Union's Horizon 2020 research and innovation
programme under Grant Agreement No 871738 (CPSoSaware) and
<a href="http://hsafoundation.com">HSA Foundation</a>.
</p>

<p><a href="download.html">Download</a>.</p>
