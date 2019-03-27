<%!
        sub_page = "Portable Computing Language (pocl) v1.3 released"
%>
<%inherit file="basic_page.makt" />

<h1>April Xth 2019: pocl v1.3 released</h2>

<h2>Release Highlights</h2>

<ul>
<li>Support for LLVM/Clang 8.0 and 7.0</li>
<li>Support ICD on OSX</li>
</ul>

<h2>Notes</h2>
<ul>
<li>Support code for older than LLVM 6.0 will be removed in the beginning of
  the next release cycle to clean up the code base. If you care need support
  for older LLVM versions in the future pocl releases and wish to maintain
  it (run a buildbot and fix issues), let us know! </li>
<li>Support for Vecmathlib has been removed.</li>
</ul>
</p>

<p>Please note that there's an official pocl maintenance policy in place.
<a href="http://portablecl.org/docs/html/maintainer-policy.html">This text</a>
describes the policy and how you can get your favourite project
that uses OpenCL to remain regression free in the future pocl releases.</p>

<h2>Acknowledgements</h2>

<p>
Most of the code that landed to the pocl code base during this release
cycle was produced for the needs of research projects funded by various
sources. Customized Parallel Computing research group of Tampere
University, Finland likes to thank the Academy of Finland (funding
decision 297548), Business Finland (FiDiPro project StreamPro,
1846/31/2014), ECSEL JU project FitOptiVis (project number 783162) and
HSA Foundation for funding most of the development work in this release.
Much appreciated!
</p>

<p><a href="download.html">Download</a>.</p>
