CLPeak
------

Currently (Dec 2017) does not work. First, there's a global memory size
detection bug in CLPeak which makes it fail on all OpenCL calls (this
can be workarounded by using POCL_MEMORY_LIMIT=1). Second, compilation
takes forever - this can't be fixed in pocl and needs to be fixed in
either CLPeak or LLVM. CLPeak sources use recursive macros to create
a giant stream of instructions. Certain optimization passes
in LLVM seem to explode exponentially on this code. The second
consequence of giant instruction stream is, it easily overflows the
instruction caches of a CPU, therefore CLPeak results are highly
dependent on whether the compiler manages to fit the code into icache,
and as such are not a reliable measure of peak device FLOPS.

Luxmark
-------

* Using the binary downloaded from www.luxmark.info might lead to pocl
  abort on creating cache directory. This is not a bug in Pocl, it's a
  consequence of the two programs (pocl & luxmark) having been compiled
  with different libstdc++. Using a distribution packaged Luxmark
  fixes this problem.

* It's recommended to remove luxmark cache (~/.config/luxrender.net)
  after updating pocl version.

* There's another bug (http://www.luxrender.net/mantis/view.php?id=1640)
  - it crashes after compiling kernels, because it doesn't recognize
  an OpenCL device. This requires editing scenes/<name>/render.cfg,
  you must add ``opencl.cpu.use = 0`` and ``film.opencl.device = 0``

* Microphone and Luxball scenes work with LLVM 5 and later.
  Hotel scene fails to compile with LLVM 5 but works with LLVM 6.
