=======================
running Pocl in Docker
=======================

Install Docker
----------------

* install docker for your distribution
* start the docker daemon
* make sure you have enough space (default location is usually ``/var/lib/docker``,
  required storage for pocl is about 1.5 GB per container)

start Pocl container
----------------------

* create an empty directory <D>
* copy Dockerfile of your choice (any file from tools/docker/) to ``<D>/Dockerfile``
* ``cd <D> ; sudo docker build -t TAG .`` .. where TAG is a name you can choose for the build.
* ``sudo docker run -t TAG``
* this will by default use master branch of pocl git; to use a different branch/commit,
  run docker build with ``--build-arg GIT_COMMIT=<branch/commit>``


Dockerfiles:
--------------
* `default`: builds pocl, then runs the internal tests from build dir.
   Uses latest release of a distribution, with whatever is the default version of LLVM.
* `<release>`: same as above, except uses specific release and specific LLVM version
  (the latest available in that release).
* `default.32bit`: same as default but sets up i386 environment
* `test_install`: builds & installs pocl into system path, then runs the internal tests
* `distro`: does a distribution-friendly build (enables runtime detection of CPU, etc)
