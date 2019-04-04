=======================
running Pocl in Docker
=======================

Install Docker
----------------

* install docker for your distribution
* start the docker daemon
* make sure you have enough space (default location is usually ``/var/lib/docker``,
  required storage for standard pocl build is about 1.5 GB per container,
  and more than 10GB for TCE/PHSA builds)

start Pocl container
----------------------

* create an empty directory <D>
* copy Dockerfile of your choice (any file from tools/docker/) to ``<D>/Dockerfile``
* ``cd <D> ; sudo docker build -t TAG .`` .. where TAG is a name you can choose for the build.
* ``sudo docker run -t TAG``
* this will by default use master branch of pocl git; to use a different branch/commit,
  run docker build with ``--build-arg GIT_COMMIT=<branch/commit>``


Dockerfiles
------------
Many are split up into two or three build stages, in which you must build all
but last stage with a proper tag (grep the dockerfiles for "FROM <TAG>").
Dockerfiles are named according to what they build:

* `base`: the first stage in multi-stage Docker builds. Downloads dependencies
   and clones pocl git repo but does nothing more.
* `default`: builds pocl, then runs the internal tests from build dir.
   Uses latest release of a distribution, with whatever is the default version of LLVM.
* `<release>`: same as above, except uses specific release and specific LLVM version
  (the latest available in that release).
* `default.32bit`: same as default but sets up i386 environment
* `test_install`: builds & installs pocl into system path, then runs the internal tests
* `distro`: does a distribution-friendly build (enables runtime detection of CPU, etc)

Some additional notes:
* Arch Dockerfiles are split up into two-stage builds
* some (not all) Ubuntu Dockerfiles are split up into multi-stage builds
* RHEL 7 was added, it's using unofficial LLVM 5.0 binaries from copr, since the official RHEL 7 LLVM is too old.
* TCE added - TCE is built using three stages (LLVM, TCE, pocl)
* PHSA added - also built using three stages (LLVM, PHSA runtime, pocl)
