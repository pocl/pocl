Information for Release Managers
================================

We aim to make a new release according to the Clang/LLVM release schedule.


For each release, a release manager is assigned. Release manager is responsible
for creating and uploading new release candidate tar balls and requesting for
testers from different platforms. After a release candidate round with
success reports and no failure reports, a release is published.

See the :ref:`maintenance-policy` for the current release criteria.

A checklist and hints for testing and making a release successfully:

* Check that CHANGES has the most interesting updates done during the release
  cycle. Add missing notable changes from git log.

* Update the release notes in *doc/notes-VERNUM.txt*.

* Create a single commit in master branch: change the version to the
  release one (without -pre), in all relevant places (CHANGES, docs,
  CMakeLists.txt, etc); update the .so version (if required);
  check that supported LLVM versions in cmake/LLVM.cmake are correct.
  Create the release branch from this commit and push it to github.

* In the master branch, create a new commit: increase version
  number (with -pre) in all relevant places; update the .so version;
  increase the supported LLVM versions in cmake/LLVM.cmake.
  Commit, push master to github. Now development can go on in master
  while the release branch is being stabilized.

* The previous two steps ensure that merge-base of release & master is
  the start of release branch, which ensures that merging release
  to the master will not screw up the version numbers in the master.
  Bugs which need to be fixed in both branches, should be comitted to
  the release branch, then release branch merged to master.

* Create a new release on Github. Mark it as pre-release. This should
  create both a tarball and a git tag.

* Upload the package to portablecl.org/downloads via SFTP or to the
  sourceforge file listing for the pocl project.

* Request for testers in Twitter and/or mailing list. Point the testers to
  send their test reports to you privately or by adding them to the wiki.
  A good way is to create a wiki page for the release schedule and a test
  log. See https://github.com/pocl/pocl/wiki/pocl-0.10-release-testing for
  an example.

* To publish a release, create a new release on Github without the
  checking the pre-release checkbox.
  Upload the tar ball to the sourceforge download page and
  to http://portablecl.org/downloads.
* Update the CHANGES and ANNOUNCEMENT text files in these directories.
  ANNOUNCEMENT is a copy of the latest release notes. A direct link to it can
  be easily circulated in IRC, for example.
* Update the http://portablecl.org web page with the release information.
* Advertise everywhere you can. At least in Twitter and the mailing list.

In case of any problems, ask any previous release manager for help.
Previous releases were managed by the following pocl developers:

* 0.14: Pekka Jääskeläinen
* 0.11: Michal Babej
* 0.10: Pekka Jääskeläinen
* 0.9: Kalle Raiskila
* 0.8: Erik Schnetter
* 0.6 and 0.7: Pekka Jääskeläinen
