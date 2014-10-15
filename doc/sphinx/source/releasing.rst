Information for Release Managers
================================

We aim to make a new release according to the Clang/LLVM release schedule.

For each release, a release manager is assigned. Release manager is responsible
for creating and uploading new release candidate tar balls and requesting for
testers from different platforms. After a release candidate round with
success reports and no failure reports, a release is published.

General requirements for a new release are:

* No regressions against the previous versions. 
  If something worked previously, it should not break in the new version.
* Support the latest released Clang/LLVM version. Support for older versions 
  is secondary and implemented with best effort.

A checklist and hints for testing and making a release successfully:

* Create a release branch in github. After branching the release, only
  bug fixes should be committed to the branch. The bug fixes are merged
  *from* the release branch to the *master*. Now development towards the next
  release can go on in *master* while the release branch is being stabilized.
* Set the correct version number without -pre or -rc in the release branch 
  (configure.ac). Increment the version in the master branch. Do not include
  an -rcX in the revision number in the source base so it is possible to 
  release the approved release candidate tar ball by just renaming the tar 
  ball file name.
* Update the new dynamic library version in the master branch. Also done in 
  configure.ac.
  Search for "4:0:3" to see the place where it's set. It includes more info
  in comments.
* Check that CHANGES has the most important updates done during the release 
  cycle. Add missing notable changes from git log.
* Disallow support for the unreleased LLVM version from the release branch 
  because it will most likely stop working before the new LLVM is released.
  That is, modify configure.ac in the release branch to not allow the 
  currently unreleased development version of LLVM.
* Update the release notes in *doc/notes-VERNUM.txt*.
* Create and test the tar ball package with 'make distcheck'. It
  creates a package with pocl-versionstring.tar.gz by default. For
  testing, rename this package to contain the RC number. For example,
  pocl-0.10-rc1.tar.gz.
* Upload the package to portablecl.org/downloads via SFTP or to the 
  sourceforge file listing for the pocl project.
* Request for testers in Twitter and/or mailing list. Point the testers to
  send their test reports to you privately or by adding them to the wiki.
  A good way is to create a wiki page for the release schedule and a test
  log. 
* To publish a release, after testing it thoroughly, rename the latest RC
  tar ball to omit the rcX tag, e.g.,
  pocl-0.10.tar.gz. Upload the tar ball to the sourceforge download page and 
  to http://portablecl.org/downloads. 
* Update the CHANGES and ANNOUNCEMENT text files in these directories. 
  ANNOUNCEMENT is a copy of the latest release notes. A direct link to it can 
  be easily circulated in IRC, for example.
* Update the web page with the release information.
* Announce everywhere you can. At least in Twitter and the mailing list.

In case of any problems, ask any previous release manager for help.
Previous releases were managed by the following pocl developers:

* 0.9: Kalle Raiskila
* 0.8: Erik Schnetter
* 0.6 and 0.7: Pekka Jääskeläinen

