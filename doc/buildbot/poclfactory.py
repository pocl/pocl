#
# Buildbot scripts to create 'buildbot factories' for pocl and LLVM.
# Author: Kalle Raiskila, no rights reserved.
#

from buildbot.process import factory
from buildbot.steps import source
from buildbot.steps.source import SVN
from buildbot.steps.source import Git
from buildbot.steps.shell import ShellCommand, Compile
from buildbot.status.results import *
from buildbot.process.properties import Property
from buildbot.process.properties import Interpolate
from buildbot.process.properties import WithProperties
import os

#The names of the external tests sources
AMD_test_pkg='AMD-APP-SDK-v2.8-RC-lnx64.tgz'
ViennaCL_test_pkg='ViennaCL-1.5.1.tar.gz'


def createPoclFactory(	environ={},
			repository='https://github.com/pocl/pocl.git',
			branch='master',
			buildICD=True,
			llvm_dir='/usr/',
			icd_dir='/usr/',
			tests_dir=None,
			config_opts='',
			pedantic=True,
			tcedir='',
			f=None,
			cmake=False,
			cache_dir=None
			):
	"""
	Create a buildbot factory object that builds pocl.
	
	environ		Dictionary:   The environment variables to append to the build. PATH and
				LD_LIBRARY_PATH will be added from llvm_dir (if given).
	repository	String: the repo to build from. defaults to pocl on github
	branch		String: the branch in 'repository' to build from. default to master
	buildICD	Bool:	if false, the ICD extension is not built.
	llvm_dir	String: LLVM installation dir. I.e. without the 'bin/' or 'lib/'.
	icd_dir		String: ICD loader installation dir. We expect here to be a ICD loader that
				understand the OCL_ICD_VENDORS parameter, i.e. ocl-icd or patched
				Khronos loader.
	tests_dir	String: Path where the extenral testsuite packages can be copied from.
				('cp' is used, so they need to be on the same filesystem).
				NOTE: currently only a placeholder - not tested on the public buildbot
	config_opts	String: extra options to pass to ./configure
	cmake		Bool:	use CMake instead of autotools to build pocl
	cache_dir	String: Set the pocl kernel cache to this dir. If not set, the kcache is disabled.
	"""

	myenviron = environ.copy()

	if 'PATH' in myenviron.keys():
		myenviron['PATH'] = llvm_dir+"/bin/:"+myenviron['PATH']+":${PATH}"
	else:
		myenviron['PATH'] = llvm_dir+"/bin/:${PATH}"
	if 'LD_LIBRARY_PATH' in myenviron.keys():
		myenviron['LD_LIBRARY_PATH'] = llvm_dir+"/lib/:"+myenviron['PATH']+":${LD_LIBRARY_PATH}"
	else:
		myenviron['LD_LIBRARY_PATH'] = llvm_dir+"/lib/:${LD_LIBRARY_PATH}"

	if tcedir:
		myenviron['PATH'] = tcedir+"/bin/:"+myenviron['PATH']
		myenviron['LD_LIBRARY_PATH'] = tcedir+"/lib/:"+myenviron['LD_LIBRARY_PATH']

	if cache_dir:
		myenviron['POCL_KERNEL_CACHE']='1'
		myenviron['POCL_CACHE_DIR']=cache_dir
	else:
		myenviron['POCL_KERNEL_CACHE']='0'

	if cmake:
		logfile="Testing/Temporary/LastTest.log"
	else:
		logfile="tests/testsuite.log"


	if f==None:
		f = factory.BuildFactory()

	f.addStep(
		Git(
			repourl=repository,
			mode=Property('git_mode'),
			ignore_ignores=True,
			branch=branch )
		)

	#clear last test round's kernel cahce. 
	#NB: if you run two slave builds on the same machine, this
	#will not work!
	if cache_dir:
		f.addStep(
			ShellCommand(
				command=['rm', '-rf', cache_dir],
				haltOnFailure=True,
				name='clean kcache',
				description='cleaning kcache',
				descriptionDone='cleaned kcache'
			))

	if not cmake:
		f.addStep(ShellCommand(
				command=["./autogen.sh"],
				haltOnFailure=True,
				name="autoconfig",
				env=myenviron,
				description="autoconfiging",
				descriptionDone="autoconf"))

	if tests_dir!=None:
		f.addStep(ShellCommand(
			haltOnFailure=True,
			command=["cp", "-u", tests_dir+AMD_test_pkg, 
			         "examples/AMD/"+AMD_test_pkg],
			name="copy AMD",
			description="copying",
			descriptionDone="copied AMD",
			#kludge around 'cp' always complaining if source is missing
			decodeRC={0:SUCCESS,1:SUCCESS}
			))
		f.addStep(ShellCommand(
			haltOnFailure=False,
			command=["cp", "-u", tests_dir+ViennaCL_test_pkg,
			         "examples/ViennaCL/"+ViennaCL_test_pkg],
			name="copy ViennaCL",
			description="copying",
			descriptionDone="copied ViennaCL",
			decodeRC={0:SUCCESS,1:SUCCESS}
			))

	if cmake:
		f.addStep(
			ShellCommand(
				command=["cmake", "."],
				env=myenviron,
				haltOnFailure=True,
				name="CMake",
				description="cmaking",
				descriptionDone="cmade"))
	else:
		configOpts=config_opts.split(' ')
		if pedantic==True:
			configOpts = configOpts + ['--enable-pedantic']
		if buildICD==False:
			configOpts = configOpts + ['--disable-icd']

		f.addStep(ShellCommand(
				command=["./configure"] + configOpts,
				haltOnFailure=True,
				name="configure pocl",
				env=myenviron,
				description="configureing",
				descriptionDone="configure"))
	
	f.addStep(Compile(env=myenviron ))

	if tests_dir!=None and not cmake:
		f.addStep(ShellCommand(command=["make", "prepare-examples"],
				haltOnFailure=True,
				name="prepare examples",
				env=myenviron,
				description="preparing",
				descriptionDone="prepare"))
	
	
	if tcedir:
		f.addStep(ShellCommand(command=["./tools/scripts/run_tta_tests"],
				haltOnFailure=True,
				name="checks",
				env=myenviron,
				description="testing",
				descriptionDone="tests",
				logfiles={"test.log": logfile},
				timeout=60*60))
	else:
		f.addStep(ShellCommand(command=["make", "check"],
				haltOnFailure=True,
				name="checks",
				env=myenviron,
				description="testing",
				descriptionDone="tests",
				logfiles={"test.log": logfile},
				#blas3 alone takes 15-20 min.
				timeout=60*60))
		#run the test once more, now from the kernel cache dir, if used
		if cache_dir:
			f.addStep(ShellCommand(command=["make", "check"],
				haltOnFailure=True,
				name="kcache checks",
				env=myenviron,
				description="testing kcache",
				descriptionDone="tested kcache",
				logfiles={"test.log": logfile},
				timeout=5))
	return f

#######
## LLVM/clang builder
##
# srcdir	- LLVM source diectory
# builddir	- LLVM build dir
# installdir	- final LLVM install directory
# test_install_dir - the LLVM install dir pocl_build tests against
def createLLVMFactory(srcdir, builddir, installdir, test_install_dir):
	
	f = factory.BuildFactory()
	f.addStep(
		SVN(
			name='svn-llvm',
			mode='update',
			baseURL='http://llvm.org/svn/llvm-project/llvm/',
			defaultBranch='trunk',
			workdir=srcdir))
	f.addStep(
		SVN(
			name='svn-clang',
			mode='update',
			baseURL='http://llvm.org/svn/llvm-project/cfe/',
			defaultBranch='trunk',
			workdir='%s/tools/clang' % srcdir))
	f.addStep(
		ShellCommand(
			command=[
				'%s/configure' % srcdir,
				'--prefix=' + installdir,
				'--enable-optimized',
				'--enable-targets=host',
				'--enable-shared'],
			workdir=builddir,
			haltOnFailure=True,
			name="configure",
			descriptionDone='configure',
			description='configuring'))
	f.addStep(
		ShellCommand(
			command=['make', '-j', '4'],
			workdir=builddir,
			haltOnFailure=True,
			name = "compile LLVM",
			descriptionDone = 'compiled LLVM',
			description='compiling LLVM'))
	f.addStep(
		ShellCommand(
			command=['make', 'check'],
			workdir=builddir,
			name='LLVM check',
			descriptionDone='checked LLVM',
			haltOnFailure=True,
			description='checking LLVM'))
	f.addStep(
		ShellCommand(
			command=['make', 'install'],
			env={'DESTDIR':test_install_dir},
			workdir=builddir,
			haltOnFailure=True,
			name = 'install for test',
			descriptionDone='install',
			description='installing'))

	f=createPoclFactory(
		llvm_dir=test_install_dir+installdir, 
		pedantic=False,
		f=f)

	f.addStep(
		ShellCommand(
			command=['make', 'install'],
			workdir=builddir,
			haltOnFailure=True,
			name = 'install final',
			descriptionDone='install',
			description='installing'))

	return f


#Use this in schedulers to trigger out documentations to not trigger builds.
def shouldBuildTrigger(change):
	for fname in change.files:
		if os.path.split(fname)[0] != 'doc':
			return True
	return False


# vim: set noexpandtab:
