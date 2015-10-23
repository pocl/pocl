#
# Template for adding a new buildslave to the 
# central pocl buildbot master.
# Please replace *every* instance of "sample" with 
# something that reflects your machine!

from buildbot.buildslave import BuildSlave
from buildbot.schedulers.basic  import SingleBranchScheduler
from buildbot.changes import filter
from buildbot.config import BuilderConfig
from buildbot.schedulers.forcesched import *
from poclfactory import createPoclFactory, shouldBuildTrigger

#overrride the 'sample_slave' with a descriptive function name
def sample_slave( c, common_branch):
        
	#create a new slave in the master's database
	c['slaves'].append(
		BuildSlave(
			"sample_slave_name",
			"password" ))

	# build the tree whenever the change poller notices a change
	c['schedulers'].append(
		 SingleBranchScheduler(name="sample_gitpoller",
			change_filter=filter.ChangeFilter(branch=common_branch),
			treeStableTimer=60,
			fileIsImportant=shouldBuildTrigger,
			builderNames=[
				"sample_builder_name"] ))
	# Allow authenticated (to the buildmaster) users to force a build
	# Optionally, force a full build, i.e. 'git clean -f -d -x' instead
	# of a incremental build (essentially 'git pull && make)
	c['schedulers'].append(
		ForceScheduler(
			name="a name for your forcescheduler",
			branch=FixedParameter(name="branch", default=""),
			revision=FixedParameter(name="revision", default=""),
			repository=FixedParameter(name="repository", default=""),
			project=FixedParameter(name="repository", default=""),
			builderNames=["sample_LLVM_builder_name"],
			properties=[
				ChoiceStringParameter(
					name="git_mode",
					label="how to update git (see buildbot docs)",
					choices=["incremental", "full"],
					default="incremental")
			]))

	#create one set of steps to build pocl. See poclfactory.py for
	#parameters to pass this function.
	#You can create as many factories+builders as your slave has space & need for
	#e.g. one per LLVM version, or different build options for pocl
	sample_factory = createPoclFactory()
	
	#register your build to the master
	c['builders'].append(
		BuilderConfig(
			name = "sample_builder_name",
			slavenames=["sample_slave_name"],
			factory = sample_factory ))

	#create one set of steps to build latest LLVM. This is a check for if LLVM
	# has introduced regressions (e.g. API changes, asserts or codegen issues)
	# This factory:
	# - builds LLVM from svn & runs its checks
	# - installs LLVM to a temporary directory
	# - builds & checks pocl against this LLVM
	# - installs the built LLVM into a permanent directory
	# You want also a pocl factory/builder to run against this second installation LLVM
	# to check if pocl has introduced regressions.
	#. See poclfactory.py for parameters to pass this function
	sample_LLVM_factory = createLLVMFactory()

	c['builders'].append(
		BuilderConfig(
			name = "sample_LLVM_builder_name",
			slavenames=["sample_slave_name"],
			factory = sample_LLVM_factory ))


