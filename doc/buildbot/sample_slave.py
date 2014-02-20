from buildbot.buildslave import BuildSlave
from buildbot.schedulers.basic  import SingleBranchScheduler
from buildbot.changes import filter
from buildbot.config import BuilderConfig
from buildbot.schedulers.forcesched import *
from poclfactory import createPoclFactory

# overrride the 'sample_slave' with a descriptive function name
# Note: when finished renaming, the string "sample" should not appear anywhere in this file!
#
# c - the global buildbot configuration data structure
# common_branch - this is the branch that the slave should build.
#                 typically 'master', but during release it will be changed
#                 to the release branch
def sample_slave( c, common_branch ):
        
	#create a new slave in the master's database
	c['slaves'].append(
		BuildSlave(
			"sample_slave_name",
			"password" ))

	# lauch the builders listed in "builderNames" whenever the change poller notices a change to github pocl
	c['schedulers'].append(
		 SingleBranchScheduler(name="name for scheduler, not sure where this is used",
			change_filter=filter.ChangeFilter(branch=common_branch),
			treeStableTimer=60,
			builderNames=[
				"sample_builder_name - this is the name that appears on the webpage"] ))

	#create one set of steps to build pocl. See poclfactory.py for details
	# on how to configure it
	sample_factory = createPoclFactory()
	
	#register your build to the master
	c['builders'].append(
		BuilderConfig(
			name = "sample_builder_name - this is the name that appears on the webpage",
			slavenames=["sample_slave_name"],
			factory = sample_factory ))


