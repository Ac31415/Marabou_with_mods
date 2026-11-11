'''
DNC Example
====================

Top contributors (to current version):
  - Andrew Wu
  - Kyle Julian
  
This file is part of the Marabou project.
Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
in the top-level source directory) and their institutional affiliations.
All rights reserved. See the file COPYING in the top-level source
directory for licensing information.
'''

import sys
import numpy as np

# Append to path if needed
#sys.path.append(</Path/To/Marabou/>)
<<<<<<< HEAD
=======
sys.path.append('/Users/wen-chungcheng/Marabou_with_mods')
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

from maraboupy import Marabou

# %%
# Load an example network and place an output constraint
<<<<<<< HEAD
nnet_file_name = "../../src/input_parsers/acas_example/ACASXU_run2a_1_1_tiny_2.nnet"
net = Marabou.read_nnet(nnet_file_name)
net.setLowerBound(net.outputVars[0][0], .5)
=======
# nnet_file_name = "../../src/input_parsers/acas_example/ACASXU_run2a_1_1_tiny_2.nnet"
nnet_file_name = "/Users/wen-chungcheng/Marabou_with_mods/src/input_parsers/acas_example/ACASXU_run2a_1_1_tiny_2.nnet"
net = Marabou.read_nnet(nnet_file_name)
# net.setLowerBound(net.outputVars[0][0], .5)
net.setLowerBound(net.outputVars[0][0][0], .5)
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

# %%
# Solve the query with DNC mode turned on, which should return satisfying variable values
options = Marabou.createOptions(snc=True, verbosity=0, initialSplits=2);
exitCode, vals, stats = net.solve(options=options)
print(exitCode)
assert(exitCode == "sat")
assert len(vals) > 0
