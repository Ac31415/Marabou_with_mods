/*********************                                                        */
/*! \file NetworkLevelReasoner.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

 **/

<<<<<<< HEAD
=======
#include "ParallelSolver.h"

>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
#include "InfeasibleQueryException.h"
#include "Layer.h"
#include "MStringf.h"
#include "NLRError.h"
#include "Options.h"
<<<<<<< HEAD
#include "ParallelSolver.h"
=======
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
#include "TimeUtils.h"

#include <boost/thread.hpp>

namespace NLR {

void ParallelSolver::clearSolverQueue( SolverQueue &freeSolvers )
{
    // Remove the solvers
    GurobiWrapper *freeSolver;
    while ( freeSolvers.pop( freeSolver ) )
        delete freeSolver;
}

void ParallelSolver::enqueueSolver( SolverQueue &solvers, GurobiWrapper *solver )
{
    if ( !solvers.push( solver ) )
    {
        ASSERT( false );
    }
}

} // namespace NLR
