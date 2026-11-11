/**********************/
/*! \file Test_WsLayerElimination.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Haoze (Andrew) Wu
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

**/

<<<<<<< HEAD
#include <cxxtest/TestSuite.h>
=======
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
#include "GurobiWrapper.h"
#include "IterativePropagator.h"
#include "NetworkLevelReasoner.h"
#include "ParallelSolver.h"

<<<<<<< HEAD
=======
#include <cxxtest/TestSuite.h>

>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
class MockForNetworkLevelReasoner
{
public:
};

class NetworkLevelReasonerTestSuite : public CxxTest::TestSuite
{
public:
<<<<<<< HEAD

=======
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    void setUp()
    {
    }

    void tearDown()
    {
    }

    void test_enqueueSolver()
    {
        NLR::NetworkLevelReasoner nlr;
        NLR::IterativePropagator mock = NLR::IterativePropagator( &nlr );
        unsigned numberOfWorkers = 4;
<<<<<<< HEAD
        NLR::ParallelSolver::SolverQueue solvers ( numberOfWorkers );
        GurobiWrapper *gurobi = new GurobiWrapper();
        TS_ASSERT_THROWS_NOTHING( mock.enqueueSolver( solvers, gurobi) );
=======
        NLR::ParallelSolver::SolverQueue solvers( numberOfWorkers );
        GurobiWrapper *gurobi = new GurobiWrapper();
        TS_ASSERT_THROWS_NOTHING( mock.enqueueSolver( solvers, gurobi ) );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
        TS_ASSERT( !solvers.empty() );
        GurobiWrapper *gurobiPtr = NULL;
        TS_ASSERT_THROWS_NOTHING( solvers.pop( gurobiPtr ) );
        TS_ASSERT( solvers.empty() );
        delete gurobiPtr;
    }

    void test_clear_solver_queue()
    {
        NLR::NetworkLevelReasoner nlr;
        NLR::IterativePropagator mock = NLR::IterativePropagator( &nlr );
        unsigned numberOfWorkers = 4;
<<<<<<< HEAD
        NLR::ParallelSolver::SolverQueue solvers ( numberOfWorkers );
        for ( unsigned i = 0; i < numberOfWorkers; ++i )
        {
            GurobiWrapper *gurobi = new GurobiWrapper();
            TS_ASSERT_THROWS_NOTHING( mock.enqueueSolver( solvers, gurobi) );
=======
        NLR::ParallelSolver::SolverQueue solvers( numberOfWorkers );
        for ( unsigned i = 0; i < numberOfWorkers; ++i )
        {
            GurobiWrapper *gurobi = new GurobiWrapper();
            TS_ASSERT_THROWS_NOTHING( mock.enqueueSolver( solvers, gurobi ) );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
        }
        TS_ASSERT( !solvers.empty() );
        TS_ASSERT_THROWS_NOTHING( mock.clearSolverQueue( solvers ) );
        TS_ASSERT( solvers.empty() );
    }
};
