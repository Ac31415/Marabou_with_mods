/*********************                                                        */
/*! \file Test_GurobiWrapper.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors gurobiWrappered in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** \brief [[ Add one-line brief description here ]]
 **
 ** [[ Add lengthier description here ]]
 **/

<<<<<<< HEAD
#include <cxxtest/TestSuite.h>

=======
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
#include "FloatUtils.h"
#include "GurobiWrapper.h"
#include "MString.h"
#include "MockErrno.h"

<<<<<<< HEAD
=======
#include <cxxtest/TestSuite.h>

>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
class GurobiWrapperTestSuite : public CxxTest::TestSuite
{
public:
    MockErrno *mockErrno;

    void setUp()
    {
        TS_ASSERT( mockErrno = new MockErrno );
    }

    void tearDown()
    {
        TS_ASSERT_THROWS_NOTHING( delete mockErrno );
    }

    void test_optimize()
    {
#ifdef ENABLE_GUROBI
        GurobiWrapper gurobi;

        gurobi.addVariable( "x", 0, 3 );
        gurobi.addVariable( "y", 0, 3 );
        gurobi.addVariable( "z", 0, 3 );

        // x + y + z <= 5
        List<GurobiWrapper::Term> contraint = {
            GurobiWrapper::Term( 1, "x" ),
            GurobiWrapper::Term( 1, "y" ),
            GurobiWrapper::Term( 1, "z" ),
        };

        gurobi.addLeqConstraint( contraint, 5 );

        // Cost: -x - 2y + z
        List<GurobiWrapper::Term> cost = {
            GurobiWrapper::Term( -1, "x" ),
            GurobiWrapper::Term( -2, "y" ),
            GurobiWrapper::Term( +1, "z" ),
        };

        gurobi.setCost( cost );

        // Solve and extract
        TS_ASSERT_THROWS_NOTHING( gurobi.solve() );

        Map<String, double> solution;
        double costValue;

        TS_ASSERT_THROWS_NOTHING( gurobi.extractSolution( solution, costValue ) );

        TS_ASSERT( FloatUtils::areEqual( solution["x"], 2 ) );
        TS_ASSERT( FloatUtils::areEqual( solution["y"], 3 ) );
        TS_ASSERT( FloatUtils::areEqual( solution["z"], 0 ) );

        TS_ASSERT( FloatUtils::areEqual( costValue, -8 ) );

#else
        TS_ASSERT( true );
#endif // ENABLE_GUROBI
    }
};
<<<<<<< HEAD

//
// Local Variables:
// compile-command: "make -C ../../.. "
// tags-file-name: "../../../TAGS"
// c-basic-offset: 4
// End:
//
=======
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
