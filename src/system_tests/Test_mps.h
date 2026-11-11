/*********************                                                        */
/*! \file Test_mps.h
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
#include <cxxtest/TestSuite.h>

#include "Engine.h"
#include "InputQuery.h"
#include "MpsParser.h"
#include "MString.h"
=======
#include "Engine.h"
#include "InputQuery.h"
#include "MString.h"
#include "MpsParser.h"

#include <cxxtest/TestSuite.h>
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

class MpsTestSuite : public CxxTest::TestSuite
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

<<<<<<< HEAD
    void test_infesiable() 
    {
        const String filename = RESOURCES_DIR  "/mps/lp_infeasible_1.mps";
=======
    void test_infesiable()
    {
        const String filename = RESOURCES_DIR "/mps/lp_infeasible_1.mps";
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

        // Extract an input query from the network
        InputQuery inputQuery;

        MpsParser mpsParser( filename );
        mpsParser.generateQuery( inputQuery );
        Engine engine;
        if ( !engine.processInputQuery( inputQuery ) )
        {
            // Got infeasible in preprocess stage
            TS_ASSERT( 1 );
        }
<<<<<<< HEAD
        else {
            bool result = engine.solve();
            TS_ASSERT ( !result );
        }
    }

    void test_fesiable() 
=======
        else
        {
            bool result = engine.solve();
            TS_ASSERT( !result );
        }
    }

    void test_fesiable()
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    {
        const char *filename = RESOURCES_DIR "/mps/lp_feasible_1.mps";

        // Extract an input query from the network
        InputQuery inputQuery;

        MpsParser mpsParser( filename );
        mpsParser.generateQuery( inputQuery );
        Engine engine;
<<<<<<< HEAD
        TS_ASSERT_THROWS_NOTHING ( engine.processInputQuery( inputQuery ) );
        TS_ASSERT_THROWS_NOTHING ( engine.solve() );
=======
        TS_ASSERT_THROWS_NOTHING( engine.processInputQuery( inputQuery ) );
        TS_ASSERT_THROWS_NOTHING( engine.solve() );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
        engine.extractSolution( inputQuery );

        // Sanity test
        double value = 0;

        double value0 = inputQuery.getSolutionValue( 0 );
        double value1 = inputQuery.getSolutionValue( 1 );
        double value2 = inputQuery.getSolutionValue( 2 );

<<<<<<< HEAD
        value += 1  * value0;
        value += 2  * value1;
=======
        value += 1 * value0;
        value += 2 * value1;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
        value += -1 * value2;

        TS_ASSERT( FloatUtils::lte( value, 11 ) )

<<<<<<< HEAD
        TS_ASSERT( value0 >= 0 ); 
        TS_ASSERT( value0 <= 2 ); 
        TS_ASSERT( value1 >= -3 ); 
        TS_ASSERT( value1 <= 3 ); 
        TS_ASSERT( value2 >= 4 ); 
        TS_ASSERT( value2 <= 6 ); 
=======
        TS_ASSERT( value0 >= 0 );
        TS_ASSERT( value0 <= 2 );
        TS_ASSERT( value1 >= -3 );
        TS_ASSERT( value1 <= 3 );
        TS_ASSERT( value2 >= 4 );
        TS_ASSERT( value2 <= 6 );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    }
};

//
// Local Variables:
// compile-command: "make -C ../../.. "
// tags-file-name: "../../../TAGS"
// c-basic-offset: 4
// End:
//
