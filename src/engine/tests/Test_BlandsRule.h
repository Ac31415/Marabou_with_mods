/*********************                                                        */
/*! \file Test_BlandsRule.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz, Duligur Ibeling
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]
<<<<<<< HEAD

**/
=======
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

**/

#include "BlandsRule.h"
#include "MarabouError.h"
#include "MockTableau.h"
<<<<<<< HEAD
#include "MarabouError.h"
=======
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

#include <cxxtest/TestSuite.h>
#include <string.h>

class MockForBlandsRule
{
public:
};

class BlandsRuleTestSuite : public CxxTest::TestSuite
{
public:
    MockForBlandsRule *mock;
    MockTableau *tableau;

    void setUp()
    {
        TS_ASSERT( mock = new MockForBlandsRule );
        TS_ASSERT( tableau = new MockTableau );
    }

    void tearDown()
    {
        TS_ASSERT_THROWS_NOTHING( delete tableau );
        TS_ASSERT_THROWS_NOTHING( delete mock );
    }

    void test_select()
    {
        BlandsRule blandsRule;

        Set<unsigned> excluded;
        List<unsigned> candidates;

        TS_ASSERT( !blandsRule.select( *tableau, candidates, excluded ) );

        candidates.append( 3 );
        tableau->nextNonBasicIndexToVariable[3] = 20;

        candidates.append( 10 );
        tableau->nextNonBasicIndexToVariable[10] = 4;

        candidates.append( 2 );
        tableau->nextNonBasicIndexToVariable[2] = 10;

        candidates.append( 51 );
        tableau->nextNonBasicIndexToVariable[51] = 6;

        TS_ASSERT( blandsRule.select( *tableau, candidates, excluded ) );
        TS_ASSERT_EQUALS( tableau->mockEnteringVariable, 10U );

        excluded.insert( 10 );
        TS_ASSERT( blandsRule.select( *tableau, candidates, excluded ) );
        TS_ASSERT_EQUALS( tableau->mockEnteringVariable, 51U );
        excluded.clear();

        candidates.append( 100 );
        tableau->nextNonBasicIndexToVariable[100] = 1;

        tableau->mockCandidates = candidates;

        // TS_ASSERT_EQUALS( blandsRule.select( candidates, *tableau ), 100U );

        TS_ASSERT( blandsRule.select( *tableau, candidates, excluded ) );
        TS_ASSERT_EQUALS( tableau->mockEnteringVariable, 100U );
    }
};

//
// Local Variables:
// compile-command: "make -C ../../.. "
// tags-file-name: "../../../TAGS"
// c-basic-offset: 4
// End:
//
