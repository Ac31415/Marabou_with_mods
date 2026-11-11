/*********************                                                        */
/*! \file Test_LinearExpression.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Haoze (Andrew) Wu
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
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

#include "LinearExpression.h"

class LinearExpressionTestSuite : public CxxTest::TestSuite
{
public:

=======
#include "LinearExpression.h"

#include <cxxtest/TestSuite.h>

class LinearExpressionTestSuite : public CxxTest::TestSuite
{
public:
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    void setUp()
    {
    }

    void tearDown()
    {
    }

    void test_evaluate()
    {
        LinearExpression expr;
        // 2x0 - 3x1 + x2 - 0.5
        expr._addends[0] = 2;
        expr._addends[1] = -3;
        expr._addends[2] = 1;
        expr._constant = -0.5;

        Map<unsigned, double> assignment;
        assignment[0] = 1;
        assignment[1] = 3;
        assignment[2] = -2;

<<<<<<< HEAD
        TS_ASSERT_EQUALS( expr.evaluate( assignment ), -9.5);
=======
        TS_ASSERT_EQUALS( expr.evaluate( assignment ), -9.5 );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    }
};
