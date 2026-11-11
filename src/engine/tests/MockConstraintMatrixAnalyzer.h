/*********************                                                        */
/*! \file MockConstraintMatrixAnalyzer.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Shantanu Thakoor, Guy Katz
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

**/

#ifndef __MockConstraintMatrixAnalyzer_h__
#define __MockConstraintMatrixAnalyzer_h__

#include "IConstraintMatrixAnalyzer.h"

class MockConstraintMatrixAnalyzer : public IConstraintMatrixAnalyzer
{
public:
<<<<<<< HEAD
	MockConstraintMatrixAnalyzer()
	{
		wasCreated = false;
		wasDiscarded = false;
=======
    MockConstraintMatrixAnalyzer()
    {
        wasCreated = false;
        wasDiscarded = false;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    }

    ~MockConstraintMatrixAnalyzer()
    {
    }

<<<<<<< HEAD
	bool wasCreated;
	bool wasDiscarded;

	void mockConstructor()
	{
		TS_ASSERT( !wasCreated );
		wasCreated = true;
	}

	void mockDestructor()
	{
		TS_ASSERT( wasCreated );
		TS_ASSERT( !wasDiscarded );
		wasDiscarded = true;
	}

    void analyze( const double */* matrix */, unsigned /* m */, unsigned /* n */ )
    {
    }

    void analyze( const SparseUnsortedList **/* matrix */, unsigned /* m */, unsigned /* n */ )
=======
    bool wasCreated;
    bool wasDiscarded;

    void mockConstructor()
    {
        TS_ASSERT( !wasCreated );
        wasCreated = true;
    }

    void mockDestructor()
    {
        TS_ASSERT( wasCreated );
        TS_ASSERT( !wasDiscarded );
        wasDiscarded = true;
    }

    void analyze( const double * /* matrix */, unsigned /* m */, unsigned /* n */ )
    {
    }

    void analyze( const SparseUnsortedList ** /* matrix */, unsigned /* m */, unsigned /* n */ )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    {
    }

    unsigned getRank() const
    {
        return 0;
    }

    List<unsigned> nextIndependentColumns;
    List<unsigned> getIndependentColumns() const
    {
        return nextIndependentColumns;
    }

    Set<unsigned> nextRedundantRows;
    Set<unsigned> getRedundantRows() const
    {
        return nextRedundantRows;
    }
};

#endif // __MockConstraintMatrixAnalyzer_h__

//
// Local Variables:
// compile-command: "make -C ../../.. "
// tags-file-name: "../../../TAGS"
// c-basic-offset: 4
// End:
//
