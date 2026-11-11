/*********************                                                        */
/*! \file MockConstraintMatrixAnalyzerFactory.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Shantanu Thakoor
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

**/

#ifndef __MockConstraintMatrixAnalyzerFactory_h__
#define __MockConstraintMatrixAnalyzerFactory_h__

#include "MockConstraintMatrixAnalyzer.h"
#include "T/ConstraintMatrixAnalyzerFactory.h"

<<<<<<< HEAD
class MockConstraintMatrixAnalyzerFactory :
	public T::Base_createConstraintMatrixAnalyzer,
	public T::Base_discardConstraintMatrixAnalyzer
{
public:
	MockConstraintMatrixAnalyzer mockConstraintMatrixAnalyzer;

	~MockConstraintMatrixAnalyzerFactory()
	{
		if ( mockConstraintMatrixAnalyzer.wasCreated )
		{
			TS_ASSERT( mockConstraintMatrixAnalyzer.wasDiscarded );
		}
	}

	IConstraintMatrixAnalyzer *createConstraintMatrixAnalyzer()
	{
		mockConstraintMatrixAnalyzer.mockConstructor();
		return &mockConstraintMatrixAnalyzer;
	}

	void discardConstraintMatrixAnalyzer( IConstraintMatrixAnalyzer *constraintMatrixAnalyzer )
	{
		TS_ASSERT_EQUALS( constraintMatrixAnalyzer, &mockConstraintMatrixAnalyzer );
		mockConstraintMatrixAnalyzer.mockDestructor();
	}
=======
class MockConstraintMatrixAnalyzerFactory
    : public T::Base_createConstraintMatrixAnalyzer
    , public T::Base_discardConstraintMatrixAnalyzer
{
public:
    MockConstraintMatrixAnalyzer mockConstraintMatrixAnalyzer;

    ~MockConstraintMatrixAnalyzerFactory()
    {
        if ( mockConstraintMatrixAnalyzer.wasCreated )
        {
            TS_ASSERT( mockConstraintMatrixAnalyzer.wasDiscarded );
        }
    }

    IConstraintMatrixAnalyzer *createConstraintMatrixAnalyzer()
    {
        mockConstraintMatrixAnalyzer.mockConstructor();
        return &mockConstraintMatrixAnalyzer;
    }

    void discardConstraintMatrixAnalyzer( IConstraintMatrixAnalyzer *constraintMatrixAnalyzer )
    {
        TS_ASSERT_EQUALS( constraintMatrixAnalyzer, &mockConstraintMatrixAnalyzer );
        mockConstraintMatrixAnalyzer.mockDestructor();
    }
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
};

#endif // __MockConstraintMatrixAnalyzerFactory_h__

//
// Local Variables:
// compile-command: "make -C ../../.. "
// tags-file-name: "../../../TAGS"
// c-basic-offset: 4
// End:
//
