/*********************                                                        */
/*! \file MockTableauFactory.h
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

#ifndef __MockTableauFactory_h__
#define __MockTableauFactory_h__

#include "BoundManager.h"
<<<<<<< HEAD

=======
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
#include "MockTableau.h"
#include "T/TableauFactory.h"

class MockTableauFactory
    : public T::Base_createTableau
    , public T::Base_discardTableau
{
public:
<<<<<<< HEAD
	MockTableau mockTableau;

	~MockTableauFactory()
	{
		if ( mockTableau.wasCreated )
		{
			TS_ASSERT( mockTableau.wasDiscarded );
		}
	}

	ITableau *createTableau( IBoundManager &/*boundManager*/ )
	{
		mockTableau.mockConstructor();
		return &mockTableau;
	}

	void discardTableau( ITableau *tableau )
	{
		TS_ASSERT_EQUALS( tableau, &mockTableau );
		mockTableau.mockDestructor();
	}
=======
    MockTableau mockTableau;

    ~MockTableauFactory()
    {
        if ( mockTableau.wasCreated )
        {
            TS_ASSERT( mockTableau.wasDiscarded );
        }
    }

    ITableau *createTableau( IBoundManager & /*boundManager*/ )
    {
        mockTableau.mockConstructor();
        return &mockTableau;
    }

    void discardTableau( ITableau *tableau )
    {
        TS_ASSERT_EQUALS( tableau, &mockTableau );
        mockTableau.mockDestructor();
    }
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
};

#endif // __MockTableauFactory_h__

//
// Local Variables:
// compile-command: "make -C ../../.. "
// tags-file-name: "../../../TAGS"
// c-basic-offset: 4
// End:
//
