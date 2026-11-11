/*********************                                                        */
/*! \file MockRowBoundTightener.h
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

#ifndef __MockRowBoundTightener_h__
#define __MockRowBoundTightener_h__

#include "IRowBoundTightener.h"

class MockRowBoundTightener : public IRowBoundTightener
{
public:
    MockRowBoundTightener()
    {
        wasCreated = false;
        wasDiscarded = false;

        setDimensionsWasCalled = false;
    }

    ~MockRowBoundTightener()
    {
    }

<<<<<<< HEAD
	bool wasCreated;
	bool wasDiscarded;
    const ITableau *lastTableau;

	void mockConstructor( const ITableau &tableau )
	{
		TS_ASSERT( !wasCreated );
		wasCreated = true;
        lastTableau = &tableau;
	}
=======
    bool wasCreated;
    bool wasDiscarded;
    const ITableau *lastTableau;

    void mockConstructor( const ITableau &tableau )
    {
        TS_ASSERT( !wasCreated );
        wasCreated = true;
        lastTableau = &tableau;
    }
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    void mockDestructor()
    {
        TS_ASSERT( wasCreated );
        TS_ASSERT( !wasDiscarded );
        wasDiscarded = true;
    }

    bool setDimensionsWasCalled;
    void setDimensions()
<<<<<<< HEAD
    {
        setDimensionsWasCalled = true;
    }

    void resetBounds()
    {
    }

    void clear() {}
    void notifyLowerBound( unsigned /* variable */, double /* bound */ ) {}
    void notifyUpperBound( unsigned /* variable */, double /* bound */ ) {}
    void examineInvertedBasisMatrix( bool /* untilSaturation */ ) {}
    void examineConstraintMatrix( bool /* untilSaturation */ ) {}
    void examinePivotRow() {}
    void getRowTightenings( List<Tightening> &/* tightenings */ ) const {}
    void setStatistics( Statistics */* statistics */ ) {}
    void examineImplicitInvertedBasisMatrix( bool /* untilSaturation */ ) {}
    void setBoundsPointers( const double */* lower */, const double */* upper */ ) {}
=======
    {
        setDimensionsWasCalled = true;
    }

    void resetBounds()
    {
    }

    void clear()
    {
    }
    void notifyLowerBound( unsigned /* variable */, double /* bound */ )
    {
    }
    void notifyUpperBound( unsigned /* variable */, double /* bound */ )
    {
    }
    void examineInvertedBasisMatrix( bool /* untilSaturation */ )
    {
    }
    void examineConstraintMatrix( bool /* untilSaturation */ )
    {
    }
    void examinePivotRow()
    {
    }
    void getRowTightenings( List<Tightening> & /* tightenings */ ) const
    {
    }
    void setStatistics( Statistics * /* statistics */ )
    {
    }
    void examineImplicitInvertedBasisMatrix( bool /* untilSaturation */ )
    {
    }
    void setBoundsPointers( const double * /* lower */, const double * /* upper */ )
    {
    }
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
};

#endif // __MockRowBoundTightener_h__

//
// Local Variables:
// compile-command: "make -C ../../.. "
// tags-file-name: "../../../TAGS"
// c-basic-offset: 4
// End:
//
