/*********************                                                        */
/*! \file RowBoundTightenerFactory.h
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

#ifndef __T__RowBoundTightenerFactory_h__
#define __T__RowBoundTightenerFactory_h__

#include "cxxtest/Mock.h"

class IRowBoundTightener;
class ITableau;


<<<<<<< HEAD
namespace T
{
	IRowBoundTightener *createRowBoundTightener( const ITableau &tableau );
	void discardRowBoundTightener( IRowBoundTightener *rowBoundTightener );
}

CXXTEST_SUPPLY( createRowBoundTightener,
				IRowBoundTightener *,
				createRowBoundTightener,
				( const ITableau &tableau ),
				T::createRowBoundTightener,
				( tableau ) );
=======
namespace T {
IRowBoundTightener *createRowBoundTightener( const ITableau &tableau );
void discardRowBoundTightener( IRowBoundTightener *rowBoundTightener );
} // namespace T

CXXTEST_SUPPLY( createRowBoundTightener,
                IRowBoundTightener *,
                createRowBoundTightener,
                ( const ITableau &tableau ),
                T::createRowBoundTightener,
                ( tableau ) );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

CXXTEST_SUPPLY_VOID( discardRowBoundTightener,
                     discardRowBoundTightener,
                     ( IRowBoundTightener * rowBoundTightener ),
                     T::discardRowBoundTightener,
                     ( rowBoundTightener ) );

#endif // __T__RowBoundTightenerFactory_h__

//
// Local Variables:
// compile-command: "make -C ../../.. "
// tags-file-name: "../../../TAGS"
// c-basic-offset: 4
// End:
//
