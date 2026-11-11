/*********************                                                        */
/*! \file RowBoundTightenerFactory.cpp
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

#include "RowBoundTightener.h"

namespace T {
IRowBoundTightener *createRowBoundTightener( const ITableau &tableau )
{
<<<<<<< HEAD
	IRowBoundTightener *createRowBoundTightener( const ITableau &tableau )
	{
		return new RowBoundTightener( tableau );
	}
=======
    return new RowBoundTightener( tableau );
}
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

void discardRowBoundTightener( IRowBoundTightener *rowBoundTightener )
{
    delete rowBoundTightener;
}
} // namespace T

//
// Local Variables:
// compile-command: "make -C ../../.. "
// tags-file-name: "../../../TAGS"
// c-basic-offset: 4
// End:
//
