/*********************                                                        */
/*! \file TableauFactory.h
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

#ifndef __T__TableauFactory_h__
#define __T__TableauFactory_h__

#include "cxxtest/Mock.h"

class ITableau;
class IBoundManager;
class ISelector;

<<<<<<< HEAD
namespace T
{
    ITableau *createTableau( IBoundManager &BoundManager );
    void discardTableau( ITableau *tableau );
}
=======
namespace T {
ITableau *createTableau( IBoundManager &BoundManager );
void discardTableau( ITableau *tableau );
} // namespace T
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

CXXTEST_SUPPLY( createTableau,
                ITableau *,
                createTableau,
<<<<<<< HEAD
                ( IBoundManager &boundManager ),
=======
                ( IBoundManager & boundManager ),
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
                T::createTableau,
                ( boundManager ) );

CXXTEST_SUPPLY_VOID( discardTableau,
                     discardTableau,
<<<<<<< HEAD
                     ( ITableau *tableau ),
=======
                     ( ITableau * tableau ),
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
                     T::discardTableau,
                     ( tableau ) );

#endif // __T__TableauFactory_h__

//
// Local Variables:
// compile-command: "make -C ../../.. "
// tags-file-name: "../../../TAGS"
// c-basic-offset: 4
// End:
//
