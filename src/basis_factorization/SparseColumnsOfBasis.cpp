/*********************                                                        */
/*! \file SparseColumnsOfBasis.cpp
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
#include "BasisFactorizationError.h"
#include "SparseColumnsOfBasis.h"

=======
#include "SparseColumnsOfBasis.h"

#include "BasisFactorizationError.h"

>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
SparseColumnsOfBasis::SparseColumnsOfBasis( unsigned m )
    : _columns( NULL )
    , _m( m )
{
    _columns = new const SparseUnsortedList *[m];
    if ( !_columns )
<<<<<<< HEAD
        throw BasisFactorizationError( BasisFactorizationError::ALLOCATION_FAILED, "SparseColumnsOfBasis::columns" );
=======
        throw BasisFactorizationError( BasisFactorizationError::ALLOCATION_FAILED,
                                       "SparseColumnsOfBasis::columns" );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
}

SparseColumnsOfBasis::~SparseColumnsOfBasis()
{
    if ( _columns )
    {
        delete[] _columns;
        _columns = NULL;
    }
}

void SparseColumnsOfBasis::dump() const
{
    for ( unsigned i = 0; i < _m; ++i )
        _columns[i]->dumpDense();
}

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
