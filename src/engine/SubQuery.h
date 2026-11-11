/*********************                                                        */
/*! \file SubQuery.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Haoze Wu
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

**/

#ifndef __SubQuery_h__
#define __SubQuery_h__

#include "List.h"
#include "MString.h"
#include "PiecewiseLinearCaseSplit.h"
#include "SmtState.h"

#include <boost/lockfree/queue.hpp>
#include <utility>

// Struct representing a subquery
struct SubQuery
{
    SubQuery()
    {
    }

    String _queryId;
    std::unique_ptr<PiecewiseLinearCaseSplit> _split;
    std::unique_ptr<SmtState> _smtState;
    unsigned _timeoutInSeconds;
    unsigned _depth;
};

// Synchronized Queue containing the Sub-Queries shared by workers
<<<<<<< HEAD
typedef boost::lockfree::queue<SubQuery *, boost::lockfree::
  fixed_sized<false>>WorkerQueue;
=======
typedef boost::lockfree::queue<SubQuery *, boost::lockfree::fixed_sized<false>> WorkerQueue;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

// A vector of Sub-Queries

// Guy: consider using our wrapper class Vector instead of std::vector
typedef List<SubQuery *> SubQueries;

#endif // __SubQuery_h__

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
