/*********************                                                        */
/*! \file PolarityBasedDivider.h
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

#ifndef __PolarityBasedDivider_h__
#define __PolarityBasedDivider_h__

#include "List.h"
#include "QueryDivider.h"

#include <math.h>

class PolarityBasedDivider : public QueryDivider
{
public:
    PolarityBasedDivider( std::shared_ptr<IEngine> engine );

    void createSubQueries( unsigned numNewSubQueries,
                           const String queryIdPrefix,
                           const unsigned previousDepth,
<<<<<<< HEAD
                           const PiecewiseLinearCaseSplit
                           &previousSplit,
=======
                           const PiecewiseLinearCaseSplit &previousSplit,
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
                           const unsigned timeoutInSeconds,
                           SubQueries &subQueries );

private:
    std::shared_ptr<IEngine> _engine;

    /*
      Aong the K earliest unfixed ReLUs in the topological order, pick the one
      with polarity closest to 0. K is equal to GlobalConfiguration::POLARITY_CAONDIDATES_THRESHOLD
    */
<<<<<<< HEAD
    PiecewiseLinearConstraint *getPLConstraintToSplit( const
                                                       PiecewiseLinearCaseSplit
                                                       &split );
=======
    PiecewiseLinearConstraint *getPLConstraintToSplit( const PiecewiseLinearCaseSplit &split );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
};

#endif // __LargestIntervalDivider_h__

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
