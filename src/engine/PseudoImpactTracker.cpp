/*********************                                                        */
/*! \file PseudoImpactTracker.cpp
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

<<<<<<< HEAD
#include "GlobalConfiguration.h"
#include "PseudoImpactTracker.h"

PseudoImpactTracker::PseudoImpactTracker()
{}

void PseudoImpactTracker::updateScore( PiecewiseLinearConstraint *constraint,
                                       double score )
=======
#include "PseudoImpactTracker.h"

#include "GlobalConfiguration.h"

PseudoImpactTracker::PseudoImpactTracker()
{
}

void PseudoImpactTracker::updateScore( PiecewiseLinearConstraint *constraint, double score )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
{
    ASSERT( _plConstraintToScore.exists( constraint ) );

    double alpha = GlobalConfiguration::EXPONENTIAL_MOVING_AVERAGE_ALPHA;
    double oldScore = _plConstraintToScore[constraint];
<<<<<<< HEAD
    double newScore = ( 1 - alpha ) * oldScore  + alpha * score;
=======
    double newScore = ( 1 - alpha ) * oldScore + alpha * score;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    ASSERT( _scores.find( ScoreEntry( constraint, oldScore ) ) != _scores.end() );
    _scores.erase( ScoreEntry( constraint, oldScore ) );
    _scores.insert( ScoreEntry( constraint, newScore ) );
    _plConstraintToScore[constraint] = newScore;
}
