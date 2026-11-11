/*********************                                                        */
/*! \file PLConstraintScoreTracker.cpp
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

#include "PLConstraintScoreTracker.h"

void PLConstraintScoreTracker::reset()
{
    _scores.clear();
    _plConstraintToScore.clear();
}

<<<<<<< HEAD
void PLConstraintScoreTracker::initialize( const List<PiecewiseLinearConstraint *>
                                           &plConstraints )
=======
void PLConstraintScoreTracker::initialize( const List<PiecewiseLinearConstraint *> &plConstraints )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
{
    reset();
    for ( const auto &constraint : plConstraints )
    {
        _scores.insert( { constraint, 0 } );
        _plConstraintToScore[constraint] = 0;
    }
}

<<<<<<< HEAD
void PLConstraintScoreTracker::setScore( PiecewiseLinearConstraint *constraint,
                                         double score )
=======
void PLConstraintScoreTracker::setScore( PiecewiseLinearConstraint *constraint, double score )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
{
    ASSERT( _plConstraintToScore.exists( constraint ) );

    double oldScore = _plConstraintToScore[constraint];

    ASSERT( _scores.find( ScoreEntry( constraint, oldScore ) ) != _scores.end() );

    _scores.erase( ScoreEntry( constraint, oldScore ) );
    _scores.insert( ScoreEntry( constraint, score ) );
    _plConstraintToScore[constraint] = score;
}

PiecewiseLinearConstraint *PLConstraintScoreTracker::topUnfixed()
{
    for ( const auto &entry : _scores )
    {
        if ( entry._constraint->isActive() && !entry._constraint->phaseFixed() )
        {
<<<<<<< HEAD
            SCORE_TRACKER_LOG( Stringf( "Score of top unfixed plConstraint: %.2f",
                                        entry._score ).ascii() );
=======
            SCORE_TRACKER_LOG(
                Stringf( "Score of top unfixed plConstraint: %.2f", entry._score ).ascii() );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
            return entry._constraint;
        }
    }
    return NULL;
}
