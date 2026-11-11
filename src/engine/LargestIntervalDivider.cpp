/*********************                                                        */
/*! \file SODDivider.h
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
#include "Debug.h"
#include "FloatUtils.h"
#include "LargestIntervalDivider.h"
#include "MStringf.h"
#include "PiecewiseLinearCaseSplit.h"

LargestIntervalDivider::LargestIntervalDivider( const List<unsigned>
                                                &inputVariables )
=======
#include "LargestIntervalDivider.h"

#include "Debug.h"
#include "FloatUtils.h"
#include "MStringf.h"
#include "PiecewiseLinearCaseSplit.h"

LargestIntervalDivider::LargestIntervalDivider( const List<unsigned> &inputVariables )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    : _inputVariables( inputVariables )
{
}

void LargestIntervalDivider::createSubQueries( unsigned numNewSubqueries,
                                               const String queryIdPrefix,
                                               const unsigned previousDepth,
<<<<<<< HEAD
                                               const PiecewiseLinearCaseSplit
                                               &previousSplit,
=======
                                               const PiecewiseLinearCaseSplit &previousSplit,
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
                                               const unsigned timeoutInSeconds,
                                               SubQueries &subQueries )
{
    unsigned numBisects = (unsigned)log2( numNewSubqueries );

    List<InputRegion> inputRegions;

    // Create the first input region from the previous case split
    InputRegion region;
    List<Tightening> bounds = previousSplit.getBoundTightenings();
    for ( const auto &bound : bounds )
    {
        if ( bound._type == Tightening::LB )
        {
            region._lowerBounds[bound._variable] = bound._value;
        }
        else
        {
            ASSERT( bound._type == Tightening::UB );
            region._upperBounds[bound._variable] = bound._value;
        }
    }
    inputRegions.append( region );

    // Repeatedly bisect the dimension with the largest interval
    for ( unsigned i = 0; i < numBisects; ++i )
    {
        List<InputRegion> newInputRegions;
        for ( const auto &inputRegion : inputRegions )
        {
            unsigned dimensionToSplit = getLargestInterval( inputRegion );
            bisectInputRegion( inputRegion, dimensionToSplit, newInputRegions );
        }
        inputRegions = newInputRegions;
    }

    unsigned queryIdSuffix = 1; // For query id
    // Create a new subquery for each newly created input region
    for ( const auto &inputRegion : inputRegions )
    {
        // Create a new query id
        String queryId;
        if ( queryIdPrefix == "" )
            queryId = queryIdPrefix + Stringf( "%u", queryIdSuffix++ );
        else
            queryId = queryIdPrefix + Stringf( "-%u", queryIdSuffix++ );

        // Create a new case split
<<<<<<< HEAD
        auto split = std::unique_ptr<PiecewiseLinearCaseSplit>
            ( new PiecewiseLinearCaseSplit() );
=======
        auto split = std::unique_ptr<PiecewiseLinearCaseSplit>( new PiecewiseLinearCaseSplit() );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
        // Add bound as equations for each input variable
        for ( const auto &variable : _inputVariables )
        {
            double lb = inputRegion._lowerBounds[variable];
            double ub = inputRegion._upperBounds[variable];
<<<<<<< HEAD
            split->storeBoundTightening( Tightening( variable, lb,
                                                     Tightening::LB ) );
            split->storeBoundTightening( Tightening( variable, ub,
                                                     Tightening::UB ) );
=======
            split->storeBoundTightening( Tightening( variable, lb, Tightening::LB ) );
            split->storeBoundTightening( Tightening( variable, ub, Tightening::UB ) );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
        }

        // Construct the new subquery and add it to subqueries
        SubQuery *subQuery = new SubQuery;
        subQuery->_queryId = queryId;
<<<<<<< HEAD
        subQuery->_split = std::move(split);
=======
        subQuery->_split = std::move( split );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
        subQuery->_timeoutInSeconds = timeoutInSeconds;
        subQuery->_depth = previousDepth + 1;
        subQueries.append( subQuery );
    }
}

<<<<<<< HEAD
unsigned LargestIntervalDivider::getLargestInterval( const InputRegion
                                                     &inputRegion )
{
    ASSERT( inputRegion._lowerBounds.size() == inputRegion._upperBounds.size() );
    unsigned dimensionToSplit = 0;
    double largestInterval = 0;
=======
unsigned LargestIntervalDivider::getLargestInterval( const InputRegion &inputRegion )
{
    ASSERT( inputRegion._lowerBounds.size() == inputRegion._upperBounds.size() );
    unsigned dimensionToSplit = 0;
    double largestInterval = -1;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    DEBUG( bool haveCandidate = false );

    for ( const auto &variable : _inputVariables )
    {
<<<<<<< HEAD
        double interval = inputRegion._upperBounds[variable] -
            inputRegion._lowerBounds[variable];

        if ( FloatUtils::isZero( interval ) )
            continue;
=======
        double interval = inputRegion._upperBounds[variable] - inputRegion._lowerBounds[variable];
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

        DEBUG( haveCandidate = true );

        if ( interval > largestInterval )
        {
            dimensionToSplit = variable;
            largestInterval = interval;
        }
    }
<<<<<<< HEAD
=======
    ASSERT( largestInterval >= 0 );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    ASSERT( haveCandidate );

    return dimensionToSplit;
}

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
