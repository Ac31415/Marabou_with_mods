/*********************                                                        */
/*! \file Simulator.cpp
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
#include "Debug.h"
#include "FloatUtils.h"
#include "Preprocessor.h"
#include "MarabouError.h"
#include "Simulator.h"
=======
#include "Simulator.h"

#include "Debug.h"
#include "FloatUtils.h"
#include "MarabouError.h"
#include "Preprocessor.h"
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

#ifdef _WIN32
#include <time.h>
#endif

void Simulator::runSimulations( const InputQuery &inputQuery, unsigned numberOfSimulations )
{
    unsigned seed = time( NULL );
    runSimulations( inputQuery, numberOfSimulations, seed );
}

<<<<<<< HEAD
void Simulator::runSimulations( const InputQuery &inputQuery, unsigned numberOfSimulations, unsigned seed )
=======
void Simulator::runSimulations( const InputQuery &inputQuery,
                                unsigned numberOfSimulations,
                                unsigned seed )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
{
    // Store the original query, so a fresh copy can be used in every simulation
    storeOriginalQuery( inputQuery );

    // Initialize randomness
    srand( seed );

    // Perform the actual simulations
    for ( unsigned i = 0; i < numberOfSimulations; ++i )
        runSingleSimulation();
}

void Simulator::storeOriginalQuery( const InputQuery &inputQuery )
{
    _originalQuery = inputQuery;

    for ( const auto &plConstraint : _originalQuery.getPiecewiseLinearConstraints() )
    {
        List<unsigned> variables = plConstraint->getParticipatingVariables();
        for ( unsigned variable : variables )
        {
            plConstraint->notifyLowerBound( variable, _originalQuery.getLowerBound( variable ) );
            plConstraint->notifyUpperBound( variable, _originalQuery.getUpperBound( variable ) );
        }
    }

    _originalQuery = *( Preprocessor().preprocess( _originalQuery, false ) );

    if ( _originalQuery.countInfiniteBounds() != 0 )
<<<<<<< HEAD
        throw MarabouError( MarabouError::SIMULATOR_ERROR, "Preprocessed query has infinite bounds" );

    if ( _originalQuery.getNumInputVariables() == 0 )
        throw MarabouError( MarabouError::SIMULATOR_ERROR, "Preprocessed query has no input variables" );
=======
        throw MarabouError( MarabouError::SIMULATOR_ERROR,
                            "Preprocessed query has infinite bounds" );

    if ( _originalQuery.getNumInputVariables() == 0 )
        throw MarabouError( MarabouError::SIMULATOR_ERROR,
                            "Preprocessed query has no input variables" );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
}

void Simulator::runSingleSimulation()
{
    InputQuery query = _originalQuery;

    List<unsigned> inputs = query.getInputVariables();
    for ( const auto &input : inputs )
    {
        double lb = query.getLowerBound( input );
        double ub = query.getUpperBound( input );

<<<<<<< HEAD
        double factor = ((double)rand()) / RAND_MAX;
=======
        double factor = ( (double)rand() ) / RAND_MAX;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
        double value = lb + factor * ( ub - lb );

        query.setLowerBound( input, value );
        query.setUpperBound( input, value );
    }

    // Propagate the random values through the network using the preprocessor
    Preprocessor preprocessor;

    // If we set elimination to true, the PL constraints will be removed
    query = *( preprocessor.preprocess( query, false ) );

    // Extract the result
    Simulator::Result result;
    for ( unsigned i = 0; i < _originalQuery.getNumberOfVariables(); ++i )
    {
        // Make sure that a value has been calculated for every variable
        ASSERT( !preprocessor.variableIsMerged( i ) );
        if ( !preprocessor.variableIsFixed( i ) )
<<<<<<< HEAD
            throw MarabouError( MarabouError::SIMULATOR_ERROR, "Could not calculate an exact assignment" );

        ASSERT( FloatUtils::areEqual( query.getLowerBound( i ),
                                      query.getUpperBound( i ) ) );
=======
            throw MarabouError( MarabouError::SIMULATOR_ERROR,
                                "Could not calculate an exact assignment" );

        ASSERT( FloatUtils::areEqual( query.getLowerBound( i ), query.getUpperBound( i ) ) );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

        result[i] = query.getLowerBound( i );
    }

    _results.append( result );
}

const List<Simulator::Result> *Simulator::getResults()
{
    return &_results;
}

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
