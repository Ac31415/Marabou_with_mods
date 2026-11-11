/*********************                                                        */
/*! \file DeepPolyInputElement.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Haoze Andrew Wu
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
#include "DeepPolyInputElement.h"
=======
#include "DeepPolyInputElement.h"

#include "Debug.h"
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
#include "FloatUtils.h"
#include "NLRError.h"

namespace NLR {

DeepPolyInputElement::DeepPolyInputElement( Layer *layer )
{
    _layer = layer;
    _size = layer->getSize();
    _layerIndex = layer->getLayerIndex();
}

DeepPolyInputElement::~DeepPolyInputElement()
{
    freeMemoryIfNeeded();
}

void DeepPolyInputElement::execute( const Map<unsigned, DeepPolyElement *> & )
{
    log( "Executing..." );
    if ( hasPredecessor() )
    {
        throw NLRError( NLRError::INPUT_LAYER_NOT_THE_FIRST_LAYER );
    }
    // Update the concrete bounds
    freeMemoryIfNeeded();
    allocateMemory();
    getConcreteBounds();
    log( "Executing - done" );
}

<<<<<<< HEAD
void DeepPolyInputElement::symbolicBoundInTermsOfPredecessor
( const double *, const double *, double *, double *, double *, double *,
  unsigned, DeepPolyElement * )
=======
void DeepPolyInputElement::symbolicBoundInTermsOfPredecessor( const double *,
                                                              const double *,
                                                              double *,
                                                              double *,
                                                              double *,
                                                              double *,
                                                              unsigned,
                                                              DeepPolyElement * )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
{
    // Input layer should not have a predecessor
    ASSERT( false );
}

void DeepPolyInputElement::log( const String &message )
{
    if ( GlobalConfiguration::NETWORK_LEVEL_REASONER_LOGGING )
        printf( "DeepPolyInputElement: %s\n", message.ascii() );
}

} // namespace NLR
