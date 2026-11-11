/*********************                                                        */
/*! \file TensorUtils.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Matthew Daggitt
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** Utilities for working with tensors while parsing networks.
<<<<<<< HEAD
**/

#include "TensorUtils.h"
=======
 **/

#include "TensorUtils.h"

>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
#include <math.h>

TensorIndices unpackIndex( TensorShape shape, PackedTensorIndices packedIndex )
{
<<<<<<< HEAD
    ASSERT ( packedIndex < tensorSize( shape ) );
=======
    ASSERT( packedIndex < tensorSize( shape ) );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    TensorIndices indices;
    int currentIndex = packedIndex;
    for ( int i = shape.size() - 1; i >= 0; i-- )
    {
        int dimension = shape[i];
        int index = currentIndex % dimension;
<<<<<<< HEAD
        currentIndex = (currentIndex - index) / dimension;
=======
        currentIndex = ( currentIndex - index ) / dimension;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
        indices.insertHead( index );
    }
    return indices;
}

<<<<<<< HEAD
PackedTensorIndices packIndex ( TensorShape shape, TensorIndices indices )
=======
PackedTensorIndices packIndex( TensorShape shape, TensorIndices indices )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
{
    ASSERT( shape.size() == indices.size() );

    unsigned int sizeSoFar = 1;
    unsigned int index = 0;
    for ( unsigned int i = shape.size(); i-- > 0; )
    {
<<<<<<< HEAD
        ASSERT ( indices[i] <= shape[i] );
        index += sizeSoFar*indices[i];
        sizeSoFar *= shape[i];
    }
    ASSERT( tensorSize ( shape ) == sizeSoFar );
=======
        ASSERT( indices[i] <= shape[i] );
        index += sizeSoFar * indices[i];
        sizeSoFar *= shape[i];
    }
    ASSERT( tensorSize( shape ) == sizeSoFar );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    return index;
}

unsigned int tensorSize( TensorShape shape )
{
    unsigned int size = 1;
    for ( unsigned int dimSize : shape )
    {
        size *= dimSize;
    }
    return size;
}

// See https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md#multidirectional-broadcasting
TensorShape getMultidirectionalBroadcastShape( TensorShape shape1, TensorShape shape2 )
{
    TensorShape output;
    auto it1 = shape1.rbegin();
    auto it2 = shape2.rbegin();
    while ( it1 != shape1.rend() || it2 != shape2.rend() )
    {
<<<<<<< HEAD
        int d1 = it1 == shape1.rend() ? 1 : *(it1++);
        int d2 = it2 == shape2.rend() ? 1 : *(it2++);

        ASSERT ( d1 == d2 || d2 == 1 || d1 == 1 );
=======
        int d1 = it1 == shape1.rend() ? 1 : *( it1++ );
        int d2 = it2 == shape2.rend() ? 1 : *( it2++ );

        ASSERT( d1 == d2 || d2 == 1 || d1 == 1 );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

        int d;
        if ( d1 == d2 || d2 == 1 )
        {
            d = d1;
        }
        else
        {
            d = d2;
        }

        output.insertHead( d );
    }
    return output;
}

/**
 * @brief Broadcasts the provided indices into those into the current tensor shape
 * from indices in the desired broadcast shape.
 */
<<<<<<< HEAD
TensorIndices broadcastIndex ( TensorShape currentShape, TensorShape broadcastShape, TensorIndices broadcastIndices )
{
    ASSERT ( broadcastIndices.size() == broadcastShape.size() );
=======
TensorIndices broadcastIndex( TensorShape currentShape,
                              TensorShape broadcastShape,
                              TensorIndices broadcastIndices )
{
    ASSERT( broadcastIndices.size() == broadcastShape.size() );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    int dimOffset = broadcastShape.size() - currentShape.size();
    TensorIndices result;
    for ( size_t i = dimOffset; i < broadcastShape.size(); i++ )
    {
        int cd = currentShape[i - dimOffset];
        int bd = broadcastShape[i];

<<<<<<< HEAD
        ASSERT ( cd == 1 || cd == bd );
=======
        ASSERT( cd == 1 || cd == bd );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

        if ( cd == bd )
        {
            result.append( broadcastIndices[i] );
        }
        else
        {
<<<<<<< HEAD
            result.append ( 0 );
=======
            result.append( 0 );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
        }
    }
    return result;
}

<<<<<<< HEAD
Padding::Padding ( int padFront, int padBack )
    : padFront( padFront )
    , padBack( padBack )
{}

Padding calculatePaddingNeeded( int inputSize, int filterSize, int stride, bool padFrontPreferentially )
=======
TensorIndex unsignIndex( unsigned int size, SignedTensorIndex signedIndex )
{
    if ( signedIndex >= 0 )
    {
        TensorIndex result = static_cast<unsigned int>( signedIndex );
        ASSERT( result < size );
        return result;
    }

    unsigned int result = static_cast<unsigned int>( -signedIndex );
    ASSERT( result <= size );
    return size - result;
}

Padding::Padding( int padFront, int padBack )
    : padFront( padFront )
    , padBack( padBack )
{
}

Padding
calculatePaddingNeeded( int inputSize, int filterSize, int stride, bool padFrontPreferentially )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
{
    int overrun = inputSize % stride;
    int paddingNeeded;
    if ( overrun == 0 )
    {
        paddingNeeded = std::max( filterSize - stride, 0 );
    }
    else
    {
        paddingNeeded = std::max( filterSize - overrun, 0 );
    }
    const int halfPaddingQuot = paddingNeeded / 2;
    const int halfPaddingRem = paddingNeeded % 2;
    if ( padFrontPreferentially )
    {
<<<<<<< HEAD
        return Padding ( halfPaddingQuot + halfPaddingRem, halfPaddingQuot );
    }
    else
    {
        return Padding ( halfPaddingQuot, halfPaddingQuot + halfPaddingRem );
=======
        return Padding( halfPaddingQuot + halfPaddingRem, halfPaddingQuot );
    }
    else
    {
        return Padding( halfPaddingQuot, halfPaddingQuot + halfPaddingRem );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    }
}

Permutation reversePermutation( unsigned int size )
{
    Permutation result;
    for ( unsigned int i = size - 1; i-- > 0; )
    {
<<<<<<< HEAD
        result.append(i);
=======
        result.append( i );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    }
    return result;
}
