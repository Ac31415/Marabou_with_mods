/*********************                                                        */
/*! \file DeepPolyAnalysis.h
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

#ifndef __DeepPolyAnalysis_h__
#define __DeepPolyAnalysis_h__

#include "DeepPolyElement.h"
#include "Layer.h"
#include "LayerOwner.h"
#include "Map.h"
<<<<<<< HEAD
=======

>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
#include <climits>

namespace NLR {

class DeepPolyAnalysis
{
public:
<<<<<<< HEAD

=======
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    DeepPolyAnalysis( LayerOwner *layerOwner );
    ~DeepPolyAnalysis();

    void run();

private:
    LayerOwner *_layerOwner;

    /*
      Maps layer index to the abstract element
    */
    Map<unsigned, DeepPolyElement *> _deepPolyElements;

    /*
      Working memory for the abstract elements to execute
    */
<<<<<<< HEAD
    double * _work1SymbolicLb;
    double * _work1SymbolicUb;
    double * _work2SymbolicLb;
    double * _work2SymbolicUb;
    double * _workSymbolicLowerBias;
    double * _workSymbolicUpperBias;

    void allocateMemory( const Map<unsigned, Layer *> &layers );
=======
    double *_work1SymbolicLb;
    double *_work1SymbolicUb;
    double *_work2SymbolicLb;
    double *_work2SymbolicUb;
    double *_workSymbolicLowerBias;
    double *_workSymbolicUpperBias;

    unsigned _maxLayerSize;

    void allocateMemory();
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    void freeMemoryIfNeeded();

    DeepPolyElement *createDeepPolyElement( Layer *layer );

    void log( const String &message );
};

} // namespace NLR

#endif // __DeepPolyAnalysis_h__
