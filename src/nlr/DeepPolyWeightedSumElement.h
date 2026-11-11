/*********************                                                        */
/*! \file DeepPolyWeightedSumElement.h
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

#ifndef __DeepPolyWeightedSumElement_h__
#define __DeepPolyWeightedSumElement_h__

#include "DeepPolyElement.h"
#include "Layer.h"
#include "MStringf.h"
#include "NLRError.h"
<<<<<<< HEAD
=======

>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
#include <climits>

namespace NLR {

class DeepPolyWeightedSumElement : public DeepPolyElement
{
public:
<<<<<<< HEAD

=======
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    DeepPolyWeightedSumElement( Layer *layer );
    ~DeepPolyWeightedSumElement();

    void execute( const Map<unsigned, DeepPolyElement *> &deepPolyElements );
<<<<<<< HEAD
    void symbolicBoundInTermsOfPredecessor
    ( const double *symbolicLb, const double*symbolicUb, double
      *symbolicLowerBias, double *symbolicUpperBias, double
      *symbolicLbInTermsOfPredecessor, double *symbolicUbInTermsOfPredecessor,
      unsigned targetLayerSize, DeepPolyElement *predecessor );

private:

=======
    void symbolicBoundInTermsOfPredecessor( const double *symbolicLb,
                                            const double *symbolicUb,
                                            double *symbolicLowerBias,
                                            double *symbolicUpperBias,
                                            double *symbolicLbInTermsOfPredecessor,
                                            double *symbolicUbInTermsOfPredecessor,
                                            unsigned targetLayerSize,
                                            DeepPolyElement *predecessor );

private:
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    /*
      Memory allocated to store concrete bounds computed at different stages
      of back substitution.
    */
    double *_workLb;
    double *_workUb;

<<<<<<< HEAD
    Set<unsigned>  _residualLayerIndices;
    Map<unsigned, double *>  _residualLb;
    Map<unsigned, double *>  _residualUb;
=======
    Set<unsigned> _residualLayerIndices;
    Map<unsigned, double *> _residualLb;
    Map<unsigned, double *> _residualUb;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    /*
      Compute the concrete upper- and lower- bounds of this layer by concretizing
      the symbolic bounds with respect to every preceding element.
    */
<<<<<<< HEAD
    void computeBoundWithBackSubstitution( const Map<unsigned, DeepPolyElement *>
                                           &deepPolyElementsBefore );
=======
    void computeBoundWithBackSubstitution(
        const Map<unsigned, DeepPolyElement *> &deepPolyElementsBefore );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    /*
      Compute concrete bounds using symbolic bounds with respect to a
      sourceElement.
    */
<<<<<<< HEAD
    void concretizeSymbolicBound( const double *symbolicLb, const double
                                  *symbolicUb, const double *symbolicLowerBias,
                                  const double *symbolicUpperBias,
                                  DeepPolyElement *sourceElement,
                                  const Map<unsigned, DeepPolyElement *>
                                  &deepPolyElementsBefore );

    void concretizeSymbolicBoundForSourceLayer( const double *symbolicLb,
                                                const double*symbolicUb,
=======
    void concretizeSymbolicBound( const double *symbolicLb,
                                  const double *symbolicUb,
                                  const double *symbolicLowerBias,
                                  const double *symbolicUpperBias,
                                  DeepPolyElement *sourceElement,
                                  const Map<unsigned, DeepPolyElement *> &deepPolyElementsBefore );

    void concretizeSymbolicBoundForSourceLayer( const double *symbolicLb,
                                                const double *symbolicUb,
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
                                                const double *symbolicLowerBias,
                                                const double *symbolicUpperBias,
                                                DeepPolyElement *sourceElement );

    void allocateMemoryForResidualsIfNeeded( unsigned residualLayerIndex,
                                             unsigned residualLayerSize );
    void allocateMemory();
    void freeMemoryIfNeeded();
    void log( const String &message );
};

} // namespace NLR

#endif // __DeepPolyWeightedSumElement_h__
