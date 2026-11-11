/*********************                                                        */
/*! \file DeepPolyAbsoluteValueElement.h
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

#ifndef __DeepPolyAbsoluteValueElement_h__
#define __DeepPolyAbsoluteValueElement_h__

#include "DeepPolyElement.h"
#include "Layer.h"
#include "MStringf.h"
#include "NLRError.h"
<<<<<<< HEAD
=======

>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
#include <climits>

namespace NLR {

class DeepPolyAbsoluteValueElement : public DeepPolyElement
{
public:
    DeepPolyAbsoluteValueElement( Layer *layer );
    ~DeepPolyAbsoluteValueElement();

<<<<<<< HEAD
    void execute( const Map<unsigned, DeepPolyElement *>
                  &deepPolyElementsBefore );

    void symbolicBoundInTermsOfPredecessor
    ( const double *symbolicLb, const double*symbolicUb, double
      *symbolicLowerBias, double *symbolicUpperBias, double
      *symbolicLbInTermsOfPredecessor, double *symbolicUbInTermsOfPredecessor,
      unsigned targetLayerSize, DeepPolyElement *predecessor );

private:

=======
    void execute( const Map<unsigned, DeepPolyElement *> &deepPolyElementsBefore );

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
    void allocateMemory();
    void freeMemoryIfNeeded();
    void log( const String &message );
};

} // namespace NLR

#endif // __DeepPolyElement_h__
