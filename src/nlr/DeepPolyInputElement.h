/*********************                                                        */
/*! \file DeepPolyInputElement.h
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

#ifndef __DeepPolyInputElement_h__
#define __DeepPolyInputElement_h__

#include "DeepPolyElement.h"
#include "Layer.h"
#include "MStringf.h"
#include "NLRError.h"
<<<<<<< HEAD
=======

>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
#include <climits>

namespace NLR {

class DeepPolyInputElement : public DeepPolyElement
{
public:
    DeepPolyInputElement( Layer *layer );
    ~DeepPolyInputElement();

<<<<<<< HEAD
    void execute( const Map<unsigned, DeepPolyElement *>
                  &deepPolyElementsBefore );

    void symbolicBoundInTermsOfPredecessor
    ( const double *symbolicLb, const double*symbolicUb, double
      *symbolicLowerBias, double *symbolicUpperBias, double
      *symbolicLbInTermsOfPredecessor, double *symbolicUbInTermsOfPredecessor,
      unsigned targetLayerSize, DeepPolyElement *predecessor );
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
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

private:
    void log( const String &message );
};

} // namespace NLR

#endif // __DeepPolyInputElement_h__
