/*********************                                                        */
/*! \file SoISearchStrategy.h
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

#ifndef __SoISearchStrategy_h__
#define __SoISearchStrategy_h__

<<<<<<< HEAD
enum class SoISearchStrategy
{
=======
enum class SoISearchStrategy {
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    // Flip the SoI cost term of a random ReLU,
    // accept the change with certain probability
    MCMC,
    // Flip the SoI cost term to reduce the overall soi greedily.
    // When hitting plateau, flip a random ReLU
    WALKSAT,
};

#endif // __SoISearchStrategy_h__
