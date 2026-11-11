/*********************                                                        */
/*! \file LPSolverType.h
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

#ifndef __LPSolverType_h__
#define __LPSolverType_h__

<<<<<<< HEAD
enum class LPSolverType
{
=======
enum class LPSolverType {
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    // Native simplex implementation. Open-sourced, more agressive
    // bound derivation, and will support proof-production.
    NATIVE,

    // Not open-sourced. Use this for optimal run-time performance.
    GUROBI,
};

#endif // __LPSolverType_h__
