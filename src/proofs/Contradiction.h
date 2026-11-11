/*********************                                                        */
/*! \file Contradiction.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Omri Isac, Guy Katz
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2022 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]
 **/

#ifndef __Contradiction_h__
#define __Contradiction_h__

#include "SparseUnsortedList.h"
#include "Vector.h"

/*
<<<<<<< HEAD
  Contains all info relevant for a simple Marabou contradiction - i.e. explanations of contradicting bounds of a variable
=======
  Contains all info relevant for a simple Marabou contradiction - i.e. explanations of contradicting
  bounds of a variable
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
*/
class Contradiction
{
public:
    Contradiction( const Vector<double> &contradiction );
    Contradiction( unsigned var );

    ~Contradiction();

    /*
      Getters for all fields
     */
    unsigned getVar() const;
    const SparseUnsortedList &getContradiction() const;

private:
    unsigned _var;
    SparseUnsortedList _contradiction;
};

#endif //__Contradiction_h__
