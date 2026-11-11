/*********************                                                        */
/*! \file Checker.h
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

#ifndef __Checker_h__
#define __Checker_h__

#include "AbsoluteValueConstraint.h"
#include "DisjunctionConstraint.h"
#include "MaxConstraint.h"
#include "Set.h"
#include "Stack.h"
#include "UnsatCertificateNode.h"

/*
  A class responsible to certify the UnsatCertificate
*/
class Checker
{
public:
    Checker( const UnsatCertificateNode *root,
             unsigned proofSize,
             const SparseMatrix *initialTableau,
             const Vector<double> &groundUpperBounds,
             const Vector<double> &groundLowerBounds,
             const List<PiecewiseLinearConstraint *> &_problemConstraints );

    /*
      Checks if the tree is indeed a correct proof of unsatisfiability.
<<<<<<< HEAD
      If called from a certificate of a satisfiable query, checks that all proofs for bound propagations and unsatisfiable leaves are correct
=======
      If called from a certificate of a satisfiable query, checks that all proofs for bound
      propagations and unsatisfiable leaves are correct
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    */
    bool check();

private:
    // The root of the tree to check
    const UnsatCertificateNode *_root;
    unsigned _proofSize;

    // Additional context data
    const SparseMatrix *_initialTableau;

    Vector<double> _groundUpperBounds;
    Vector<double> _groundLowerBounds;

    List<PiecewiseLinearConstraint *> _problemConstraints;

    unsigned _delegationCounter;

<<<<<<< HEAD
    // Keeps track of bounds changes, so only stored bounds will be reverted when traversing the tree
=======
    // Keeps track of bounds changes, so only stored bounds will be reverted when traversing the
    // tree
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    Stack<Set<unsigned>> _upperBoundChanges;
    Stack<Set<unsigned>> _lowerBoundChanges;

    /*
      Checks a node in the certificate tree
    */
    bool checkNode( const UnsatCertificateNode *node );

    /*
<<<<<<< HEAD
      Return true iff all changes in the ground bounds are certified, with tolerance to errors with at most size epsilon
=======
      Return true iff all changes in the ground bounds are certified, with tolerance to errors with
      at most size epsilon
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    */
    bool checkAllPLCExplanations( const UnsatCertificateNode *node, double epsilon );

    /*
      Return a change in the ground bounds caused by a ReLU constraint.
    */
<<<<<<< HEAD
    double checkReluLemma( const PLCLemma &expl, PiecewiseLinearConstraint &constraint, double epsilon );
=======
    double
    checkReluLemma( const PLCLemma &expl, PiecewiseLinearConstraint &constraint, double epsilon );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    /*
      Return a change in the ground bounds caused by a Sign constraint.
    */
<<<<<<< HEAD
    double checkSignLemma( const PLCLemma &expl, PiecewiseLinearConstraint &constraint, double epsilon );
=======
    double
    checkSignLemma( const PLCLemma &expl, PiecewiseLinearConstraint &constraint, double epsilon );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    /*
      Return a change in the ground bounds caused by a Absolute Value constraint.
    */
<<<<<<< HEAD
    double checkAbsLemma( const PLCLemma &expl, PiecewiseLinearConstraint &constraint, double epsilon );
=======
    double
    checkAbsLemma( const PLCLemma &expl, PiecewiseLinearConstraint &constraint, double epsilon );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    /*
      Return a change in the ground bounds caused by a Max constraint.
    */
    double checkMaxLemma( const PLCLemma &expl, PiecewiseLinearConstraint &constraint );

    /*
      Checks a contradiction
    */
    bool checkContradiction( const UnsatCertificateNode *node ) const;

    /*
      Computes a bound according to an explanation
    */
    double explainBound( unsigned var, bool isUpper, const SparseUnsortedList &explanation ) const;

    /*
      Write the data marked to delegate to a smtlib file format
    */
    void writeToFile();

    /*
      Return a pointer to the problem constraint representing the split
    */
<<<<<<< HEAD
    PiecewiseLinearConstraint *getCorrespondingConstraint( const List<PiecewiseLinearCaseSplit> &splits );
=======
    PiecewiseLinearConstraint *
    getCorrespondingConstraint( const List<PiecewiseLinearCaseSplit> &splits );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    /*
      Return a pointer to a ReLU problem constraint representing the split
    */
<<<<<<< HEAD
    PiecewiseLinearConstraint *getCorrespondingReluConstraint( const List<PiecewiseLinearCaseSplit> &splits );
=======
    PiecewiseLinearConstraint *
    getCorrespondingReluConstraint( const List<PiecewiseLinearCaseSplit> &splits );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    /*
      Return a pointer to a sign problem constraint representing the split
    */
<<<<<<< HEAD
    PiecewiseLinearConstraint *getCorrespondingSignConstraint( const List<PiecewiseLinearCaseSplit> &splits );
=======
    PiecewiseLinearConstraint *
    getCorrespondingSignConstraint( const List<PiecewiseLinearCaseSplit> &splits );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    /*
      Return a pointer to a absolute value problem constraint representing the split
    */
<<<<<<< HEAD
    PiecewiseLinearConstraint *getCorrespondingAbsConstraint( const List<PiecewiseLinearCaseSplit> &splits );
=======
    PiecewiseLinearConstraint *
    getCorrespondingAbsConstraint( const List<PiecewiseLinearCaseSplit> &splits );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    /*
      Return a pointer to a max problem constraint representing the split
    */
<<<<<<< HEAD
    PiecewiseLinearConstraint *getCorrespondingMaxConstraint( const List<PiecewiseLinearCaseSplit> &splits );
=======
    PiecewiseLinearConstraint *
    getCorrespondingMaxConstraint( const List<PiecewiseLinearCaseSplit> &splits );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    /*
      Return a pointer to a disjunction problem constraint representing the split
    */
<<<<<<< HEAD
    PiecewiseLinearConstraint *getCorrespondingDisjunctionConstraint( const List<PiecewiseLinearCaseSplit> &splits );
=======
    PiecewiseLinearConstraint *
    getCorrespondingDisjunctionConstraint( const List<PiecewiseLinearCaseSplit> &splits );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    /*
      Return true iff a list of splits represents a splits over a single variable
    */
    bool checkSingleVarSplits( const List<PiecewiseLinearCaseSplit> &splits );

    /*
      Fix phase of the child split
    */
<<<<<<< HEAD
    void fixChildSplitPhase( UnsatCertificateNode *child,  PiecewiseLinearConstraint *childrenSplitConstraint );
=======
    void fixChildSplitPhase( UnsatCertificateNode *child,
                             PiecewiseLinearConstraint *childrenSplitConstraint );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
};

#endif //__Checker_h__