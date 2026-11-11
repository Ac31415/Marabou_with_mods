/*********************                                                        */
/*! \file SigmoidConstraint.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Teruhiro Tagomori, Haoze Wu
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

**/

#ifndef __SigmoidConstraint_h__
#define __SigmoidConstraint_h__

#include "List.h"
#include "Map.h"
<<<<<<< HEAD
#include "TranscendentalConstraint.h"

#include <cmath>

class SigmoidConstraint : public TranscendentalConstraint
=======
#include "NonlinearConstraint.h"

#include <cmath>

class SigmoidConstraint : public NonlinearConstraint
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
{
public:
    SigmoidConstraint( unsigned b, unsigned f );
    SigmoidConstraint( const String &serializedSigmoid );

    /*
      Get the type of this constraint.
    */
<<<<<<< HEAD
    TranscendentalFunctionType getType() const override;
=======
    NonlinearFunctionType getType() const override;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    /*
      Return a clone of the constraint.
    */
<<<<<<< HEAD
    TranscendentalConstraint *duplicateConstraint() const override;
=======
    NonlinearConstraint *duplicateConstraint() const override;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    /*
      Restore the state of this constraint from the given one.
    */
<<<<<<< HEAD
    void restoreState( const TranscendentalConstraint *state ) override; 
=======
    void restoreState( const NonlinearConstraint *state ) override;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    /*
      Register/unregister the constraint with a talbeau.
     */
    void registerAsWatcher( ITableau *tableau ) override;
    void unregisterAsWatcher( ITableau *tableau ) override;

    /*
      These callbacks are invoked when a watched variable's value
      changes, or when its bounds change.
    */
    void notifyLowerBound( unsigned variable, double bound ) override;
    void notifyUpperBound( unsigned variable, double bound ) override;

    /*
<<<<<<< HEAD
      Returns true iff the variable participates in this transcendental constraint
=======
      Returns true iff the variable participates in this nonlinear constraint
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    */
    bool participatingVariable( unsigned variable ) const override;

    /*
      Get the list of variables participating in this constraint.
    */
    List<unsigned> getParticipatingVariables() const override;

    /*
      Preprocessing related functions, to inform that a variable has
      been eliminated completely because it was fixed to some value,
      or that a variable's index has changed (e.g., x4 is now called
      x2). constraintObsolete() returns true iff and the constraint
      has become obsolote as a result of variable eliminations.
    */
    void eliminateVariable( unsigned variable, double fixedValue ) override;
    void updateVariableIndex( unsigned oldIndex, unsigned newIndex ) override;
    bool constraintObsolete() const override;

<<<<<<< HEAD
=======
    bool supportVariableElimination() const override
    {
        return true;
    };

>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    /*
      Get the tightenings entailed by the constraint.
    */
    void getEntailedTightenings( List<Tightening> &tightenings ) const override;

    /*
<<<<<<< HEAD
=======
      Returns true iff the assignment satisfies the constraint.
    */
    bool satisfied() const override;

    /*
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
      Dump the current state of the constraint.
    */
    void dump( String &output ) const override;

    /*
      Returns string with shape: sigmoid, _f, _b
    */
    String serializeToString() const override;

    /*
      Get the index of the B and F variables.
    */
    unsigned getB() const;
    unsigned getF() const;

    /*
      Compute the sigmoid function.
    */
    static double sigmoid( double x );

    /*
      Compute the inverse of the sigmoid function.
    */
    static double sigmoidInverse( double y );

    /*
      Compute the derivative of the sigmoid function.
    */
    static double sigmoidDerivative( double x );

private:
<<<<<<< HEAD
    unsigned _b, _f; 
=======
    unsigned _b, _f;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    bool _haveEliminatedVariables;
};

#endif // __SigmoidConstraint_h__
