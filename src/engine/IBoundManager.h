/*********************                                                        */
/*! \file IBoundManager.h
 ** \verbatim
 ** Top contributors (to current version):
 **  Aleksandar Zeljic, Haoze Wu,
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** IBoundManager is interface for implementation of a BoundManager class.
 **
 ** IBoundManager provides a method to obtain a new variable with:
 ** registerNewVariable().
 **
 ** There are two sets of methods to set bounds:
 **   * set*Bounds     - local method used to update bounds
 **   * tighten*Bounds - shared method to update bounds, propagates the new bounds
 **
 ** As soon as bounds become inconsistent, i.e. lowerBound > upperBound, a
 ** _consistentBounds flag is set which can be checked using consistentBounds() method.
 **
 **/

#ifndef __IBoundManager_h__
#define __IBoundManager_h__

<<<<<<< HEAD
#include <cstdint>
#include "List.h"
#include "Vector.h"
#include "PiecewiseLinearFunctionType.h"

enum BoundType : unsigned
{
=======
#include "List.h"
#include "PiecewiseLinearFunctionType.h"
#include "Vector.h"

#include <cstdint>

enum BoundType : unsigned {
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    LOWER = 0,
    UPPER = 1,
};
class BoundExplainer;
class SparseUnsortedList;
class TableauRow;
class Tightening;
class ITableau;
class IRowBoundTightener;
class IBoundManager
{
public:
<<<<<<< HEAD
    const static constexpr unsigned NO_VARIABLE_FOUND = UINT32_MAX -1;

    virtual ~IBoundManager() {};
=======
    const static constexpr unsigned NO_VARIABLE_FOUND = UINT32_MAX - 1;

    virtual ~IBoundManager(){};
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    /*
       Registers a new variable, grows the BoundManager size and bound vectors,
       initializes new bounds to +/-inf, and returns the identifier of the new
       variable.
     */
    virtual unsigned registerNewVariable() = 0;

    /*
       Initialize BoundManager to a given number of variables;
     */
    virtual void initialize( unsigned numberOfVariables ) = 0;

    /*
       Returns number of registered variables
     */
    virtual unsigned getNumberOfVariables() const = 0;

    /*
       Records a new bound if tighter than the current bound.
     */
    virtual bool tightenLowerBound( unsigned variable, double value ) = 0;
    virtual bool tightenUpperBound( unsigned variable, double value ) = 0;

    /*
       Sets the bould to the assigned value, checks bound consistency.
     */
    virtual bool setLowerBound( unsigned variable, double value ) = 0;
    virtual bool setUpperBound( unsigned variable, double value ) = 0;

    /*
       Return bound value.
     */
    virtual double getLowerBound( unsigned variable ) const = 0;
    virtual double getUpperBound( unsigned variable ) const = 0;

    /*
      Get pointers to latest bounds used for access by tableau and tighteners
     */
<<<<<<< HEAD
    virtual const double * getLowerBounds() const = 0;
    virtual const double * getUpperBounds() const = 0;
=======
    virtual const double *getLowerBounds() const = 0;
    virtual const double *getUpperBounds() const = 0;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    /*
     * Store and restore bounds after push/pop
     */
    /*
       Obtain a list of all the bound updates since the last call to
       getTightenings.
     */
    virtual void getTightenings( List<Tightening> &tightenings ) = 0;
    virtual void clearTightenings() = 0;

    /*
      Returns true if the bounds for the variable is valid. Used to
      detect a conflict state.
    */
    virtual bool consistentBounds() const = 0;

    /*
      Register Tableau for callbacks.
     */
    virtual void registerTableau( ITableau *tableau ) = 0;

    /*
       Register RowBoundTightener for callbacks.
     */
    virtual void registerRowBoundTightener( IRowBoundTightener *ptrRowBoundTightener ) = 0;

    /*
       Tighten bounds and update their explanations according to some object representing the row
    */
    virtual bool tightenLowerBound( unsigned variable, double value, const TableauRow &row ) = 0;
    virtual bool tightenUpperBound( unsigned variable, double value, const TableauRow &row ) = 0;

<<<<<<< HEAD
    virtual bool tightenLowerBound( unsigned variable, double value, const SparseUnsortedList &row ) = 0;
    virtual bool tightenUpperBound( unsigned variable, double value, const SparseUnsortedList &row ) = 0;
=======
    virtual bool
    tightenLowerBound( unsigned variable, double value, const SparseUnsortedList &row ) = 0;
    virtual bool
    tightenUpperBound( unsigned variable, double value, const SparseUnsortedList &row ) = 0;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    /*
      Add a lemma to the UNSATCertificateNode object
      Return true iff adding the lemma was successful
    */
<<<<<<< HEAD
    virtual bool addLemmaExplanationAndTightenBound( unsigned var, double value, BoundType affectedVarBound,
                                                    const List<unsigned> &causingVars, BoundType causingVarBound,
                                                    PiecewiseLinearFunctionType constraintType ) = 0;

    /*
      Return the content of the object containing all explanations for variable bounds in the tableau
=======
    virtual bool
    addLemmaExplanationAndTightenBound( unsigned var,
                                        double value,
                                        BoundType affectedVarBound,
                                        const List<unsigned> &causingVars,
                                        BoundType causingVarBound,
                                        PiecewiseLinearFunctionType constraintType ) = 0;

    /*
      Return the content of the object containing all explanations for variable bounds in the
      tableau
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    */
    virtual const BoundExplainer *getBoundExplainer() const = 0;

    /*
      Deep-copy the BoundExplainer object content
     */
<<<<<<< HEAD
    virtual void copyBoundExplainerContent( const BoundExplainer* boundExplainer ) = 0;
=======
    virtual void copyBoundExplainerContent( const BoundExplainer *boundExplainer ) = 0;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    /*
      Initialize the boundExplainer
     */
    virtual void initializeBoundExplainer( unsigned numberOfVariables, unsigned numberOfRows ) = 0;

    /*
      Given a row, updates the values of the bound explanations of a var according to the row
    */
    virtual void updateBoundExplanation( const TableauRow &row, bool isUpper, unsigned var ) = 0;

    /*
<<<<<<< HEAD
      Given a row as SparseUnsortedList, updates the values of the bound explanations of a var according to the row
    */
    virtual void updateBoundExplanationSparse( const SparseUnsortedList &row, bool isUpper, unsigned var ) = 0;
=======
      Given a row as SparseUnsortedList, updates the values of the bound explanations of a var
      according to the row
    */
    virtual void
    updateBoundExplanationSparse( const SparseUnsortedList &row, bool isUpper, unsigned var ) = 0;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    /*
      Get the index of a variable with inconsistent bounds, if exists, or -1 otherwise
    */
    virtual unsigned getInconsistentVariable() const = 0;

    /*
      Return true iff boundManager should produce proofs
    */
    virtual bool shouldProduceProofs() const = 0;
};

#endif // __IBoundManager_h__
