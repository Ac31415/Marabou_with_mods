/*********************                                                        */
/*! \file MockBoundManager.h
 ** \verbatim
 ** Top contributors (to current version):
 **  Aleksandar Zeljic, Haoze Wu,
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __MockBoundManager_h__
#define __MockBoundManager_h__

<<<<<<< HEAD
#include "IBoundManager.h"
#include "FloatUtils.h"
#include "List.h"
#include "Vector.h"
#include "Map.h"
#include "Tightening.h"
#include "MarabouError.h"
=======
#include "FloatUtils.h"
#include "IBoundManager.h"
#include "List.h"
#include "Map.h"
#include "MarabouError.h"
#include "Tightening.h"
#include "Vector.h"
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

class Tableau;
class MockBoundManager : public IBoundManager
{
public:
    MockBoundManager()
<<<<<<< HEAD
      : _size( 0 )
      , _lowerBounds( nullptr )
      , _upperBounds( nullptr )
      , _tightenedLower( nullptr )
      , _tightenedUpper( nullptr )
    {
    };
=======
        : _size( 0 )
        , _lowerBounds( nullptr )
        , _upperBounds( nullptr )
        , _tightenedLower( nullptr )
        , _tightenedUpper( nullptr ){};
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    ~MockBoundManager()
    {
        if ( _lowerBounds != nullptr )
<<<<<<< HEAD
            delete [] _lowerBounds;

        if ( _upperBounds != nullptr )
            delete [] _upperBounds;

        if ( _tightenedLower != nullptr )
            delete [] _tightenedLower;

        if ( _tightenedUpper != nullptr )
            delete [] _tightenedUpper;
=======
            delete[] _lowerBounds;

        if ( _upperBounds != nullptr )
            delete[] _upperBounds;

        if ( _tightenedLower != nullptr )
            delete[] _tightenedLower;

        if ( _tightenedUpper != nullptr )
            delete[] _tightenedUpper;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    };

    /*
       Registers a new variable, grows the BoundManager size and bound vectors,
       initializes new bounds to +/-inf, and returns the index of the new
       variable.
     */
    unsigned registerNewVariable()
    {
<<<<<<< HEAD
      return -1;
=======
        return -1;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    };

    /*
       Initialize local bounds
     */
    void initialize( unsigned size )
    {
<<<<<<< HEAD
      _lowerBounds = new double[size];
      _upperBounds = new double[size];
      _tightenedUpper = new bool[size];
      _tightenedLower = new bool[size];

      if ( _lowerBounds == nullptr || _upperBounds == nullptr ||
           _tightenedLower == nullptr || _tightenedUpper == nullptr )
        throw MarabouError( MarabouError::ALLOCATION_FAILED, "MockBoundManager" );

      _size = size;
=======
        _lowerBounds = new double[size];
        _upperBounds = new double[size];
        _tightenedUpper = new bool[size];
        _tightenedLower = new bool[size];

        if ( _lowerBounds == nullptr || _upperBounds == nullptr || _tightenedLower == nullptr ||
             _tightenedUpper == nullptr )
            throw MarabouError( MarabouError::ALLOCATION_FAILED, "MockBoundManager" );

        _size = size;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    }

    /*
       Returns number of registered variables
     */
    unsigned getNumberOfVariables() const
    {
<<<<<<< HEAD
      return _size;
=======
        return _size;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    };

    double getLowerBound( unsigned variable ) const
    {
        return _lowerBounds[variable];
    }

    /*
       Records a new bound if tighter the current bound.
     */
    bool tightenLowerBound( unsigned variable, double value )
    {
<<<<<<< HEAD
      if ( FloatUtils::lt( getLowerBound( variable), value ) )
      {
        setLowerBound( variable, value);
        _tightenedLower[variable] = true;
        return true;
      }

      return false;
=======
        if ( FloatUtils::lt( getLowerBound( variable ), value ) )
        {
            setLowerBound( variable, value );
            _tightenedLower[variable] = true;
            return true;
        }

        return false;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    }

    double getUpperBound( unsigned variable ) const
    {
<<<<<<< HEAD
      return _upperBounds[variable];
=======
        return _upperBounds[variable];
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    }

    bool tightenUpperBound( unsigned variable, double value )
    {
<<<<<<< HEAD
      if ( FloatUtils::gt( getUpperBound( variable ), value ) )
      {
        setUpperBound( variable, value);
        _tightenedUpper[variable] = true;
        return true;
      }

      return false;
=======
        if ( FloatUtils::gt( getUpperBound( variable ), value ) )
        {
            setUpperBound( variable, value );
            _tightenedUpper[variable] = true;
            return true;
        }

        return false;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    };

    /*
       Silently sets bounds to the assigned value, checks bound consistency.
     */
    bool setLowerBound( unsigned variable, double value )
    {
<<<<<<< HEAD
      _lowerBounds[variable] = value;
      return true;
=======
        _lowerBounds[variable] = value;
        return true;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    };

    bool setUpperBound( unsigned variable, double value )
    {
<<<<<<< HEAD
      _upperBounds[variable] = value;
      return true;
    };

    const double * getLowerBounds() const
=======
        _upperBounds[variable] = value;
        return true;
    };

    const double *getLowerBounds() const
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    {
        return _lowerBounds;
    };

<<<<<<< HEAD
    const double * getUpperBounds() const
=======
    const double *getUpperBounds() const
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    {
        return _upperBounds;
    };

<<<<<<< HEAD
     /*
       Obtain a list of all the bound updates since the last call to
       getTightenings.
     */
    void getTightenings( List<Tightening> &tightenings )
    {

=======
    /*
      Obtain a list of all the bound updates since the last call to
      getTightenings.
    */
    void getTightenings( List<Tightening> &tightenings )
    {
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
        for ( unsigned var = 0; var < _size; ++var )
        {
            if ( _tightenedLower[var] )
            {
<<<<<<< HEAD
                tightenings.append(
                    Tightening( var, _lowerBounds[var], Tightening::LB ) );
=======
                tightenings.append( Tightening( var, _lowerBounds[var], Tightening::LB ) );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
                _tightenedLower[var] = false;
            }
        }

        for ( unsigned var = 0; var < _size; ++var )
        {
            if ( _tightenedUpper[var] )
            {
<<<<<<< HEAD
                tightenings.append(
                    Tightening( var, _upperBounds[var], Tightening::UB ) );
=======
                tightenings.append( Tightening( var, _upperBounds[var], Tightening::UB ) );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
                _tightenedUpper[var] = false;
            }
        }
    }
    /*
       Clear tightened flags;
     */
    void clearTightenings()
    {
        for ( unsigned var = 0; var < _size; ++var )
        {
            _tightenedLower[var] = false;
            _tightenedUpper[var] = false;
        }
    };

    /*
      Returns true if the bounds of all variables are consistent. Used to
      detect a conflict state.
    */
    bool consistentBounds() const
    {
<<<<<<< HEAD
      for ( unsigned i = 0; i < _size; ++i )
        if ( !consistentBounds( i ) )
          return false;

      return true;
=======
        for ( unsigned i = 0; i < _size; ++i )
            if ( !consistentBounds( i ) )
                return false;

        return true;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    }

    /*
      Returns true if the bounds for the variable is valid, used to
      detect a conflict state.
    */
    bool consistentBounds( unsigned variable ) const
    {
<<<<<<< HEAD
      return getLowerBound( variable ) <= getUpperBound( variable );
    };

    void registerTableau( ITableau */* tableau */ )
    {
    }

    void registerRowBoundTightener( IRowBoundTightener */* rowBoundTightener */ )
    {
    }

    const BoundExplainer* getBoundExplainer() const
=======
        return getLowerBound( variable ) <= getUpperBound( variable );
    };

    void registerTableau( ITableau * /* tableau */ )
    {
    }

    void registerRowBoundTightener( IRowBoundTightener * /* rowBoundTightener */ )
    {
    }

    const BoundExplainer *getBoundExplainer() const
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    {
        return NULL;
    }

<<<<<<< HEAD
    void copyBoundExplainerContent( const BoundExplainer* /* boundsExplanations */ )
    {
    }

    void updateBoundExplanation( const TableauRow &/* row */, bool /* isUpper */, unsigned /* var */ )
    {
    }

    void updateBoundExplanationSparse( const SparseUnsortedList &/* row */, bool /* isUpper */, unsigned /* var */ )
=======
    void copyBoundExplainerContent( const BoundExplainer * /* boundsExplanations */ )
    {
    }

    void
    updateBoundExplanation( const TableauRow & /* row */, bool /* isUpper */, unsigned /* var */ )
    {
    }

    void updateBoundExplanationSparse( const SparseUnsortedList & /* row */,
                                       bool /* isUpper */,
                                       unsigned /* var */ )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    {
    }

private:
    unsigned _size;

    double *_lowerBounds;
    double *_upperBounds;
    bool *_tightenedLower;
    bool *_tightenedUpper;
    /* Map<unsigned, double> _lowerBounds; */
    /* Map<unsigned, double> _upperBounds; */
    /* Map<unsigned, bool> _tightenedLower; */
    /* Map<unsigned, bool> _tightenedUpper; */

<<<<<<< HEAD
    bool tightenLowerBound( unsigned variable, double value, const TableauRow &/* row */ )
=======
    bool tightenLowerBound( unsigned variable, double value, const TableauRow & /* row */ )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    {
        return tightenLowerBound( variable, value );
    }

<<<<<<< HEAD
    bool tightenUpperBound( unsigned variable, double value, const TableauRow &/* row */ )
=======
    bool tightenUpperBound( unsigned variable, double value, const TableauRow & /* row */ )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    {
        return tightenUpperBound( variable, value );
    }

<<<<<<< HEAD
    bool tightenLowerBound( unsigned variable, double value, const SparseUnsortedList &/* row */ )
=======
    bool tightenLowerBound( unsigned variable, double value, const SparseUnsortedList & /* row */ )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    {
        return tightenLowerBound( variable, value );
    }

<<<<<<< HEAD
    bool tightenUpperBound( unsigned variable, double value, const SparseUnsortedList &/* row */ )
=======
    bool tightenUpperBound( unsigned variable, double value, const SparseUnsortedList & /* row */ )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    {
        return tightenUpperBound( variable, value );
    }

<<<<<<< HEAD
    bool addLemmaExplanationAndTightenBound(unsigned /* var */, double /* value */, BoundType /* affectedVarBound */,
                                            const List<unsigned> &/* causingVar */, BoundType /* causingVarBound */,
                                            PiecewiseLinearFunctionType /* constraintType */ )
=======
    bool addLemmaExplanationAndTightenBound( unsigned /* var */,
                                             double /* value */,
                                             BoundType /* affectedVarBound */,
                                             const List<unsigned> & /* causingVar */,
                                             BoundType /* causingVarBound */,
                                             PiecewiseLinearFunctionType /* constraintType */ )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    {
        return true;
    }

    void initializeBoundExplainer( unsigned /* numberOfVariables */, unsigned /* numberOfRows */ )
    {
    }

    unsigned getInconsistentVariable() const
    {
        return NO_VARIABLE_FOUND;
    }

    bool shouldProduceProofs() const
    {
        return false;
    }
};

#endif // __MockBoundManager_h__
<<<<<<< HEAD

  
=======
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
