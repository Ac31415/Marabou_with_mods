/*********************                                                        */
/*! \file BoundManager.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Haoze Wu, Aleksandar Zeljic
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

**/

#include "BoundManager.h"

#include "Debug.h"
#include "FloatUtils.h"
#include "InfeasibleQueryException.h"
<<<<<<< HEAD
#include "Tableau.h"
#include "Tightening.h"
#include "MarabouError.h"
=======
#include "MarabouError.h"
#include "Tableau.h"
#include "Tightening.h"
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

using namespace CVC4::context;

BoundManager::BoundManager( Context &context )
    : _context( context )
    , _size( 0 )
    , _allocated( 0 )
    , _tableau( nullptr )
    , _rowBoundTightener( nullptr )
    , _engine( nullptr )
    , _consistentBounds( &_context )
    , _firstInconsistentTightening( 0, 0.0, Tightening::LB )
    , _lowerBounds( nullptr )
    , _upperBounds( nullptr )
    , _boundExplainer( nullptr )
{
    _consistentBounds = true;
};

BoundManager::~BoundManager()
{
    if ( _lowerBounds )
    {
        delete[] _lowerBounds;
        _lowerBounds = nullptr;
    }

    if ( _upperBounds )
    {
        delete[] _upperBounds;
        _upperBounds = nullptr;
    }

    for ( unsigned i = 0; i < _size; ++i )
    {
        _storedLowerBounds[i]->deleteSelf();
        _storedUpperBounds[i]->deleteSelf();
        _tightenedLower[i]->deleteSelf();
        _tightenedUpper[i]->deleteSelf();
    }

    if ( _boundExplainer )
    {
        delete _boundExplainer;
        _boundExplainer = NULL;
    }
};

void BoundManager::initialize( unsigned numberOfVariables )
{
    ASSERT( _size == 0 );

    allocateLocalBounds( numberOfVariables );

    for ( unsigned i = 0; i < numberOfVariables; ++i )
        registerNewVariable();

    ASSERT( _size == numberOfVariables );
    ASSERT( _allocated == numberOfVariables );
}

void BoundManager::allocateLocalBounds( unsigned size )
{
    _lowerBounds = new double[size];
    if ( !_lowerBounds )
        throw MarabouError( MarabouError::ALLOCATION_FAILED, "BoundManager::lowerBounds" );
    std::fill_n( _lowerBounds, size, FloatUtils::negativeInfinity() );

    _upperBounds = new double[size];
    if ( !_upperBounds )
        throw MarabouError( MarabouError::ALLOCATION_FAILED, "BoundManager::upperBounds" );
    std::fill_n( _upperBounds, size, FloatUtils::infinity() );
    _allocated = size;

    if ( _tableau )
<<<<<<< HEAD
      _tableau->setBoundsPointers( _lowerBounds, _upperBounds );

    if ( _rowBoundTightener )
      _rowBoundTightener->setBoundsPointers( _lowerBounds, _upperBounds );
=======
        _tableau->setBoundsPointers( _lowerBounds, _upperBounds );

    if ( _rowBoundTightener )
        _rowBoundTightener->setBoundsPointers( _lowerBounds, _upperBounds );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
}

unsigned BoundManager::registerNewVariable()
{
    ASSERT( _size == _storedLowerBounds.size() );
    ASSERT( _size == _storedUpperBounds.size() );
    ASSERT( _size == _tightenedLower.size() );
    ASSERT( _size == _tightenedUpper.size() );

    unsigned newVar = _size++;

    if ( _allocated < _size )
    {
<<<<<<< HEAD
      double * oldLowerBounds = _upperBounds;
      double * oldUpperBounds = _lowerBounds;

      allocateLocalBounds( 2*_allocated );
      std::memcpy( _lowerBounds, oldLowerBounds, _allocated );
      std::memcpy( _upperBounds, oldUpperBounds, _allocated );
      _allocated *= 2;

      delete[] oldLowerBounds;
      delete[] oldUpperBounds;
=======
        double *oldLowerBounds = _upperBounds;
        double *oldUpperBounds = _lowerBounds;

        allocateLocalBounds( 2 * _allocated );
        std::memcpy( _lowerBounds, oldLowerBounds, _allocated );
        std::memcpy( _upperBounds, oldUpperBounds, _allocated );
        _allocated *= 2;

        delete[] oldLowerBounds;
        delete[] oldUpperBounds;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    }

    _storedLowerBounds.append( new ( true ) CDO<double>( &_context ) );
    _storedUpperBounds.append( new ( true ) CDO<double>( &_context ) );
    _tightenedLower.append( new ( true ) CDO<bool>( &_context ) );
    _tightenedUpper.append( new ( true ) CDO<bool>( &_context ) );

    *_storedLowerBounds[newVar] = FloatUtils::negativeInfinity();
    *_storedUpperBounds[newVar] = FloatUtils::infinity();
    *_tightenedLower[newVar] = false;
    *_tightenedUpper[newVar] = false;

    return newVar;
}

unsigned BoundManager::getNumberOfVariables() const
{
    return _size;
}

bool BoundManager::tightenLowerBound( unsigned variable, double value )
{
    bool tightened = setLowerBound( variable, value );
    if ( tightened && _tableau != nullptr )
        _tableau->updateVariableToComplyWithLowerBoundUpdate( variable, value );
    return tightened;
}

bool BoundManager::tightenUpperBound( unsigned variable, double value )
{
    bool tightened = setUpperBound( variable, value );
    if ( tightened && _tableau != nullptr )
        _tableau->updateVariableToComplyWithUpperBoundUpdate( variable, value );
    return tightened;
}

<<<<<<< HEAD
void BoundManager::recordInconsistentBound( unsigned variable, double value, Tightening::BoundType type )
{
  if ( _consistentBounds )
  {
    _consistentBounds = false;
    _firstInconsistentTightening = Tightening( variable, value, type );
  }
=======
void BoundManager::recordInconsistentBound( unsigned variable,
                                            double value,
                                            Tightening::BoundType type )
{
    if ( _consistentBounds )
    {
        _consistentBounds = false;
        _firstInconsistentTightening = Tightening( variable, value, type );
    }
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
}

bool BoundManager::setLowerBound( unsigned variable, double value )
{
    ASSERT( variable < _size );
    if ( value > _lowerBounds[variable] )
    {
        _lowerBounds[variable] = value;
        *_tightenedLower[variable] = true;
        if ( !consistentBounds( variable ) )
            recordInconsistentBound( variable, value, Tightening::LB );
        return true;
    }
    return false;
}


bool BoundManager::setUpperBound( unsigned variable, double value )
{
    ASSERT( variable < _size );
    if ( value < _upperBounds[variable] )
    {
        _upperBounds[variable] = value;
        *_tightenedUpper[variable] = true;
        if ( !consistentBounds( variable ) )
<<<<<<< HEAD
          recordInconsistentBound( variable, value, Tightening::UB );
=======
            recordInconsistentBound( variable, value, Tightening::UB );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
        return true;
    }
    return false;
}

double BoundManager::getLowerBound( unsigned variable ) const
{
    ASSERT( variable < _size );
    return _lowerBounds[variable];
}

double BoundManager::getUpperBound( unsigned variable ) const
{
    ASSERT( variable < _size );
    return _upperBounds[variable];
}

<<<<<<< HEAD
const double * BoundManager::getLowerBounds() const
=======
const double *BoundManager::getLowerBounds() const
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
{
    return _lowerBounds;
}

<<<<<<< HEAD
const double * BoundManager::getUpperBounds() const
=======
const double *BoundManager::getUpperBounds() const
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
{
    return _upperBounds;
}

void BoundManager::storeLocalBounds()
{
    for ( unsigned i = 0; i < _size; ++i )
    {
<<<<<<< HEAD
      *_storedLowerBounds[i]=_lowerBounds[i];
      *_storedUpperBounds[i]=_upperBounds[i];
=======
        *_storedLowerBounds[i] = _lowerBounds[i];
        *_storedUpperBounds[i] = _upperBounds[i];
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    }
}

void BoundManager::restoreLocalBounds()
{
    for ( unsigned i = 0; i < _size; ++i )
    {
<<<<<<< HEAD
      _lowerBounds[i]=*_storedLowerBounds[i];
      _upperBounds[i]=*_storedUpperBounds[i];
=======
        _lowerBounds[i] = *_storedLowerBounds[i];
        _upperBounds[i] = *_storedUpperBounds[i];
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    }
}

void BoundManager::getTightenings( List<Tightening> &tightenings )
{
    for ( unsigned i = 0; i < _size; ++i )
    {
        if ( *_tightenedLower[i] )
        {
            tightenings.append( Tightening( i, _lowerBounds[i], Tightening::LB ) );
            *_tightenedLower[i] = false;
        }

        if ( *_tightenedUpper[i] )
        {
            tightenings.append( Tightening( i, _upperBounds[i], Tightening::UB ) );
            *_tightenedUpper[i] = false;
        }
    }
}

void BoundManager::clearTightenings()
{
    for ( unsigned i = 0; i < _size; ++i )
    {
<<<<<<< HEAD
      *_tightenedLower[i]=false;
      *_tightenedUpper[i]=false;
=======
        *_tightenedLower[i] = false;
        *_tightenedUpper[i] = false;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    }
}

void BoundManager::propagateTightenings()
{
    for ( unsigned i = 0; i < _size; ++i )
    {
<<<<<<< HEAD
      if ( *_tightenedLower[i] )
      {
        _tableau->notifyLowerBound( i, getLowerBound( i ) );
        *_tightenedLower[i] = false;
      }

      if ( *_tightenedUpper[i] )
      {
        _tableau->notifyUpperBound( i, getUpperBound( i ) );
        *_tightenedUpper[i] = false;
      }
=======
        if ( *_tightenedLower[i] )
        {
            _tableau->notifyLowerBound( i, getLowerBound( i ) );
            *_tightenedLower[i] = false;
        }

        if ( *_tightenedUpper[i] )
        {
            _tableau->notifyUpperBound( i, getUpperBound( i ) );
            *_tightenedUpper[i] = false;
        }
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    }
}

bool BoundManager::consistentBounds() const
{
    return _consistentBounds;
}

bool BoundManager::consistentBounds( unsigned variable ) const
{
    ASSERT( variable < _size );
    return FloatUtils::gte( getUpperBound( variable ), getLowerBound( variable ) );
}

void BoundManager::registerTableau( ITableau *ptrTableau )
{
    ASSERT( _tableau == nullptr );
    _tableau = ptrTableau;
}

void BoundManager::registerRowBoundTightener( IRowBoundTightener *ptrRowBoundTightener )
{
    ASSERT( _rowBoundTightener == nullptr );
    _rowBoundTightener = ptrRowBoundTightener;
}

const SparseUnsortedList &BoundManager::getExplanation( unsigned variable, bool isUpper ) const
{
    ASSERT( _engine->shouldProduceProofs() && variable < _size );
    return _boundExplainer->getExplanation( variable, isUpper );
}

bool BoundManager::tightenLowerBound( unsigned variable, double value, const TableauRow &row )
{
    bool tightened = setLowerBound( variable, value );

    if ( tightened )
    {
        if ( _engine->shouldProduceProofs() )
            _boundExplainer->updateBoundExplanation( row, BoundType::LOWER, variable );

        if ( _tableau != nullptr )
            _tableau->updateVariableToComplyWithLowerBoundUpdate( variable, value );
    }
    return tightened;
}

bool BoundManager::tightenUpperBound( unsigned variable, double value, const TableauRow &row )
{
    bool tightened = setUpperBound( variable, value );

    if ( tightened )
    {
        if ( _engine->shouldProduceProofs() )
            _boundExplainer->updateBoundExplanation( row, BoundType::UPPER, variable );

        if ( _tableau != nullptr )
            _tableau->updateVariableToComplyWithUpperBoundUpdate( variable, value );
    }
    return tightened;
}

<<<<<<< HEAD
bool BoundManager::tightenLowerBound( unsigned variable, double value, const SparseUnsortedList &row )
=======
bool BoundManager::tightenLowerBound( unsigned variable,
                                      double value,
                                      const SparseUnsortedList &row )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
{
    bool tightened = setLowerBound( variable, value );

    if ( tightened )
    {
        if ( _engine->shouldProduceProofs() )
            _boundExplainer->updateBoundExplanationSparse( row, BoundType::LOWER, variable );

        if ( _tableau != nullptr )
            _tableau->updateVariableToComplyWithLowerBoundUpdate( variable, value );
    }
    return tightened;
}

<<<<<<< HEAD
bool BoundManager::tightenUpperBound( unsigned variable, double value, const SparseUnsortedList &row )
=======
bool BoundManager::tightenUpperBound( unsigned variable,
                                      double value,
                                      const SparseUnsortedList &row )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
{
    bool tightened = setUpperBound( variable, value );

    if ( tightened )
    {
        if ( _engine->shouldProduceProofs() )
            _boundExplainer->updateBoundExplanationSparse( row, BoundType::UPPER, variable );

        if ( _tableau != nullptr )
            _tableau->updateVariableToComplyWithUpperBoundUpdate( variable, value );
    }

    return tightened;
}

void BoundManager::resetExplanation( const unsigned var, const bool isUpper ) const
{
    _boundExplainer->resetExplanation( var, isUpper );
}

<<<<<<< HEAD
void BoundManager::setExplanation( const SparseUnsortedList &explanation, unsigned var, bool isUpper ) const
=======
void BoundManager::setExplanation( const SparseUnsortedList &explanation,
                                   unsigned var,
                                   bool isUpper ) const
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
{
    _boundExplainer->setExplanation( explanation, var, isUpper );
}

const BoundExplainer *BoundManager::getBoundExplainer() const
{
    return _boundExplainer;
}

void BoundManager::copyBoundExplainerContent( const BoundExplainer *boundsExplainer )
{
    *_boundExplainer = *boundsExplainer;
}

void BoundManager::updateBoundExplanation( const TableauRow &row, bool isUpper, unsigned var )
{
    _boundExplainer->updateBoundExplanation( row, isUpper, var );
}

<<<<<<< HEAD
void BoundManager::updateBoundExplanationSparse( const SparseUnsortedList &row, bool isUpper, unsigned var )
=======
void BoundManager::updateBoundExplanationSparse( const SparseUnsortedList &row,
                                                 bool isUpper,
                                                 unsigned var )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
{
    _boundExplainer->updateBoundExplanationSparse( row, isUpper, var );
}

<<<<<<< HEAD
bool BoundManager::addLemmaExplanationAndTightenBound( unsigned var, double value, BoundType affectedVarBound,
                                                      const List<unsigned> &causingVars, BoundType causingVarBound,
                                                      PiecewiseLinearFunctionType constraintType )
=======
bool BoundManager::addLemmaExplanationAndTightenBound( unsigned var,
                                                       double value,
                                                       BoundType affectedVarBound,
                                                       const List<unsigned> &causingVars,
                                                       BoundType causingVarBound,
                                                       PiecewiseLinearFunctionType constraintType )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
{
    if ( !shouldProduceProofs() )
        return false;

    ASSERT( var < _tableau->getN() );

    // Register new ground bound, update certificate, and reset explanation
    Vector<SparseUnsortedList> allExplanations( 0 );

<<<<<<< HEAD
    bool tightened = affectedVarBound == BoundType::UPPER ? tightenUpperBound( var, value ) : tightenLowerBound( var, value );
=======
    bool tightened = affectedVarBound == BoundType::UPPER ? tightenUpperBound( var, value )
                                                          : tightenLowerBound( var, value );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    if ( tightened )
    {
        if ( constraintType == RELU || constraintType == SIGN )
        {
            ASSERT( causingVars.size() == 1 );
            allExplanations.append( getExplanation( causingVars.front(), causingVarBound ) );
        }
        else if ( constraintType == ABSOLUTE_VALUE )
        {
            if ( causingVars.size() == 1 )
                allExplanations.append( getExplanation( causingVars.front(), causingVarBound ) );
            else
            {
                // Used for two cases:
                // 1. Lemma of the type _f = max(upperBound(b), -lowerBound(b)).
<<<<<<< HEAD
                //    Two explanations are stored so the checker could check that f has the maximal value of the two.
                // 2. Lemmas of the type lowerBound(f) > -lowerBound(b) or upperBound(b).
                //    Again, two explanations are involved in the proof.
                // Add zero vectors to maintain consistency of explanations size
                allExplanations.append( getExplanation( causingVars.front(), causingVarBound == BoundType::UPPER ) );
=======
                //    Two explanations are stored so the checker could check that f has the maximal
                //    value of the two.
                // 2. Lemmas of the type lowerBound(f) > -lowerBound(b) or upperBound(b).
                //    Again, two explanations are involved in the proof.
                // Add zero vectors to maintain consistency of explanations size
                allExplanations.append(
                    getExplanation( causingVars.front(), causingVarBound == BoundType::UPPER ) );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

                allExplanations.append( getExplanation( causingVars.back(), BoundType::LOWER ) );
            }
        }
        else if ( constraintType == MAX )
            for ( const auto &element : causingVars )
                allExplanations.append( getExplanation( element, BoundType::UPPER ) );
        else
            throw MarabouError( MarabouError::FEATURE_NOT_YET_SUPPORTED );

<<<<<<< HEAD
        std::shared_ptr<PLCLemma> PLCExpl = std::make_shared<PLCLemma>( causingVars, var, value, causingVarBound, affectedVarBound, allExplanations, constraintType );
        _engine->getUNSATCertificateCurrentPointer()->addPLCLemma(PLCExpl );
        affectedVarBound == BoundType::UPPER ? _engine->updateGroundUpperBound( var, value ) : _engine->updateGroundLowerBound( var, value );
=======
        std::shared_ptr<PLCLemma> PLCExpl = std::make_shared<PLCLemma>( causingVars,
                                                                        var,
                                                                        value,
                                                                        causingVarBound,
                                                                        affectedVarBound,
                                                                        allExplanations,
                                                                        constraintType );
        _engine->getUNSATCertificateCurrentPointer()->addPLCLemma( PLCExpl );
        affectedVarBound == BoundType::UPPER ? _engine->updateGroundUpperBound( var, value )
                                             : _engine->updateGroundLowerBound( var, value );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
        resetExplanation( var, affectedVarBound );
    }
    return true;
}

<<<<<<< HEAD
void BoundManager::registerEngine( IEngine *engine)
=======
void BoundManager::registerEngine( IEngine *engine )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
{
    _engine = engine;
}

void BoundManager::initializeBoundExplainer( unsigned numberOfVariables, unsigned numberOfRows )
{
    if ( _engine->shouldProduceProofs() )
        _boundExplainer = new BoundExplainer( numberOfVariables, numberOfRows, _context );
}

unsigned BoundManager::getInconsistentVariable() const
{
    if ( _consistentBounds )
        return NO_VARIABLE_FOUND;
    return _firstInconsistentTightening._variable;
}

double BoundManager::computeRowBound( const TableauRow &row, const bool isUpper ) const
{
    double bound = 0;
    double multiplier;
    unsigned var;

    for ( unsigned i = 0; i < row._size; ++i )
    {
        var = row._row[i]._var;
        if ( FloatUtils::isZero( row[i] ) )
            continue;

<<<<<<< HEAD
        multiplier = ( isUpper && FloatUtils::isPositive( row[i] ) ) || ( !isUpper && FloatUtils::isNegative( row[i] ) ) ? _upperBounds[var] : _lowerBounds[var];
=======
        multiplier = ( isUpper && FloatUtils::isPositive( row[i] ) ) ||
                             ( !isUpper && FloatUtils::isNegative( row[i] ) )
                       ? _upperBounds[var]
                       : _lowerBounds[var];
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
        multiplier = FloatUtils::isZero( multiplier ) ? 0 : multiplier * row[i];
        bound += FloatUtils::isZero( multiplier ) ? 0 : multiplier;
    }

    bound += FloatUtils::isZero( row._scalar ) ? 0 : row._scalar;
    return bound;
}

<<<<<<< HEAD
double BoundManager::computeSparseRowBound( const SparseUnsortedList &row, const bool isUpper, const unsigned var ) const
=======
double BoundManager::computeSparseRowBound( const SparseUnsortedList &row,
                                            const bool isUpper,
                                            const unsigned var ) const
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
{
    ASSERT( !row.empty() && var < _size );

    unsigned curVar;
    double ci = 0;
    double bound = 0;
    double realCoefficient;
    double curVal;
    double multiplier;

    for ( const auto &entry : row )
    {
        if ( entry._index == var )
        {
            ci = entry._value;
            break;
        }
    }

    ASSERT( !FloatUtils::isZero( ci ) );

    for ( const auto &entry : row )
    {
        curVar = entry._index;
        curVal = entry._value;

        if ( FloatUtils::isZero( curVal ) || curVar == var )
            continue;

        realCoefficient = curVal / -ci;

        if ( FloatUtils::isZero( realCoefficient ) )
            continue;

<<<<<<< HEAD
        multiplier = ( isUpper && realCoefficient  > 0 ) || ( !isUpper &&  realCoefficient < 0 ) ? _upperBounds[curVar] : _lowerBounds[curVar];
=======
        multiplier = ( isUpper && realCoefficient > 0 ) || ( !isUpper && realCoefficient < 0 )
                       ? _upperBounds[curVar]
                       : _lowerBounds[curVar];
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
        multiplier = FloatUtils::isZero( multiplier ) ? 0 : multiplier * realCoefficient;
        bound += FloatUtils::isZero( multiplier ) ? 0 : multiplier;
    }

    return bound;
}

bool BoundManager::isExplanationTrivial( unsigned var, bool isUpper ) const
{
    return _boundExplainer->isExplanationTrivial( var, isUpper );
}

bool BoundManager::shouldProduceProofs() const
{
    return _boundExplainer != nullptr;
}