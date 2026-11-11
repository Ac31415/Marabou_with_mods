/*********************                                                        */
/*! \file LUFactorization.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz, Derek Huang
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

 **/

#include "LUFactorization.h"

#include "BasisFactorizationError.h"
#include "Debug.h"
#include "EtaMatrix.h"
#include "FloatUtils.h"
#include "GlobalConfiguration.h"
#include "LPElement.h"
<<<<<<< HEAD
#include "LUFactorization.h"
=======
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
#include "MalformedBasisException.h"

LUFactorization::LUFactorization( unsigned m, const BasisColumnOracle &basisColumnOracle )
    : IBasisFactorization( basisColumnOracle )
    , _B( NULL )
<<<<<<< HEAD
	, _m( m )
=======
    , _m( m )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    , _luFactors( m )
    , _gaussianEliminator( m )
    , _z( NULL )
{
<<<<<<< HEAD
    _B = new double[m*m];
    if ( !_B )
        throw BasisFactorizationError( BasisFactorizationError::ALLOCATION_FAILED, "LUFactorization::B" );

    _z = new double[m];
    if ( !_z )
        throw BasisFactorizationError( BasisFactorizationError::ALLOCATION_FAILED, "LUFactorization::z" );
=======
    _B = new double[m * m];
    if ( !_B )
        throw BasisFactorizationError( BasisFactorizationError::ALLOCATION_FAILED,
                                       "LUFactorization::B" );

    _z = new double[m];
    if ( !_z )
        throw BasisFactorizationError( BasisFactorizationError::ALLOCATION_FAILED,
                                       "LUFactorization::z" );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
}

LUFactorization::~LUFactorization()
{
    freeIfNeeded();
}

void LUFactorization::freeIfNeeded()
{
<<<<<<< HEAD
	if ( _B )
	{
		delete[] _B;
		_B = NULL;
	}
=======
    if ( _B )
    {
        delete[] _B;
        _B = NULL;
    }
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    List<EtaMatrix *>::iterator it;
    for ( it = _etas.begin(); it != _etas.end(); ++it )
        delete *it;

    if ( _z )
    {
        delete[] _z;
        _z = NULL;
    }
    _etas.clear();
}

const double *LUFactorization::getBasis() const
{
<<<<<<< HEAD
	return _B;
=======
    return _B;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
}

const SparseMatrix *LUFactorization::getSparseBasis() const
{
    printf( "Error! sparse getBasis() not supported for LUFactorization!\n" );
    exit( 1 );
}

const List<EtaMatrix *> LUFactorization::getEtas() const
{
    return _etas;
}

void LUFactorization::updateToAdjacentBasis( unsigned columnIndex,
                                             const double *changeColumn,
<<<<<<< HEAD
                                             const double */* newColumn */ )
=======
                                             const double * /* newColumn */ )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
{
    EtaMatrix *matrix = new EtaMatrix( _m, columnIndex, changeColumn );
    _etas.append( matrix );

<<<<<<< HEAD
	if ( _etas.size() > GlobalConfiguration::REFACTORIZATION_THRESHOLD )
	{
        LU_FACTORIZATION_LOG( "Number of etas exceeds threshold. Refactoring basis\n" );
        obtainFreshBasis();
	}
=======
    if ( _etas.size() > GlobalConfiguration::REFACTORIZATION_THRESHOLD )
    {
        LU_FACTORIZATION_LOG( "Number of etas exceeds threshold. Refactoring basis\n" );
        obtainFreshBasis();
    }
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
}

void LUFactorization::forwardTransformation( const double *y, double *x ) const
{
    /*
      We are solving Bx = y, where B = B0 * E1 ... * En.
      First we solve B0 * z = y using a forward transformation.
    */
    _luFactors.forwardTransformation( y, x );

    /*
      Now we are left with E1 * ... * En * x = z (z is stored in x)
      Eliminate etas one by one.
    */
    for ( const auto &eta : _etas )
    {
        double inverseDiagonal = 1 / eta->_column[eta->_columnIndex];
        double factor = x[eta->_columnIndex] * inverseDiagonal;

        // Solve all non-diagonal rows
        for ( unsigned i = 0; i < _m; ++i )
        {
            if ( i == eta->_columnIndex )
                continue;

            x[i] -= ( factor * eta->_column[i] );
            if ( FloatUtils::isZero( x[i] ) )
                x[i] = 0.0;
        }

        // Handle the digonal element
        x[eta->_columnIndex] *= inverseDiagonal;
        if ( FloatUtils::isZero( x[eta->_columnIndex] ) )
            x[eta->_columnIndex] = 0.0;
    }
}

void LUFactorization::backwardTransformation( const double *y, double *x ) const
{
    /*
      We are solving xB = y, where B = B0 * E1 ... * En.
      The first step is to eliminate the eta matrices.
    */
<<<<<<< HEAD
    memcpy( _z, y, sizeof(double) * _m );
    for ( auto eta = _etas.rbegin(); eta != _etas.rend(); ++eta )
    {
        // The only entry in y that changes is columnIndex
        unsigned columnIndex = (*eta)->_columnIndex;
        for ( unsigned i = 0; i < _m; ++i )
        {
            if ( i != columnIndex )
                _z[columnIndex] -= (_z[i] * (*eta)->_column[i]);
        }

        _z[columnIndex] = _z[columnIndex] / (*eta)->_column[columnIndex];
=======
    memcpy( _z, y, sizeof( double ) * _m );
    for ( auto eta = _etas.rbegin(); eta != _etas.rend(); ++eta )
    {
        // The only entry in y that changes is columnIndex
        unsigned columnIndex = ( *eta )->_columnIndex;
        for ( unsigned i = 0; i < _m; ++i )
        {
            if ( i != columnIndex )
                _z[columnIndex] -= ( _z[i] * ( *eta )->_column[i] );
        }

        _z[columnIndex] = _z[columnIndex] / ( *eta )->_column[columnIndex];
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

        if ( FloatUtils::isZero( _z[columnIndex] ) )
            _z[columnIndex] = 0.0;
    }

    /*
      We now need to solve xB0 = z. Use a backward transformation.
    */
    _luFactors.backwardTransformation( _z, x );
}

void LUFactorization::clearFactorization()
{
<<<<<<< HEAD
	List<EtaMatrix *>::iterator it;
=======
    List<EtaMatrix *>::iterator it;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    for ( it = _etas.begin(); it != _etas.end(); ++it )
        delete *it;
    _etas.clear();
}

void LUFactorization::factorizeBasis()
{
    clearFactorization();

    try
    {
        _gaussianEliminator.run( _B, &_luFactors );
    }
    catch ( const BasisFactorizationError &e )
    {
        if ( e.getCode() == BasisFactorizationError::GAUSSIAN_ELIMINATION_FAILED )
            throw MalformedBasisException();
        else
            throw e;
    }
}

void LUFactorization::storeFactorization( IBasisFactorization *other )
{
    LUFactorization *otherLUFactorization = (LUFactorization *)other;

    ASSERT( _m == otherLUFactorization->_m );
    ASSERT( otherLUFactorization->_etas.size() == 0 );

    obtainFreshBasis();

    // Store the new basis and factorization
<<<<<<< HEAD
    memcpy( otherLUFactorization->_B, _B, sizeof(double) * _m * _m );
=======
    memcpy( otherLUFactorization->_B, _B, sizeof( double ) * _m * _m );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    _luFactors.storeToOther( &otherLUFactorization->_luFactors );
}

void LUFactorization::restoreFactorization( const IBasisFactorization *other )
{
    const LUFactorization *otherLUFactorization = (const LUFactorization *)other;

    ASSERT( _m == otherLUFactorization->_m );
    ASSERT( otherLUFactorization->_etas.size() == 0 );

    // Clear any existing data
    clearFactorization();

    // Store the new basis and factorization
<<<<<<< HEAD
    memcpy( _B, otherLUFactorization->_B, sizeof(double) * _m * _m );
=======
    memcpy( _B, otherLUFactorization->_B, sizeof( double ) * _m * _m );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    otherLUFactorization->_luFactors.storeToOther( &_luFactors );
}

void LUFactorization::invertBasis( double *result )
{
    if ( !_etas.empty() )
        throw BasisFactorizationError( BasisFactorizationError::CANT_INVERT_BASIS_BECAUSE_OF_ETAS );

    ASSERT( result );

    _luFactors.invertBasis( result );
}

bool LUFactorization::explicitBasisAvailable() const
{
    return _etas.empty();
}

void LUFactorization::makeExplicitBasisAvailable()
{
    obtainFreshBasis();
}

void LUFactorization::dump() const
{
    printf( "*** Dumping LU factorization ***\n\n" );

    printf( "\nDumping LU factors:\n" );
    _luFactors.dump();
    printf( "\n\n" );

    printf( "Dumping etas:\n" );
    for ( const auto &eta : _etas )
    {
        eta->dump();
        printf( "\n" );
    }
    printf( "*** Done dumping LU factorization ***\n\n" );
}

void LUFactorization::obtainFreshBasis()
{
    for ( unsigned column = 0; column < _m; ++column )
    {
        _basisColumnOracle->getColumnOfBasis( column, _z );
        for ( unsigned row = 0; row < _m; ++row )
            _B[row * _m + column] = _z[row];
    }

    factorizeBasis();
}

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
