/*********************                                                        */
/*! \file SmtLibWriter.h
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

#ifndef __SmtLibWriter_h__
#define __SmtLibWriter_h__

#include "File.h"
#include "List.h"
#include "MString.h"
#include "PiecewiseLinearConstraint.h"
#include "SparseUnsortedList.h"
#include "Vector.h"

/*
 * A class responsible for writing instances of LP+PLC into SMTLIB format
 */
class SmtLibWriter
{
public:
    /*
      Precision used for writing floating point numbers
     */
    static const unsigned SMTLIBWRITER_PRECISION;

    /*
      Adds a SMTLIB header to the SMTLIB instance with numberOfVariables variables
    */
    static void addHeader( unsigned numberOfVariables, List<String> &instance );

    /*
      Adds a SMTLIB footer to the SMTLIB instance
    */
    static void addFooter( List<String> &instance );

    /*
      Adds a line representing a ReLU constraint, in SMTLIB format, to the SMTLIB instance
    */
    static void
    addReLUConstraint( unsigned b, unsigned f, const PhaseStatus status, List<String> &instance );

    /*
      Adds a line representing a sign constraint, in SMTLIB format, to the SMTLIB instance
    */
    static void
    addSignConstraint( unsigned b, unsigned f, const PhaseStatus status, List<String> &instance );

    /*
     Adds a line representing an absolute value constraint, in SMTLIB format, to the SMTLIB instance
   */
    static void
    addAbsConstraint( unsigned b, unsigned f, const PhaseStatus status, List<String> &instance );

    /*
     Adds a line representing a max constraint, in SMTLIB format, to the SMTLIB instance
    */
    static void addMaxConstraint( unsigned f,
                                  const Set<unsigned> &elements,
                                  const PhaseStatus status,
                                  double maxVal,
                                  List<String> &instance );

    /*
     Adds a line representing a disjunction constraint, in SMTLIB format, to the SMTLIB instance
    */
    static void addDisjunctionConstraint( const List<PiecewiseLinearCaseSplit> &disjuncts,
                                          List<String> &instance );

    /*
      Adds a line representing a Tableau Row, in SMTLIB format, to the SMTLIB instance
    */
    static void addTableauRow( const SparseUnsortedList &row, List<String> &instance );

    /*
     Adds a line representing an equation , in SMTLIB format, to the SMTLIB instance
   */
    static void addEquation( const Equation &eq, List<String> &instance );

    /*
      Adds lines representing the ground upper bounds, in SMTLIB format, to the SMTLIB instance
    */
    static void addGroundUpperBounds( Vector<double> &bounds, List<String> &instance );

    /*
      Adds lines representing the ground lower bounds, in SMTLIB format, to the SMTLIB instance
    */
    static void addGroundLowerBounds( Vector<double> &bounds, List<String> &instance );

    /*
      Adds lines representing a tightening, in SMTLIB format, to the SMTLIB instance
    */
    static void addTightening( Tightening t, List<String> &instance );

    /*
      Writes an instances to a file
     */
    static void writeInstanceToFile( IFile &file, const List<String> &instance );

    /*
      Returns a string representing the value of a double
     */
    static String signedValue( double val );
};

#endif //__SmtLibWriter_h__