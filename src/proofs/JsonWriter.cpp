/*********************                                                        */
/*! \file JsonWriter.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Omri Isac, Guy Katz
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2023 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]
 **/

<<<<<<< HEAD
#include <iomanip>
#include "JsonWriter.h"

=======
#include "JsonWriter.h"

#include <iomanip>

>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
const char JsonWriter::AFFECTED_VAR[] = "\"affVar\" : ";
const char JsonWriter::AFFECTED_BOUND[] = "\"affBound\" : ";
const char JsonWriter::BOUND[] = "\"bound\" : ";
const char JsonWriter::CAUSING_VAR[] = "\"causVar\" : ";
const char JsonWriter::CAUSING_VARS[] = "\"causVars\" : ";
const char JsonWriter::CAUSING_BOUND[] = "\"causBound\" : ";
const char JsonWriter::CONTRADICTION[] = "\"contradiction\" : ";
const char JsonWriter::CONSTRAINT[] = "\"constraint\" : ";
const char JsonWriter::CONSTRAINTS[] = "\"constraints\" : ";
const char JsonWriter::CONSTRAINT_TYPE[] = "\"constraintType\" : ";
const char JsonWriter::CHILDREN[] = "\"children\" : ";
const char JsonWriter::EXPLANATION[] = "\"expl\" : ";
const char JsonWriter::EXPLANATIONS[] = "\"expls\" : ";
const char JsonWriter::LEMMAS[] = "\"lemmas\" : ";
const char JsonWriter::LOWER_BOUND[] = "\"L\"";
const char JsonWriter::LOWER_BOUNDS[] = "\"lowerBounds\" : ";
const char JsonWriter::PROOF[] = "\"proof\" : ";
const char JsonWriter::SPLIT[] = "\"split\" : ";
const char JsonWriter::TABLEAU[] = "\"tableau\" : ";
const char JsonWriter::UPPER_BOUND[] = "\"U\"";
const char JsonWriter::UPPER_BOUNDS[] = "\"upperBounds\" : ";
const char JsonWriter::VALUE[] = "\"val\" : ";
const char JsonWriter::VARIABLE[] = "\"var\" : ";
const char JsonWriter::VARIABLES[] = "\"vars\" : ";

const bool JsonWriter::PROVE_LEMMAS = true;
<<<<<<< HEAD
const unsigned JsonWriter::JSONWRITER_PRECISION = ( unsigned ) std::log10( 1 / GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS );
const char JsonWriter::PROOF_FILENAME[] = "proof.json";

void JsonWriter::writeProofToJson(  const UnsatCertificateNode *root,
                                    unsigned explanationSize,
                                    const SparseMatrix *initialTableau,
                                    const Vector<double> &upperBounds,
                                    const Vector<double> &lowerBounds,
                                    const List<PiecewiseLinearConstraint *> &problemConstraints,
                                    IFile &file )
=======
const unsigned JsonWriter::JSONWRITER_PRECISION =
    (unsigned)std::log10( 1 / GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS );
const char JsonWriter::PROOF_FILENAME[] = "proof.json";

void JsonWriter::writeProofToJson( const UnsatCertificateNode *root,
                                   unsigned explanationSize,
                                   const SparseMatrix *initialTableau,
                                   const Vector<double> &upperBounds,
                                   const Vector<double> &lowerBounds,
                                   const List<PiecewiseLinearConstraint *> &problemConstraints,
                                   IFile &file )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
{
    ASSERT( root );
    ASSERT( upperBounds.size() > 0 );
    ASSERT( lowerBounds.size() > 0 );
    ASSERT( explanationSize > 0 );

    List<String> jsonLines;
    jsonLines.append( "{\n" );

    // Add initial query information to the instance
    writeInitialTableau( initialTableau, explanationSize, jsonLines );
    writeBounds( upperBounds, UPPER, jsonLines );
    writeBounds( lowerBounds, LOWER, jsonLines );
    writePiecewiseLinearConstraints( problemConstraints, jsonLines );

    // Add UNSAT certificate proof tree object to the instance
    jsonLines.append( String( PROOF ) + String( "{ \n" ) );
    writeUnsatCertificateNode( root, explanationSize, jsonLines );
    jsonLines.append( "}\n" );

    jsonLines.append( "}\n" );

    // Wrap-up everything and write to a file
    writeInstanceToFile( file, jsonLines );
}

<<<<<<< HEAD
void JsonWriter::writeBounds( const Vector<double> &bounds, BoundType isUpper, List<String> &instance )
=======
void JsonWriter::writeBounds( const Vector<double> &bounds,
                              BoundType isUpper,
                              List<String> &instance )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
{
    String boundsString = isUpper == UPPER ? UPPER_BOUNDS : LOWER_BOUNDS;

    boundsString += convertDoubleArrayToString( bounds.data(), bounds.size() );

    instance.append( boundsString );
    instance.append( String( ",\n" ) );
}

<<<<<<< HEAD
void JsonWriter::writeInitialTableau( const SparseMatrix *initialTableau, unsigned explanationSize, List<String> &instance )
=======
void JsonWriter::writeInitialTableau( const SparseMatrix *initialTableau,
                                      unsigned explanationSize,
                                      List<String> &instance )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
{
    instance.append( String( TABLEAU ) );
    instance.append( "[\n" );

    SparseUnsortedList tableauRow = SparseUnsortedList();

    // Write each tableau row separately
    for ( unsigned i = 0; i < explanationSize; ++i )
    {
        instance.append( "[" );
        initialTableau->getRow( i, &tableauRow );
        instance.append( convertSparseUnsortedListToString( tableauRow ) );

        // Not adding a comma after the last element
        if ( i != explanationSize - 1 )
            instance.append( "],\n" );
        else
            instance.append( "]\n" );
    }

<<<<<<< HEAD
    instance.append ( "], \n" );
}

void JsonWriter::writePiecewiseLinearConstraints( const List<PiecewiseLinearConstraint *> &problemConstraints, List<String> &instance )
=======
    instance.append( "], \n" );
}

void JsonWriter::writePiecewiseLinearConstraints(
    const List<PiecewiseLinearConstraint *> &problemConstraints,
    List<String> &instance )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
{
    instance.append( CONSTRAINTS );
    instance.append( "[\n" );
    String type = "";
    String vars = "";
    unsigned counter = 0;
    unsigned size = problemConstraints.size();
    for ( auto constraint : problemConstraints )
    {
        type = std::to_string( constraint->getType() );

        // Vars are written in the same order as in the get method
        vars = "[";
        for ( unsigned var : constraint->getParticipatingVariables() )
            vars += std::to_string( var ) + ", ";

        List<unsigned> tableauVars = constraint->getTableauAuxVars();

        for ( unsigned var : tableauVars )
        {
<<<<<<< HEAD
            vars += std::to_string( var ) ;
=======
            vars += std::to_string( var );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
            if ( var != tableauVars.back() )
                vars += ", ";
        }

        vars += "]";

<<<<<<< HEAD
        instance.append( String( "{" ) + String( CONSTRAINT_TYPE ) + type + String( ", " ) + String( VARIABLES ) + vars );

        // Not adding a comma after the last element
        if (counter != size - 1 )
=======
        instance.append( String( "{" ) + String( CONSTRAINT_TYPE ) + type + String( ", " ) +
                         String( VARIABLES ) + vars );

        // Not adding a comma after the last element
        if ( counter != size - 1 )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
            instance.append( "},\n" );
        else
            instance.append( "}\n" );
        ++counter;
    }
    instance.append( "],\n" );
}

<<<<<<< HEAD
void JsonWriter::writeUnsatCertificateNode( const UnsatCertificateNode *node, unsigned explanationSize ,List<String> &instance )
=======
void JsonWriter::writeUnsatCertificateNode( const UnsatCertificateNode *node,
                                            unsigned explanationSize,
                                            List<String> &instance )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
{
    // For SAT examples only (used for debugging)
    if ( !node->getVisited() || node->getSATSolutionFlag() )
        return;

    writeHeadSplit( node->getSplit(), instance );

    writePLCLemmas( node->getPLCLemmas(), instance );

    if ( node->getChildren().empty() )
        writeContradiction( node->getContradiction(), instance );
    else
    {
        instance.append( CHILDREN );
        instance.append( "[\n" );

        unsigned counter = 0;
        unsigned size = node->getChildren().size();
        for ( auto child : node->getChildren() )
        {
<<<<<<< HEAD
            instance.append("{\n" );
            writeUnsatCertificateNode( child, explanationSize, instance );

            // Not adding a comma after the last element
            if (counter != size - 1)
                instance.append("},\n" );
            else
                instance.append("}\n" );
=======
            instance.append( "{\n" );
            writeUnsatCertificateNode( child, explanationSize, instance );

            // Not adding a comma after the last element
            if ( counter != size - 1 )
                instance.append( "},\n" );
            else
                instance.append( "}\n" );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
            ++counter;
        }

        instance.append( "]\n" );
    }
}

void JsonWriter::writeHeadSplit( const PiecewiseLinearCaseSplit &headSplit, List<String> &instance )
{
    String boundTypeString;
    unsigned counter = 0;
    unsigned size = headSplit.getBoundTightenings().size();

    if ( !size )
        return;

    instance.append( SPLIT );
    instance.append( "[" );
    for ( auto tightening : headSplit.getBoundTightenings() )
    {
<<<<<<< HEAD
        instance.append( String ( "{" ) + String( VARIABLE ) + std::to_string( tightening._variable ) + String(", " ) + \
        String ( VALUE ) + convertDoubleToString( tightening._value ) + String(", " ) + \
        String ( BOUND ) );
        boundTypeString = tightening._type == Tightening::UB ? String ( UPPER_BOUND ) : String ( LOWER_BOUND );
        boundTypeString += String( "}" );
        // Not adding a comma after the last element
        if ( counter !=  size - 1 )
=======
        instance.append( String( "{" ) + String( VARIABLE ) +
                         std::to_string( tightening._variable ) + String( ", " ) + String( VALUE ) +
                         convertDoubleToString( tightening._value ) + String( ", " ) +
                         String( BOUND ) );
        boundTypeString =
            tightening._type == Tightening::UB ? String( UPPER_BOUND ) : String( LOWER_BOUND );
        boundTypeString += String( "}" );
        // Not adding a comma after the last element
        if ( counter != size - 1 )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
            boundTypeString += ", ";
        instance.append( boundTypeString );
        ++counter;
    }
    instance.append( "],\n" );
}

void JsonWriter::writeContradiction( const Contradiction *contradiction, List<String> &instance )
{
    String contradictionString = CONTRADICTION;
    const SparseUnsortedList explanation = contradiction->getContradiction();
    contradictionString += "[ ";
<<<<<<< HEAD
    contradictionString += explanation.empty() ? std::to_string( contradiction->getVar() ) : convertSparseUnsortedListToString( explanation );
=======
    contradictionString += explanation.empty() ? std::to_string( contradiction->getVar() )
                                               : convertSparseUnsortedListToString( explanation );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    contradictionString += String( " ]\n" );
    instance.append( contradictionString );
}

<<<<<<< HEAD
void JsonWriter::writePLCLemmas( const List<std::shared_ptr<PLCLemma>> &PLCExplanations, List<String> &instance )
=======
void JsonWriter::writePLCLemmas( const List<std::shared_ptr<PLCLemma>> &PLCExplanations,
                                 List<String> &instance )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
{
    unsigned counter = 0;
    unsigned size = PLCExplanations.size();
    if ( !size )
        return;
    String affectedBoundType = "";
    String causingBoundType = "";

    instance.append( LEMMAS );
    instance.append( "[\n" );

    // Write all fields of each lemma
    for ( auto lemma : PLCExplanations )
    {
        instance.append( String( "{" ) );
<<<<<<< HEAD
        instance.append( String( AFFECTED_VAR) + std::to_string( lemma->getAffectedVar() ) + String( ", " ) );
        affectedBoundType = lemma->getAffectedVarBound() == UPPER ? String ( UPPER_BOUND ) : String ( LOWER_BOUND );
        instance.append( String( AFFECTED_BOUND ) + affectedBoundType  + String( ", " ) );
=======
        instance.append( String( AFFECTED_VAR ) + std::to_string( lemma->getAffectedVar() ) +
                         String( ", " ) );
        affectedBoundType =
            lemma->getAffectedVarBound() == UPPER ? String( UPPER_BOUND ) : String( LOWER_BOUND );
        instance.append( String( AFFECTED_BOUND ) + affectedBoundType + String( ", " ) );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
        instance.append( String( BOUND ) + convertDoubleToString( lemma->getBound() ) );

        if ( PROVE_LEMMAS )
        {
            instance.append( String( ", " ) );
            List<unsigned> causingVars = lemma->getCausingVars();

            if ( causingVars.size() == 1 )
<<<<<<< HEAD
                instance.append( String( CAUSING_VAR ) + std::to_string( causingVars.front() ) + String( ", " ) );
            else
            {
                instance.append( String( CAUSING_VARS ) + "[ ");
=======
                instance.append( String( CAUSING_VAR ) + std::to_string( causingVars.front() ) +
                                 String( ", " ) );
            else
            {
                instance.append( String( CAUSING_VARS ) + "[ " );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
                for ( unsigned var : causingVars )
                {
                    instance.append( std::to_string( var ) );
                    if ( var != causingVars.back() )
                        instance.append( ", " );
                }
<<<<<<< HEAD
                instance.append( " ]\n");
            }

            causingBoundType = lemma->getCausingVarBound() == UPPER ? String( UPPER_BOUND ) : String( LOWER_BOUND );
            instance.append( String( CAUSING_BOUND ) + causingBoundType + String( ", " ) );
            instance.append( String( CONSTRAINT ) + std::to_string( lemma->getConstraintType() ) + String( ",\n" ) );

            List<SparseUnsortedList> expls = lemma->getExplanations();
            if ( expls.size() == 1 )
                instance.append( String( EXPLANATION ) + "[ " + convertSparseUnsortedListToString( expls.front() ) + " ]");
            else
            {
                instance.append( String( EXPLANATIONS ) + "[ ");
                unsigned innerCounter = 0;
                for ( SparseUnsortedList &expl : expls )
                {
                    instance.append( "[ ");
                    instance.append( convertSparseUnsortedListToString( expl ) );
                    instance.append( " ]");
=======
                instance.append( " ]\n" );
            }

            causingBoundType = lemma->getCausingVarBound() == UPPER ? String( UPPER_BOUND )
                                                                    : String( LOWER_BOUND );
            instance.append( String( CAUSING_BOUND ) + causingBoundType + String( ", " ) );
            instance.append( String( CONSTRAINT ) + std::to_string( lemma->getConstraintType() ) +
                             String( ",\n" ) );

            List<SparseUnsortedList> expls = lemma->getExplanations();
            if ( expls.size() == 1 )
                instance.append( String( EXPLANATION ) + "[ " +
                                 convertSparseUnsortedListToString( expls.front() ) + " ]" );
            else
            {
                instance.append( String( EXPLANATIONS ) + "[ " );
                unsigned innerCounter = 0;
                for ( SparseUnsortedList &expl : expls )
                {
                    instance.append( "[ " );
                    instance.append( convertSparseUnsortedListToString( expl ) );
                    instance.append( " ]" );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
                    if ( innerCounter != expls.size() - 1 )
                        instance.append( ",\n" );

                    ++innerCounter;
                }
                instance.append( " ]\n" );
            }
        }

        instance.append( String( "}" ) );

        // Not adding a comma after the last element
        if ( counter != size - 1 )
            instance.append( ",\n" );
        ++counter;
    }

    instance.append( "\n],\n" );
}

void JsonWriter::writeInstanceToFile( IFile &file, const List<String> &instance )
{
    file.open( File::MODE_WRITE_TRUNCATE );

    for ( const String &s : instance )
        file.write( s );

    file.close();
}

String JsonWriter::convertDoubleToString( double value )
{
    std::stringstream s;
    s << std::fixed << std::setprecision( JSONWRITER_PRECISION ) << value;
<<<<<<< HEAD
    String str = String ( s.str() ).trimZerosFromRight();
=======
    String str = String( s.str() ).trimZerosFromRight();
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    // Add .0 for integers for some JSON parsers.
    if ( !str.contains( "." ) )
        str += String( ".0" );
    return str;
}

String JsonWriter::convertDoubleArrayToString( const double *arr, unsigned size )
{
    String arrString = "[";
    for ( unsigned i = 0; i < size - 1; ++i )
        arrString += convertDoubleToString( arr[i] ) + ", ";

<<<<<<< HEAD
    arrString += convertDoubleToString( arr[size -1] ) + "]";
=======
    arrString += convertDoubleToString( arr[size - 1] ) + "]";
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    return arrString;
}

String JsonWriter::convertSparseUnsortedListToString( SparseUnsortedList sparseList )
{
    String sparseListString = "";
    unsigned counter = 0;
    unsigned size = sparseList.getNnz();
<<<<<<< HEAD
    for ( auto entry = sparseList.begin() ; entry != sparseList.end(); ++entry )
    {
        sparseListString +=  String( "{" ) + String( VARIABLE ) + std::to_string( entry->_index ) + String(", " ) +
                         String( VALUE ) + convertDoubleToString( entry->_value ) + String ( "}" );

        // Not adding a comma after the last element
        if ( counter != size - 1 )
            sparseListString += ", " ;
=======
    for ( auto entry = sparseList.begin(); entry != sparseList.end(); ++entry )
    {
        sparseListString += String( "{" ) + String( VARIABLE ) + std::to_string( entry->_index ) +
                            String( ", " ) + String( VALUE ) +
                            convertDoubleToString( entry->_value ) + String( "}" );

        // Not adding a comma after the last element
        if ( counter != size - 1 )
            sparseListString += ", ";
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
        ++counter;
    }
    return sparseListString;
}