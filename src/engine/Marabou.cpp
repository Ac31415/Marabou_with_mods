/*********************                                                        */
/*! \file Marabou.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** \brief [[ Add one-line brief description here ]]
 **
 ** [[ Add lengthier description here ]]
 **/

<<<<<<< HEAD
#include "AcasParser.h"
#include "AutoFile.h"
#include "GlobalConfiguration.h"
#include "File.h"
#include "MStringf.h"
#include "Marabou.h"
#include "OnnxParser.h"
#include "Options.h"
#include "PropertyParser.h"
#include "MarabouError.h"
#include "QueryLoader.h"
=======
#include "Marabou.h"

#include "AcasParser.h"
#include "AutoFile.h"
#include "File.h"
#include "GlobalConfiguration.h"
#include "MStringf.h"
#include "MarabouError.h"
#include "OnnxParser.h"
#include "Options.h"
#include "PropertyParser.h"
#include "QueryLoader.h"
#include "VnnLibParser.h"
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

#ifdef _WIN32
#undef ERROR
#endif

Marabou::Marabou()
    : _acasParser( NULL )
    , _onnxParser( NULL )
    , _engine()
{
}

Marabou::~Marabou()
{
    if ( _acasParser )
    {
        delete _acasParser;
        _acasParser = NULL;
    }

    if ( _onnxParser )
    {
        delete _onnxParser;
        _onnxParser = NULL;
    }
}

void Marabou::run()
{
    struct timespec start = TimeUtils::sampleMicro();

    prepareInputQuery();
    solveQuery();

    struct timespec end = TimeUtils::sampleMicro();

    unsigned long long totalElapsed = TimeUtils::timePassed( start, end );
    displayResults( totalElapsed );

<<<<<<< HEAD
    if( Options::get()->getBool( Options::EXPORT_ASSIGNMENT ) )
=======
    if ( Options::get()->getBool( Options::EXPORT_ASSIGNMENT ) )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
        exportAssignment();
}

void Marabou::prepareInputQuery()
{
    String inputQueryFilePath = Options::get()->getString( Options::INPUT_QUERY_FILE_PATH );
    if ( inputQueryFilePath.length() > 0 )
    {
        /*
          Step 1: extract the query
        */
        if ( !File::exists( inputQueryFilePath ) )
        {
<<<<<<< HEAD
            printf( "Error: the specified inputQuery file (%s) doesn't exist!\n", inputQueryFilePath.ascii() );
=======
            printf( "Error: the specified inputQuery file (%s) doesn't exist!\n",
                    inputQueryFilePath.ascii() );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
            throw MarabouError( MarabouError::FILE_DOESNT_EXIST, inputQueryFilePath.ascii() );
        }

        printf( "InputQuery: %s\n", inputQueryFilePath.ascii() );
        _inputQuery = QueryLoader::loadQuery( inputQueryFilePath );
<<<<<<< HEAD
        _inputQuery.constructNetworkLevelReasoner();
=======
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    }
    else
    {
        /*
          Step 1: extract the network
        */
        String networkFilePath = Options::get()->getString( Options::INPUT_FILE_PATH );

        if ( networkFilePath.length() == 0 )
        {
            printf( "Error: no network file provided!\n" );
            throw MarabouError( MarabouError::FILE_DOESNT_EXIST, networkFilePath.ascii() );
        }

        if ( !File::exists( networkFilePath ) )
        {
<<<<<<< HEAD
            printf( "Error: the specified network file (%s) doesn't exist!\n", networkFilePath.ascii() );
=======
            printf( "Error: the specified network file (%s) doesn't exist!\n",
                    networkFilePath.ascii() );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
            throw MarabouError( MarabouError::FILE_DOESNT_EXIST, networkFilePath.ascii() );
        }
        printf( "Network: %s\n", networkFilePath.ascii() );

<<<<<<< HEAD
        if ( ((String) networkFilePath).endsWith( ".onnx" ) )
=======
        if ( ( (String)networkFilePath ).endsWith( ".onnx" ) )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
        {
            _onnxParser = new OnnxParser( networkFilePath );
            _onnxParser->generateQuery( _inputQuery );
        }
        else
        {
            _acasParser = new AcasParser( networkFilePath );
            _acasParser->generateQuery( _inputQuery );
        }

<<<<<<< HEAD
        _inputQuery.constructNetworkLevelReasoner();

=======
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
        /*
          Step 2: extract the property in question
        */
        String propertyFilePath = Options::get()->getString( Options::PROPERTY_FILE_PATH );
        if ( propertyFilePath != "" )
        {
            printf( "Property: %s\n", propertyFilePath.ascii() );
<<<<<<< HEAD
            PropertyParser().parse( propertyFilePath, _inputQuery );
=======
            if ( propertyFilePath.endsWith( ".vnnlib" ) )
            {
                VnnLibParser().parse( propertyFilePath, _inputQuery );
            }
            else
            {
                PropertyParser().parse( propertyFilePath, _inputQuery );
            }
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
        }
        else
            printf( "Property: None\n" );

        printf( "\n" );
    }

    if ( Options::get()->getBool( Options::DEBUG_ASSIGNMENT ) )
        importDebuggingSolution();

    String queryDumpFilePath = Options::get()->getString( Options::QUERY_DUMP_FILE );
    if ( queryDumpFilePath.length() > 0 )
    {
        _inputQuery.saveQuery( queryDumpFilePath );
        printf( "\nInput query successfully dumped to file\n" );
        exit( 0 );
    }
}

void Marabou::importDebuggingSolution()
{
<<<<<<< HEAD
    String fileName=  Options::get()->getString( Options::IMPORT_ASSIGNMENT_FILE_PATH );
=======
    String fileName = Options::get()->getString( Options::IMPORT_ASSIGNMENT_FILE_PATH );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    AutoFile input( fileName );

    if ( !IFile::exists( fileName ) )
    {
<<<<<<< HEAD
        throw MarabouError( MarabouError::FILE_DOES_NOT_EXIST, Stringf( "File %s not found.\n", fileName.ascii() ).ascii() );
=======
        throw MarabouError( MarabouError::FILE_DOES_NOT_EXIST,
                            Stringf( "File %s not found.\n", fileName.ascii() ).ascii() );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    }

    input->open( IFile::MODE_READ );

    unsigned numVars = atoi( input->readLine().trim().ascii() );
    ASSERT( numVars == _inputQuery.getNumberOfVariables() );

    unsigned var;
    double value;
    String line;

    // Import each assignment
<<<<<<< HEAD
    for ( unsigned i = 0;  i < numVars; ++i )
=======
    for ( unsigned i = 0; i < numVars; ++i )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    {
        line = input->readLine();
        List<String> tokens = line.tokenize( "," );
        auto it = tokens.begin();
        var = atoi( it->ascii() );
        ASSERT( var == i );
        it++;
        value = atof( it->ascii() );
        it++;
        ASSERT( it == tokens.end() );
<<<<<<< HEAD
        _inputQuery.storeDebuggingSolution( var, value);
=======
        _inputQuery.storeDebuggingSolution( var, value );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    }

    input->close();
}

void Marabou::exportAssignment() const
{
    String assignmentFileName = "assignment.txt";
    AutoFile exportFile( assignmentFileName );
    exportFile->open( IFile::MODE_WRITE_TRUNCATE );

    unsigned numberOfVariables = _inputQuery.getNumberOfVariables();
    // Number of Variables
    exportFile->write( Stringf( "%u\n", numberOfVariables ) );

    // Export each assignment
<<<<<<< HEAD
    for ( unsigned var = 0;  var < numberOfVariables; ++var )
=======
    for ( unsigned var = 0; var < numberOfVariables; ++var )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
        exportFile->write( Stringf( "%u, %f\n", var, _inputQuery.getSolutionValue( var ) ) );

    exportFile->close();
}

void Marabou::solveQuery()
{
    if ( _engine.processInputQuery( _inputQuery ) )
    {
        _engine.solve( Options::get()->getInt( Options::TIMEOUT ) );
        if ( _engine.shouldProduceProofs() && _engine.getExitCode() == Engine::UNSAT )
            _engine.certifyUNSATCertificate();
    }

    if ( _engine.getExitCode() == Engine::SAT )
        _engine.extractSolution( _inputQuery );
}

void Marabou::displayResults( unsigned long long microSecondsElapsed ) const
{
    Engine::ExitCode result = _engine.getExitCode();
    String resultString;

    if ( result == Engine::UNSAT )
    {
        resultString = "unsat";
        printf( "unsat\n" );
    }
    else if ( result == Engine::SAT )
    {
        resultString = "sat";
        printf( "sat\n" );

        printf( "Input assignment:\n" );
        for ( unsigned i = 0; i < _inputQuery.getNumInputVariables(); ++i )
<<<<<<< HEAD
            printf( "\tx%u = %lf\n", i, _inputQuery.getSolutionValue( _inputQuery.inputVariableByIndex( i ) ) );
=======
            printf( "\tx%u = %lf\n",
                    i,
                    _inputQuery.getSolutionValue( _inputQuery.inputVariableByIndex( i ) ) );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

        if ( _inputQuery._networkLevelReasoner )
        {
            double *input = new double[_inputQuery.getNumInputVariables()];
            for ( unsigned i = 0; i < _inputQuery.getNumInputVariables(); ++i )
                input[i] = _inputQuery.getSolutionValue( _inputQuery.inputVariableByIndex( i ) );

            NLR::NetworkLevelReasoner *nlr = _inputQuery._networkLevelReasoner;
            NLR::Layer *lastLayer = nlr->getLayer( nlr->getNumberOfLayers() - 1 );
            double *output = new double[lastLayer->getSize()];

            nlr->evaluate( input, output );

            printf( "\n" );
            printf( "Output:\n" );
            for ( unsigned i = 0; i < lastLayer->getSize(); ++i )
                printf( "\ty%u = %lf\n", i, output[i] );
            printf( "\n" );
            delete[] input;
            delete[] output;
        }
        else
        {
            printf( "\n" );
            printf( "Output:\n" );
            for ( unsigned i = 0; i < _inputQuery.getNumOutputVariables(); ++i )
<<<<<<< HEAD
                printf( "\ty%u = %lf\n", i, _inputQuery.getSolutionValue( _inputQuery.outputVariableByIndex( i ) ) );
=======
                printf( "\ty%u = %lf\n",
                        i,
                        _inputQuery.getSolutionValue( _inputQuery.outputVariableByIndex( i ) ) );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
            printf( "\n" );
        }
    }
    else if ( result == Engine::TIMEOUT )
    {
        resultString = "TIMEOUT";
        printf( "Timeout\n" );
    }
    else if ( result == Engine::ERROR )
    {
        resultString = "ERROR";
        printf( "Error\n" );
    }
    else
    {
        resultString = "UNKNOWN";
        printf( "UNKNOWN EXIT CODE! (this should not happen)" );
    }

    // Create a summary file, if requested
    String summaryFilePath = Options::get()->getString( Options::SUMMARY_FILE );
    if ( summaryFilePath != "" )
    {
        File summaryFile( summaryFilePath );
        summaryFile.open( File::MODE_WRITE_TRUNCATE );

        // Field #1: result
        summaryFile.write( resultString );

        // Field #2: total elapsed time
        summaryFile.write( Stringf( " %u ", microSecondsElapsed / 1000000 ) ); // In seconds

        // Field #3: number of visited tree states
        summaryFile.write( Stringf( "%u ",
<<<<<<< HEAD
                                    _engine.getStatistics()->
                                    getUnsignedAttribute
                                    ( Statistics::NUM_VISITED_TREE_STATES ) ) );

        // Field #4: average pivot time in micro seconds
        summaryFile.write( Stringf( "%u",
                                    _engine.getStatistics()->getAveragePivotTimeInMicro() ) );
=======
                                    _engine.getStatistics()->getUnsignedAttribute(
                                        Statistics::NUM_VISITED_TREE_STATES ) ) );

        // Field #4: average pivot time in micro seconds
        summaryFile.write( Stringf( "%u", _engine.getStatistics()->getAveragePivotTimeInMicro() ) );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

        summaryFile.write( "\n" );
    }
}

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
