/*********************                                                        */
/*! \file OnnxParser.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Matthew Daggitt
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** A parser for ONNX neural network files.
 **/

#ifndef __OnnxParser_h__
#define __OnnxParser_h__

<<<<<<< HEAD
#include "onnx.proto3.pb.h"
#include "Map.h"
#include "List.h"
#include "Vector.h"
#include "Set.h"
#include "MString.h"
#include "InputQuery.h"
#include "NetworkParser.h"
#include "TensorUtils.h"

#define ONNX_LOG(x, ...) LOG(GlobalConfiguration::ONNX_PARSER_LOGGING, "OnnxParser: %s\n", x)
=======
#include "InputQuery.h"
#include "List.h"
#include "MString.h"
#include "Map.h"
#include "NetworkParser.h"
#include "Set.h"
#include "TensorUtils.h"
#include "Vector.h"
#include "onnx.proto3.pb.h"

#define ONNX_LOG( x, ... ) LOG( GlobalConfiguration::ONNX_PARSER_LOGGING, "OnnxParser: %s\n", x )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390


class OnnxParser : public NetworkParser
{
public:
<<<<<<< HEAD
    OnnxParser( const String& path );

    void generateQuery( InputQuery &inputQuery );
    void generatePartialQuery( InputQuery &inputQuery , Set<String> &inputNames, String &outputName );


private:

=======
    OnnxParser( const String &path );

    void generateQuery( InputQuery &inputQuery );
    void generatePartialQuery( InputQuery &inputQuery,
                               Set<String> &inputNames,
                               Set<String> &terminalNames );


private:
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    // Settings //

    onnx::GraphProto _network;
    Set<String> _inputNames;
<<<<<<< HEAD
    String _outputName;
=======

    /*
      The set of terminal nodes for the query. Note that these doesn't have to be outputs of
      the network, they can be intermediate nodes.
    */
    Set<String> _terminalNames;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    // State //

    Map<String, TensorShape> _shapeMap;
    Map<String, Vector<Variable>> _varMap;
<<<<<<< HEAD
    Map<String, const Vector<int>> _constantIntTensors;
=======
    Map<String, const Vector<int64_t>> _constantIntTensors;
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    Map<String, const Vector<double>> _constantFloatTensors;
    Set<String> _processedNodes;
    unsigned _numberOfFoundInputs;

    // Methods //

<<<<<<< HEAD
    void readNetwork( const String& path );
    Set<String> readInputNames();
    String readOutputName();
    void initializeShapeAndConstantMaps();
    void validateUserInputNames( Set<String>& inputNames) ;
    void validateUserOutputNames( String& outputName );
    void validateAllInputsAndOutputsFound();

    void processGraph( Set<String>& inputNames, String& outputName , InputQuery &inputQuery );
    void processNode( String& nodeName, bool makeEquations );
    void makeMarabouEquations ( onnx::NodeProto& node, bool makeEquations );
    Set<String> getInputsToNode( onnx::NodeProto& node );
    List<onnx::NodeProto> getNodesWithOutput( String& nodeName );
    Vector<Variable> makeNodeVariables ( String& nodeName, bool isInput );

    bool isConstantNode( String name );
    void transferValues ( String oldName, String newName );
    void insertConstant( String name, const onnx::TensorProto& tensor, TensorShape shape );

    void constant( onnx::NodeProto& node );
    void identity( onnx::NodeProto& node );
    void cast( onnx::NodeProto& node );
    void reshape( onnx::NodeProto& node );
    void flatten( onnx::NodeProto& node );
    void transpose( onnx::NodeProto& node );
    void batchNormEquations( onnx::NodeProto& node, bool makeEquations );
    void maxPoolEquations( onnx::NodeProto& node, bool makeEquations );
    void convEquations( onnx::NodeProto& node, bool makeEquations );
    void gemmEquations( onnx::NodeProto& node, bool makeEquations );
    void scaleAndAddEquations( onnx::NodeProto& node, bool makeEquations, double coefficient1, double coefficient2 );
    void matMulEquations( onnx::NodeProto& node, bool makeEquations );
    void reluEquations( onnx::NodeProto& node, bool makeEquations );
    void sigmoidEquations( onnx::NodeProto& node, bool makeEquations );
    void tanhEquations( onnx::NodeProto& node, bool makeEquations );
=======
    void readNetwork( const String &path );
    Set<String> readInputNames();
    Set<String> readOutputNames();
    void initializeShapeAndConstantMaps();
    void validateUserInputNames( Set<String> &inputNames );
    void validateUserTerminalNames( Set<String> &terminalNames );
    void validateAllInputsAndOutputsFound();

    void
    processGraph( Set<String> &inputNames, Set<String> &terminalNames, InputQuery &inputQuery );
    void processNode( String &nodeName, bool makeEquations );
    void makeMarabouEquations( onnx::NodeProto &node, bool makeEquations );
    Set<String> getInputsToNode( onnx::NodeProto &node );
    List<onnx::NodeProto> getNodesWithOutput( String &nodeName );
    Vector<Variable> makeNodeVariables( String &nodeName, bool isInput );

    bool isConstantNode( String name );

    void transferValues( String oldName, String newName );
    void insertConstant( String name, const onnx::TensorProto &tensor, TensorShape shape );

    void constant( onnx::NodeProto &node );
    void identity( onnx::NodeProto &node );
    void cast( onnx::NodeProto &node );
    void reshape( onnx::NodeProto &node );
    void squeeze( onnx::NodeProto &node );
    void unsqueeze( onnx::NodeProto &node );
    void flatten( onnx::NodeProto &node );
    void transpose( onnx::NodeProto &node );
    void batchNormEquations( onnx::NodeProto &node, bool makeEquations );
    void maxPoolEquations( onnx::NodeProto &node, bool makeEquations );
    void convEquations( onnx::NodeProto &node, bool makeEquations );
    void gemmEquations( onnx::NodeProto &node, bool makeEquations );
    void scaleAndAddEquations( onnx::NodeProto &node,
                               bool makeEquations,
                               double coefficient1,
                               double coefficient2 );
    void matMulEquations( onnx::NodeProto &node, bool makeEquations );
    void reluEquations( onnx::NodeProto &node, bool makeEquations );
    void leakyReluEquations( onnx::NodeProto &node, bool makeEquations );
    void sigmoidEquations( onnx::NodeProto &node, bool makeEquations );
    void tanhEquations( onnx::NodeProto &node, bool makeEquations );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
};

#endif // __OnnxParser_h__

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
