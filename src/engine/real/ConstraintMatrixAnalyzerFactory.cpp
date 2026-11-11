/*********************                                                        */
/*! \file ConstraintMatrixAnalyzerFactory.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Shantanu Thakoor
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

**/

#include "ConstraintMatrixAnalyzer.h"

<<<<<<< HEAD
namespace T
{
	IConstraintMatrixAnalyzer *createConstraintMatrixAnalyzer()
	{
		return new ConstraintMatrixAnalyzer();
	}

	void discardConstraintMatrixAnalyzer( IConstraintMatrixAnalyzer *constraintMatrixAnalyzer )
	{
		delete constraintMatrixAnalyzer;
	}
}
=======
namespace T {
IConstraintMatrixAnalyzer *createConstraintMatrixAnalyzer()
{
    return new ConstraintMatrixAnalyzer();
}

void discardConstraintMatrixAnalyzer( IConstraintMatrixAnalyzer *constraintMatrixAnalyzer )
{
    delete constraintMatrixAnalyzer;
}
} // namespace T
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

//
// Local Variables:
// compile-command: "make -C ../../.. "
// tags-file-name: "../../../TAGS"
// c-basic-offset: 4
// End:
//
