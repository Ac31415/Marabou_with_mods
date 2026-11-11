/*********************                                                        */
/*! \file FileFactory.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Christopher Lazarus
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

#include "File.h"

<<<<<<< HEAD
namespace T
{
	IFile *createFile( const String &path )
	{
		return new File( path );
	}

	void discardFile( IFile *file )
	{
		delete file;
	}
}
=======
namespace T {
IFile *createFile( const String &path )
{
    return new File( path );
}

void discardFile( IFile *file )
{
    delete file;
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
