/*********************                                                        */
/*! \file FileFactory.h
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

#ifndef __T__FileFactory_h__
#define __T__FileFactory_h__

#include "cxxtest/Mock.h"

class IFile;
class String;

<<<<<<< HEAD
namespace T
{
	IFile *createFile( const String &path );
	void discardFile( IFile *file );
}

CXXTEST_SUPPLY( createFile,        /* => T::Base_AllocateIrp */
				IFile *,           /* Return type            */
				createFile,        /* Name of mock member    */
				( const String &path ),  /* Prototype              */
				T::createFile,        /* Name of real function  */
				( path )       /* Parameter list         */ );

CXXTEST_SUPPLY_VOID( discardFile,        /* => T::Base_AllocateIrp */
					 discardFile,        /* Name of mock member    */
					 ( IFile *file ),  /* Prototype              */
					 T::discardFile,        /* Name of real function  */
					 ( file )            /* Parameter list         */ );
=======
namespace T {
IFile *createFile( const String &path );
void discardFile( IFile *file );
} // namespace T

CXXTEST_SUPPLY( createFile,             /* => T::Base_AllocateIrp */
                IFile *,                /* Return type            */
                createFile,             /* Name of mock member    */
                ( const String &path ), /* Prototype              */
                T::createFile,          /* Name of real function  */
                ( path ) /* Parameter list         */ );

CXXTEST_SUPPLY_VOID( discardFile,     /* => T::Base_AllocateIrp */
                     discardFile,     /* Name of mock member    */
                     ( IFile *file ), /* Prototype              */
                     T::discardFile,  /* Name of real function  */
                     ( file ) /* Parameter list         */ );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

#endif // __T__FileFactory_h__

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
