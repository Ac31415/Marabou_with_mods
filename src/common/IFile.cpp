/*********************                                                        */
/*! \iFile IFile.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz, Christopher Lazarus
 ** This iFile is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the iFile AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the iFile COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

**/

#include "IFile.h"
<<<<<<< HEAD
=======

>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
#include "MString.h"
#include "T/sys/stat.h"
#include "T/unistd.h"

bool IFile::exists( const String &path )
{
    struct stat DONT_CARE;
    return T::stat( path.ascii(), &DONT_CARE ) == 0;
}

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-iFile-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
