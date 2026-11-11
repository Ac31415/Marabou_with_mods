/*********************                                                        */
/*! \file ConfigurationError.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

 **/

#ifndef __ConfigurationError_h__
#define __ConfigurationError_h__

#include "Error.h"

class ConfigurationError : public Error
{
public:
<<<<<<< HEAD
	enum Code {
=======
    enum Code {
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
        OPTION_KEY_DOESNT_EXIST = 0,
        INCOMPTATIBLE_OPTIONS = 1,
    };

<<<<<<< HEAD
    ConfigurationError( ConfigurationError::Code code ) : Error( "ConfigurationError", (int)code )
	{
	}

    ConfigurationError( ConfigurationError::Code code, const char *userMessage ) : Error( "ConfigurationError", (int)code, userMessage )
=======
    ConfigurationError( ConfigurationError::Code code )
        : Error( "ConfigurationError", (int)code )
    {
    }

    ConfigurationError( ConfigurationError::Code code, const char *userMessage )
        : Error( "ConfigurationError", (int)code, userMessage )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    {
    }
};

#endif // __ConfigurationError_h__

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
