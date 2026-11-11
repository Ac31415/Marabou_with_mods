/*********************                                                        */
/*! \file DantzigsRule.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz, Duligur Ibeling
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

**/

#ifndef __DantzigsRule_h__
#define __DantzigsRule_h__

#include "EntrySelectionStrategy.h"

<<<<<<< HEAD
#define DANTZIG_LOG( x, ... ) LOG( GlobalConfiguration::DANTZIGS_RULE_LOGGING, "DantzigsRule: %s\n", x )
=======
#define DANTZIG_LOG( x, ... )                                                                      \
    LOG( GlobalConfiguration::DANTZIGS_RULE_LOGGING, "DantzigsRule: %s\n", x )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

class String;

class DantzigsRule : public EntrySelectionStrategy
{
public:
    /*
      Apply Dantzig's rule: choose the candidate associated with the
      largest coefficient (in absolute value) in the cost function.
    */
<<<<<<< HEAD
    bool select( ITableau &tableau,
                 const List<unsigned> &candidates,
                 const Set<unsigned> &excluded );
=======
    bool
    select( ITableau &tableau, const List<unsigned> &candidates, const Set<unsigned> &excluded );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
};

#endif // __DantzigsRule_h__

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
