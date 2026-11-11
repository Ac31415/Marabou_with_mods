/*********************                                                        */
/*! \file FloatUtils.h
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

#ifndef __FloatUtils_h__
#define __FloatUtils_h__

#include "Debug.h"
#include "GlobalConfiguration.h"
#include "MString.h"

#include <cfloat>
#include <cmath>

#ifdef _WIN32
#undef max
#undef min
#endif

class FloatUtils
{
public:
<<<<<<< HEAD
    static bool areEqual( double x, double y, double epsilon = GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS );
    static String doubleToString( double x,
                                  unsigned precision = GlobalConfiguration::DEFAULT_DOUBLE_TO_STRING_PRECISION );
=======
    static bool areEqual( double x,
                          double y,
                          double epsilon = GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS );
    static String
    doubleToString( double x,
                    unsigned precision = GlobalConfiguration::DEFAULT_DOUBLE_TO_STRING_PRECISION );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

    static bool isNan( double x );

    static bool isInf( double x );

<<<<<<< HEAD
    static bool isZero( double x, double epsilon = GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS )
    {
        ASSERT( epsilon > 0 );                                                                                                                
        double lower = -epsilon; 
        double upper = epsilon;                                                                                                               
        return ( x - upper ) * ( x - lower ) <= 0;
    }

    static bool isPositive( double x, double epsilon = GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS )
    {
        ASSERT( epsilon > 0 ); 
        return x > epsilon;
    }

    static bool isNegative( double x, double epsilon = GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS )
    {
        ASSERT( epsilon > 0 ); 
=======
    static bool isZero( double x,
                        double epsilon = GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS )
    {
        ASSERT( epsilon > 0 );
        double lower = -epsilon;
        double upper = epsilon;
        return ( x - upper ) * ( x - lower ) <= 0;
    }

    static bool isPositive( double x,
                            double epsilon = GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS )
    {
        ASSERT( epsilon > 0 );
        return x > epsilon;
    }

    static bool isNegative( double x,
                            double epsilon = GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS )
    {
        ASSERT( epsilon > 0 );
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
        return x < -epsilon;
    }

    static double abs( double x )
    {
        return fabs( x );
    }

    static bool areDisequal( double x,
                             double y,
                             double epsilon = GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS )
    {
        return !areEqual( x, y, epsilon );
    }

<<<<<<< HEAD
    static double roundToZero( double x, double epsilon = GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS )
=======
    static double
    roundToZero( double x, double epsilon = GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    {
        return isZero( x, epsilon ) ? 0.0 : x;
    }

<<<<<<< HEAD
    static bool gt( double x, double y, double epsilon = GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS )
=======
    static bool
    gt( double x, double y, double epsilon = GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    {
        return isPositive( x - y, epsilon );
    }

<<<<<<< HEAD
    static bool gte( double x, double y, double epsilon = GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS )
=======
    static bool
    gte( double x, double y, double epsilon = GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    {
        return !isNegative( x - y, epsilon );
    }

<<<<<<< HEAD
    static bool lt( double x, double y, double epsilon = GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS )
=======
    static bool
    lt( double x, double y, double epsilon = GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    {
        return gt( y, x, epsilon );
    }

<<<<<<< HEAD
    static bool lte( double x, double y, double epsilon = GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS )
=======
    static bool
    lte( double x, double y, double epsilon = GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    {
        return gte( y, x, epsilon );
    }

<<<<<<< HEAD
    static double min( double x, double y, double epsilon = GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS )
=======
    static double
    min( double x, double y, double epsilon = GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    {
        return lt( x, y, epsilon ) ? x : y;
    }

<<<<<<< HEAD
    static double max( double x, double y, double epsilon = GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS )
=======
    static double
    max( double x, double y, double epsilon = GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    {
        return gt( x, y, epsilon ) ? x : y;
    }

<<<<<<< HEAD
=======
    static double round( double x,
                         double epsilon = GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS )
    {
        // # Implement pytorch round https://pytorch.org/docs/stable/generated/torch.round.html
        // "Values equidistant from two integers are rounded towards the nearest even value"
        // i.e., round(2.5) == round(1.5) == 2
        double f = floor( x );
        double c = ceil( x );
        if ( areEqual( x - f, 0.5, epsilon ) && areEqual( c - x, 0.5, epsilon ) )
            return ( (int)c % 2 == 0 ) ? c : f;
        else
            return gt( x - f, 0.5, epsilon ) ? c : f;
    }

>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    static double infinity()
    {
        return DBL_MAX;
    }

    static double negativeInfinity()
    {
        return -DBL_MAX;
    }

    static bool isFinite( double x )
    {
        return ( x != infinity() ) && ( x != negativeInfinity() );
    }

    static bool wellFormed( double x )
    {
        return !isNan( x ) && !isInf( x );
    }
<<<<<<< HEAD

=======
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
};

#endif // __FloatUtils_h__

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
