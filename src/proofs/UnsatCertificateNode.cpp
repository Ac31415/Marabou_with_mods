/*********************                                                        */
/*! \file UnsatCertificateNode.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Omri Isac, Guy Katz
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2022 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]
 **/

<<<<<<< HEAD
#include <Options.h>
#include "UnsatCertificateNode.h"

UnsatCertificateNode::UnsatCertificateNode( UnsatCertificateNode *parent, PiecewiseLinearCaseSplit split )
=======
#include "UnsatCertificateNode.h"

#include <Options.h>

UnsatCertificateNode::UnsatCertificateNode( UnsatCertificateNode *parent,
                                            PiecewiseLinearCaseSplit split )
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
    : _parent( parent )
    , _contradiction( NULL )
    , _headSplit( std::move( split ) )
    , _hasSATSolution( false )
    , _wasVisited( false )
    , _delegationStatus( DelegationStatus::DONT_DELEGATE )
{
    if ( parent )
        parent->_children.append( this );
}

UnsatCertificateNode::~UnsatCertificateNode()
{
    for ( auto *child : _children )
    {
        if ( child )
        {
            delete child;
            child = NULL;
        }
    }

    _children.clear();
    deletePLCExplanations();

    if ( _contradiction )
    {
        delete _contradiction;
        _contradiction = NULL;
    }

    _parent = NULL;
}

void UnsatCertificateNode::setContradiction( Contradiction *contradiction )
{
    _contradiction = contradiction;
}

const Contradiction *UnsatCertificateNode::getContradiction() const
{
    return _contradiction;
}

UnsatCertificateNode *UnsatCertificateNode::getParent() const
{
    return _parent;
}

const PiecewiseLinearCaseSplit &UnsatCertificateNode::getSplit() const
{
    return _headSplit;
}

<<<<<<< HEAD
const List<UnsatCertificateNode*> &UnsatCertificateNode::getChildren() const
=======
const List<UnsatCertificateNode *> &UnsatCertificateNode::getChildren() const
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
{
    return _children;
}

const List<std::shared_ptr<PLCLemma>> &UnsatCertificateNode::getPLCLemmas() const
{
    return _PLCExplanations;
}

void UnsatCertificateNode::addPLCLemma( std::shared_ptr<PLCLemma> &explanation )
{
    _PLCExplanations.append( explanation );
}


<<<<<<< HEAD
UnsatCertificateNode *UnsatCertificateNode::getChildBySplit( const PiecewiseLinearCaseSplit &split ) const
=======
UnsatCertificateNode *
UnsatCertificateNode::getChildBySplit( const PiecewiseLinearCaseSplit &split ) const
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
{
    for ( UnsatCertificateNode *child : _children )
    {
        if ( child->_headSplit == split )
            return child;
    }

    return NULL;
}

bool UnsatCertificateNode::getSATSolutionFlag() const
{
    return _hasSATSolution;
}

void UnsatCertificateNode::setSATSolutionFlag()
{
    _hasSATSolution = true;
}

bool UnsatCertificateNode::getVisited() const
{
    return _wasVisited;
}

void UnsatCertificateNode::setVisited()
{
    _wasVisited = true;
}

DelegationStatus UnsatCertificateNode::getDelegationStatus() const
{
    return _delegationStatus;
}

void UnsatCertificateNode::setDelegationStatus( DelegationStatus delegationStatus )
{
    _delegationStatus = delegationStatus;
}

void UnsatCertificateNode::deletePLCExplanations()
{
    _PLCExplanations.clear();
}

void UnsatCertificateNode::makeLeaf()
{
    for ( UnsatCertificateNode *child : _children )
    {
        if ( child )
        {
            delete child;
            child = NULL;
        }
    }

    _children.clear();
}

bool UnsatCertificateNode::isValidLeaf() const
{
    return _contradiction && _children.empty();
}

bool UnsatCertificateNode::isValidNonLeaf() const
{
    return !_contradiction && !_children.empty();
}
