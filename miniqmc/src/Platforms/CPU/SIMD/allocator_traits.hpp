//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_ACCESS_TRAITS_H
#define QMCPLUSPLUS_ACCESS_TRAITS_H

namespace qmcplusplus
{
/** template class defines whether the memory allocated by the allocator is host accessible
 */
template<class Allocator>
struct allocator_traits
{
  const static bool is_host_accessible = true;
};

template<class Allocator>
using IsHostSafe = typename std::enable_if<allocator_traits<Allocator>::is_host_accessible>::type;

template<class Allocator>
using IsNotHostSafe = typename std::enable_if<!allocator_traits<Allocator>::is_host_accessible>::type;

} // namespace qmcplusplus

#endif // QMCPLUSPLUS_ACCESS_TRAITS_H
