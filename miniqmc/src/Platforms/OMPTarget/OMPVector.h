////////////////////////////////////////////////////////////////////////////////
//// This file is distributed under the University of Illinois/NCSA Open Source
//// License.  See LICENSE file in top directory for details.
////
//// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
////
//// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory.
////
//// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory.
//////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_OMP_VECTOR_HPP
#define QMCPLUSPLUS_OMP_VECTOR_HPP

#include <vector>

namespace qmcplusplus
{

template<typename T, class Container = std::vector<T>>
class OMPVector : public Container
{
  protected:
  size_t device_id;
  T * vec_ptr;

  public:
  inline OMPVector(size_t size = 0, size_t id = 0): device_id(id), vec_ptr(nullptr)
  {
    resize(size);
  }

  inline void resize(size_t size)
  {
    if(size!=Container::size())
    {
      if(Container::size()!=0)
      {
#ifdef ENABLE_OFFLOAD
        #pragma omp target exit data map(delete:vec_ptr) device(device_id)
#endif
        vec_ptr = nullptr;
      }
      Container::resize(size);
      if(size>0)
      {
        vec_ptr = Container::data();
        //std::cout << "YYYY resize OMPVector " << Container::size() << std::endl;
#ifdef ENABLE_OFFLOAD
        #pragma omp target enter data map(alloc:vec_ptr[0:size]) device(device_id)
#endif
      }
    }
  }

  inline void update_to_device() const
  {
#ifdef ENABLE_OFFLOAD
    #pragma omp target update to(vec_ptr[0:Container::size()]) device(device_id)
#endif
  }

  inline void update_from_device() const 
  {
#ifdef ENABLE_OFFLOAD
    #pragma omp target update from(vec_ptr[0:Container::size()]) device(device_id)
#endif
  }

  inline ~OMPVector()
  {
#ifdef ENABLE_OFFLOAD
    #pragma omp target exit data map(delete:vec_ptr) device(device_id)
#endif
  }

};

}
#endif
