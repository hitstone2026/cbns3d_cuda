#include "get_primitive.h"

#include <thrust/copy.h>


namespace block3d_cuda {

  void get_primitive(const Block3dInfo *block_info, const Block3dData *block_data,
		     value_type *rho, value_type *u, value_type *v, value_type *w, value_type *p) {

    // Transfer primitive variable data from device to host

    const size_type array_size = block_info->IM_G * block_info->JM_G * block_info->KM_G;

    thrust::copy(block_data->rho.cbegin(), block_data->rho.cbegin() + array_size, rho);
    thrust::copy(block_data->u.cbegin(), block_data->u.cbegin() + array_size, u);
    thrust::copy(block_data->v.cbegin(), block_data->v.cbegin() + array_size, v);
    thrust::copy(block_data->w.cbegin(), block_data->w.cbegin() + array_size, w);
    thrust::copy(block_data->p.cbegin(), block_data->p.cbegin() + array_size, p);

  }

}
