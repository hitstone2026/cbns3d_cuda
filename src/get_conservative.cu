#include "get_conservative.h"

#include <thrust/copy.h>


namespace block3d_cuda {

  void get_conservative(const Block3dInfo *block_info, const Block3dData *block_data,
			value_type *Q) {

    // Transfer conservative variable data from device to host 
  
    const size_type array_size = block_info->NEQ * block_info->IM_G * block_info->JM_G * block_info->KM_G;

    thrust::copy(block_data->Q.cbegin(), block_data->Q.cbegin() + array_size, Q);

  }

}
