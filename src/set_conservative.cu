#include "set_conservative.h"

#include <thrust/copy.h>


namespace block3d_cuda {

  void set_conservative(const Block3dInfo *block_info, Block3dData *block_data,
			const value_type *Q) {

    // Transfer conservative variable data from host to device 
  
    const size_type array_size = block_info->NEQ * block_info->IM_G * block_info->JM_G * block_info->KM_G;

    // Host → device
    thrust::copy(Q, Q + array_size, block_data->Q.begin());

    // Device → device
    block_data->Q_p = block_data->Q;

  }

}
