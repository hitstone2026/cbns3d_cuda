#include "calc_conservative.h"
#include "conservative_kernel.h"

#include <thrust/copy.h>


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern const size_type num_thread_x;
  extern const size_type num_thread_y;
  extern const size_type num_thread_z;

  // ---------------------------------------------------------------------------

  void calc_conservative(const Block3dInfo *block_info, Block3dData *block_data,
			 value_type *Q) {

    // Compute conservative variables

    dim3 num_threads(num_thread_x, num_thread_y, num_thread_z);
    dim3 num_blocks((block_info->IM_G + num_threads.x - 1) / num_threads.x,
		    (block_info->JM_G + num_threads.y - 1) / num_threads.y,
		    (block_info->KM_G + num_threads.z - 1) / num_threads.z
		    );
    
    conservative_kernel<<< num_blocks, num_threads >>>(block_data->rho_ptr(),
						       block_data->u_ptr(),
						       block_data->v_ptr(),
						       block_data->w_ptr(),
						       block_data->p_ptr(),
						       Q
						       );

    ERROR_CHECK( cudaDeviceSynchronize() );

    // Keep Q_p in sync with Q after updates (device→device)
    block_data->Q_p = block_data->Q;

  }

}
