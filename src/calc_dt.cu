#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#include "calc_dt.h"
#include "time_step_kernel.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern const size_type num_thread_x;
  extern const size_type num_thread_y;
  extern const size_type num_thread_z;

  // ---------------------------------------------------------------------------

  value_type calc_dt(const Block3dInfo *block_info, Block3dData *block_data) {
    
    // Compute time step using CFL condition to ensure numerical stability

    dim3 num_threads(num_thread_x, num_thread_y, num_thread_z);
    dim3 num_blocks((block_info->IM + num_threads.x - 1) / num_threads.x,
		    (block_info->JM + num_threads.y - 1) / num_threads.y,
		    (block_info->KM + num_threads.z - 1) / num_threads.z
		    );
    
    time_step_kernel<<< num_blocks, num_threads >>>(block_data->rho_ptr(),
						    block_data->u_ptr(),
						    block_data->v_ptr(),
						    block_data->w_ptr(),
						    block_data->p_ptr(),
						    block_data->mu_ptr(),
						    block_data->xi_x_ptr(),
						    block_data->xi_y_ptr(),
						    block_data->xi_z_ptr(),
						    block_data->eta_x_ptr(),
						    block_data->eta_y_ptr(),
						    block_data->eta_z_ptr(),
						    block_data->zeta_x_ptr(),
						    block_data->zeta_y_ptr(),
						    block_data->zeta_z_ptr(),
						    block_data->dt_ptr()
						    );

    ERROR_CHECK( cudaDeviceSynchronize() );

    const size_type array_size = block_info->IM * block_info->JM * block_info->KM;

    thrust::device_ptr<value_type> result =
      thrust::max_element(thrust::device_pointer_cast(block_data->dt_ptr()),
			  thrust::device_pointer_cast(block_data->dt_ptr()) + array_size);

    const value_type dt = block_info->CFL / *result;

    if(std::isnan(dt)) {
      throw std::runtime_error("An error occurred while calculating the time step.\n");
    }

    return dt;

  }

}
