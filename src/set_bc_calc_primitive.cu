#include "set_bc_calc_primitive.h"
#include "primitive_bc_kernel.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern const size_type num_thread_x;
  extern const size_type num_thread_y;
  extern const size_type num_thread_z;

  // ---------------------------------------------------------------------------

  void set_bc_calc_primitive(const Block3dInfo *block_info, Block3dData *block_data,
			     value_type *Q) {

    // Apply boundary conditions and compute primitive variables

    dim3 num_threads(num_thread_x, num_thread_y, num_thread_z);
    dim3 num_blocks((block_info->IM + num_threads.x - 1) / num_threads.x,
		    (block_info->JM + num_threads.y - 1) / num_threads.y,
		    (block_info->KM + num_threads.z - 1) / num_threads.z
		    );
  
    primitive_bc_kernel<<< num_blocks, num_threads >>>(Q,
						       block_data->rho_ptr(),
						       block_data->u_ptr(),
						       block_data->v_ptr(),
						       block_data->w_ptr(),
						       block_data->p_ptr(),
						       block_data->T_ptr(),
						       block_data->xi_x_ptr(),
						       block_data->xi_y_ptr(),
						       block_data->xi_z_ptr(),
						       block_data->eta_x_ptr(),
						       block_data->eta_y_ptr(),
						       block_data->eta_z_ptr(),
						       block_data->zeta_x_ptr(),
						       block_data->zeta_y_ptr(),
						       block_data->zeta_z_ptr(),
						       block_data->Jac_ptr()
						       );

    ERROR_CHECK( cudaDeviceSynchronize() );
    
  }

}
