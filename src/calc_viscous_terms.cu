#include "calc_viscous_terms.h"
#include "gradient_kernel.h"
#include "viscous_terms_kernel.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern const size_type num_thread_x;
  extern const size_type num_thread_y;
  extern const size_type num_thread_z;

  // ---------------------------------------------------------------------------

  void calc_viscous_terms(const Block3dInfo *block_info, Block3dData *block_data) {

    // Compute viscosity-related terms for the Navier-Stokes equations
  
#ifndef IS_INVISCID
    dim3 num_threads(num_thread_x, num_thread_y, num_thread_z);
    dim3 num_blocks((block_info->IM_G + num_threads.x - 1) / num_threads.x,
		    (block_info->JM_G + num_threads.y - 1) / num_threads.y,
		    (block_info->KM_G + num_threads.z - 1) / num_threads.z
		    );
  
    gradient_wg_kernel<<< num_blocks, num_threads >>>(block_data->u_ptr(),
						      block_data->u_xi_ptr(),
						      block_data->u_eta_ptr(),
						      block_data->u_zeta_ptr()
						      );

    gradient_wg_kernel<<< num_blocks, num_threads >>>(block_data->v_ptr(),
						      block_data->v_xi_ptr(),
						      block_data->v_eta_ptr(),
						      block_data->v_zeta_ptr()
						      );
  
    gradient_wg_kernel<<< num_blocks, num_threads >>>(block_data->w_ptr(),
						      block_data->w_xi_ptr(),
						      block_data->w_eta_ptr(),
						      block_data->w_zeta_ptr()
						      );

    gradient_wg_kernel<<< num_blocks, num_threads >>>(block_data->T_ptr(),
						      block_data->T_xi_ptr(),
						      block_data->T_eta_ptr(),
						      block_data->T_zeta_ptr()
						      );
    ERROR_CHECK( cudaDeviceSynchronize() );

    viscous_terms_kernel<<< num_blocks, num_threads >>>(block_data->T_ptr(),
							block_data->xi_x_ptr(),
							block_data->xi_y_ptr(),
							block_data->xi_z_ptr(),
							block_data->eta_x_ptr(),
							block_data->eta_y_ptr(),
							block_data->eta_z_ptr(),
							block_data->zeta_x_ptr(),
							block_data->zeta_y_ptr(),
							block_data->zeta_z_ptr(),
							block_data->u_xi_ptr(),
							block_data->u_eta_ptr(),
							block_data->u_zeta_ptr(),
							block_data->v_xi_ptr(),
							block_data->v_eta_ptr(),
							block_data->v_zeta_ptr(),
							block_data->w_xi_ptr(),
							block_data->w_eta_ptr(),
							block_data->w_zeta_ptr(),
							block_data->T_xi_ptr(),
							block_data->T_eta_ptr(),
							block_data->T_zeta_ptr(),
							block_data->mu_ptr(),
							block_data->tau_xx_ptr(),
							block_data->tau_yy_ptr(),
							block_data->tau_zz_ptr(),
							block_data->tau_xy_ptr(),
							block_data->tau_xz_ptr(),
							block_data->tau_yz_ptr(),
							block_data->q_x_ptr(),
							block_data->q_y_ptr(),
							block_data->q_z_ptr()
							);
    
    ERROR_CHECK( cudaDeviceSynchronize() );
#endif
  }

}
