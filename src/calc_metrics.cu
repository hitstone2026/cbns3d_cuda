#include "calc_metrics.h"
#include "gradient_kernel.h"
#include "metrics_kernel.h"

#include <thrust/copy.h>


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern const size_type num_thread_x;
  extern const size_type num_thread_y;
  extern const size_type num_thread_z;

  // ---------------------------------------------------------------------------

  void calc_metrics(const Block3dInfo *block_info, Block3dData *block_data,
		    const value_type *x, const value_type *y, const value_type *z) {

    // Calculate geometric parameters for coordinate transformation

    // Transfer coordinate data from host memory to device memory
    const size_type array_size = block_info->IM * block_info->JM * block_info->KM;

    block_data->x.resize(array_size);
    block_data->y.resize(array_size);
    block_data->z.resize(array_size);

    thrust::copy(x, x + array_size, block_data->x.begin());
    thrust::copy(y, y + array_size, block_data->y.begin());
    thrust::copy(z, z + array_size, block_data->z.begin());

    block_data->x_xi.resize(array_size);
    block_data->y_xi.resize(array_size);
    block_data->z_xi.resize(array_size);
    block_data->x_eta.resize(array_size);
    block_data->y_eta.resize(array_size);
    block_data->z_eta.resize(array_size);
    block_data->x_zeta.resize(array_size);
    block_data->y_zeta.resize(array_size);
    block_data->z_zeta.resize(array_size);

    dim3 num_threads(num_thread_x, num_thread_y, num_thread_z);
    dim3 num_blocks((block_info->IM + num_threads.x - 1) / num_threads.x,
		    (block_info->JM + num_threads.y - 1) / num_threads.y,
		    (block_info->KM + num_threads.z - 1) / num_threads.z
		    );
  
    gradient_kernel<<< num_blocks, num_threads >>>(block_data->x_ptr(),
						   block_data->x_xi_ptr(),
						   block_data->x_eta_ptr(),
						   block_data->x_zeta_ptr()
						   );

    gradient_kernel<<< num_blocks, num_threads >>>(block_data->y_ptr(),
						   block_data->y_xi_ptr(),
						   block_data->y_eta_ptr(),
						   block_data->y_zeta_ptr()
						   );
  
    gradient_kernel<<< num_blocks, num_threads >>>(block_data->z_ptr(),
						   block_data->z_xi_ptr(),
						   block_data->z_eta_ptr(),
						   block_data->z_zeta_ptr()
						   );

    ERROR_CHECK( cudaDeviceSynchronize() );

    // -------------------------------------------------------------------------

    metrics_kernel<<< num_blocks, num_threads >>>(block_data->x_xi_ptr(),
						  block_data->x_eta_ptr(),
						  block_data->x_zeta_ptr(),
						  block_data->y_xi_ptr(),
						  block_data->y_eta_ptr(),
						  block_data->y_zeta_ptr(),
						  block_data->z_xi_ptr(),
						  block_data->z_eta_ptr(),
						  block_data->z_zeta_ptr(),
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

    // Temporary coordinate buffers/gradients are kept in Block3dData for now.
    // If desired later, we can clear them here to release memory:
    // block_data->x.clear(); ... etc.
  }

}
