#include "get_metrics.h"

#include <thrust/copy.h>

namespace block3d_cuda {

  void get_metrics(const Block3dInfo *block_info, const Block3dData *block_data,
		   value_type* xi_x, value_type* xi_y, value_type* xi_z,
		   value_type* eta_x, value_type* eta_y, value_type* eta_z,
		   value_type* zeta_x, value_type* zeta_y, value_type* zeta_z,
		   value_type* Jac) {
    
    // Transfer computed geometric metrics from the device to the host

    const size_type IM = block_info->IM;
    const size_type JM = block_info->JM;
    const size_type KM = block_info->KM;

    const size_type NG = block_info->NG;

    const size_type IM_G = IM + 2*NG;
    const size_type JM_G = JM + 2*NG;
    const size_type KM_G = KM + 2*NG;

    auto copy_array = [&](const value_type* p_s, value_type* p_d) {
      for(size_type k = 0; k < KM; k++) {
	for(size_type j = 0; j < JM; j++) {
	  for(size_type i = 0; i < IM; i++) {
	    size_type idx1 = i + IM * (j + JM * k);
	    size_type idx2 = (NG+i) + IM_G * ((NG+j) + JM_G * (NG+k));
	    p_d[idx1] = p_s[idx2];
	  }
	}
      }
    };

    const size_type array_size = IM_G * JM_G * KM_G;
    value_type *tmp = new value_type[array_size];

    thrust::copy(block_data->xi_x.cbegin(), block_data->xi_x.cbegin() + array_size, tmp);
    copy_array(tmp, xi_x);
    thrust::copy(block_data->xi_y.cbegin(), block_data->xi_y.cbegin() + array_size, tmp);
    copy_array(tmp, xi_y);
    thrust::copy(block_data->xi_z.cbegin(), block_data->xi_z.cbegin() + array_size, tmp);
    copy_array(tmp, xi_z);
  
    thrust::copy(block_data->eta_x.cbegin(), block_data->eta_x.cbegin() + array_size, tmp);
    copy_array(tmp, eta_x);
    thrust::copy(block_data->eta_y.cbegin(), block_data->eta_y.cbegin() + array_size, tmp);
    copy_array(tmp, eta_y);
    thrust::copy(block_data->eta_z.cbegin(), block_data->eta_z.cbegin() + array_size, tmp);
    copy_array(tmp, eta_z);
  
    thrust::copy(block_data->zeta_x.cbegin(), block_data->zeta_x.cbegin() + array_size, tmp);
    copy_array(tmp, zeta_x);
    thrust::copy(block_data->zeta_y.cbegin(), block_data->zeta_y.cbegin() + array_size, tmp);
    copy_array(tmp, zeta_y);
    thrust::copy(block_data->zeta_z.cbegin(), block_data->zeta_z.cbegin() + array_size, tmp);
    copy_array(tmp, zeta_z);

    thrust::copy(block_data->Jac.cbegin(), block_data->Jac.cbegin() + array_size, tmp);
    copy_array(tmp, Jac);

    delete[] tmp;
  
  }

}
