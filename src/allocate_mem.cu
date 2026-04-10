#include "allocate_mem.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern __constant__ Block3dInfo blk_info;
  
  // ---------------------------------------------------------------------------


  void allocate_mem(const Block3dInfo *block_info, Block3dData *block_data) {

    const size_type NEQ = block_info->NEQ;

    const size_type IM = block_info->IM;
    const size_type JM = block_info->JM;
    const size_type KM = block_info->KM;

    const size_type IM_G = block_info->IM_G;
    const size_type JM_G = block_info->JM_G;
    const size_type KM_G = block_info->KM_G;
  
    // Allocates device memory (RAII via thrust::device_vector)
    size_type array_size = IM * JM * KM;
    block_data->dt.resize(array_size);

    array_size = IM_G * JM_G * KM_G;

    block_data->xi_x.resize(array_size);
    block_data->xi_y.resize(array_size);
    block_data->xi_z.resize(array_size);
    block_data->eta_x.resize(array_size);
    block_data->eta_y.resize(array_size);
    block_data->eta_z.resize(array_size);
    block_data->zeta_x.resize(array_size);
    block_data->zeta_y.resize(array_size);
    block_data->zeta_z.resize(array_size);

    block_data->Jac.resize(array_size);

    block_data->rho.resize(array_size);
    block_data->u.resize(array_size);
    block_data->v.resize(array_size);
    block_data->w.resize(array_size);
    block_data->p.resize(array_size);

#ifndef IS_INVISCID
    block_data->T.resize(array_size);
    block_data->mu.resize(array_size);

    block_data->u_xi.resize(array_size);
    block_data->v_xi.resize(array_size);
    block_data->w_xi.resize(array_size);
    block_data->u_eta.resize(array_size);
    block_data->v_eta.resize(array_size);
    block_data->w_eta.resize(array_size);
    block_data->u_zeta.resize(array_size);
    block_data->v_zeta.resize(array_size);
    block_data->w_zeta.resize(array_size);

    block_data->T_xi.resize(array_size);
    block_data->T_eta.resize(array_size);
    block_data->T_zeta.resize(array_size);

    block_data->tau_xx.resize(array_size);
    block_data->tau_yy.resize(array_size);
    block_data->tau_zz.resize(array_size);
    block_data->tau_xy.resize(array_size);
    block_data->tau_xz.resize(array_size);
    block_data->tau_yz.resize(array_size);

    block_data->q_x.resize(array_size);
    block_data->q_y.resize(array_size);
    block_data->q_z.resize(array_size);

    array_size = (NEQ - 1) * (IM_G - 2) * (JM_G - 2) * (KM_G - 2);

    block_data->Ev.resize(array_size);
    block_data->Fv.resize(array_size);
    block_data->Gv.resize(array_size);

    array_size = (NEQ - 1) * (IM - 2) * (JM - 2) * (KM - 2);

    block_data->diff_flux_vis.resize(array_size);
#endif
    
    array_size = NEQ * IM_G * JM_G * KM_G;
    block_data->Q.resize(array_size);
    block_data->Q_p.resize(array_size);
  
    array_size = NEQ * (IM - 1) * (JM - 2) * (KM - 2);
    block_data->Ep.resize(array_size);

    array_size = NEQ * (IM - 2) * (JM - 1) * (KM - 2);
    block_data->Fp.resize(array_size);

    array_size = NEQ * (IM - 2) * (JM - 2) * (KM - 1);
    block_data->Gp.resize(array_size);
  
  }

}
