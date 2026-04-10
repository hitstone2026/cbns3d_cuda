#include "free_mem.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern __constant__ Block3dInfo blk_info;
  
  // ---------------------------------------------------------------------------


  void free_mem(Block3dData *block_data) {

    // Device memory is owned by thrust::device_vector members in Block3dData.
    // Explicit release is optional; keep this as a size-reset helper for symmetry.
    block_data->dt.clear();

    block_data->xi_x.clear();
    block_data->xi_y.clear();
    block_data->xi_z.clear();
    block_data->eta_x.clear();
    block_data->eta_y.clear();
    block_data->eta_z.clear();
    block_data->zeta_x.clear();
    block_data->zeta_y.clear();
    block_data->zeta_z.clear();

    block_data->Jac.clear();

    block_data->rho.clear();
    block_data->u.clear();
    block_data->v.clear();
    block_data->w.clear();
    block_data->p.clear();

#ifndef IS_INVISCID
    block_data->T.clear();
    block_data->mu.clear();

    block_data->u_xi.clear();
    block_data->v_xi.clear();
    block_data->w_xi.clear();
    block_data->u_eta.clear();
    block_data->v_eta.clear();
    block_data->w_eta.clear();
    block_data->u_zeta.clear();
    block_data->v_zeta.clear();
    block_data->w_zeta.clear();

    block_data->T_xi.clear();
    block_data->T_eta.clear();
    block_data->T_zeta.clear();

    block_data->tau_xx.clear();
    block_data->tau_yy.clear();
    block_data->tau_zz.clear();
    block_data->tau_xy.clear();
    block_data->tau_xz.clear();
    block_data->tau_yz.clear();

    block_data->q_x.clear();
    block_data->q_y.clear();
    block_data->q_z.clear();

    block_data->Ev.clear();
    block_data->Fv.clear();
    block_data->Gv.clear();

    block_data->diff_flux_vis.clear();
#endif

    block_data->Q.clear();
    block_data->Q_p.clear();

    block_data->Ep.clear();
    block_data->Fp.clear();
    block_data->Gp.clear();
  }

}
