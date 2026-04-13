/**
 * @brief Header file defining utilities and structures for CUDA-based simulations.
 *
 * This file contains:
 * - Error checking utilities for CUDA API calls.
 * - A structure to encapsulate computational block information for simulations.
 * - A structure to manage device memory pointers for flow field data.
 */

#ifndef _BLOCK3D_CUDA_H_
#define _BLOCK3D_CUDA_H_

#include <cstdio>

// Windows headers (pulled in by cuda_runtime.h) define min/max macros and break Thrust/CUB.
#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#endif

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>

#include "constants.h"

// Macro to check the return status of CUDA API calls and handle errors.
#define ERROR_CHECK(call)                                                   \
{                                                                           \
  const cudaError_t error = call;                                           \
  if (error != cudaSuccess) {                                               \
    std::printf("Error: %s:%d, ", __FILE__, __LINE__);			    \
    std::printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
    std::exit(EXIT_FAILURE);                                                \
  }                                                                         \
}


namespace block3d_cuda {

  struct Block3dInfo {

    // Contains essential data for computational blocks in a simulation.

    // Constant to define the number of equations to solve
    static const size_type NEQ {constant::NEQ};

    // Number of ghost points (used in boundary conditions)
    static const size_type NG {constant::NG};

    // Number of points in the computational mesh 
    size_type IM; // Number of points in the i-direction
    size_type JM; // Number of points in the j-direction
    size_type KM; // Number of points in the k-direction

    size_type IM_G; 
    size_type JM_G; 
    size_type KM_G; 

    value_type CFL;

    // Test case parameters
    value_type angle_attack;
    
    size_type i_begin;
    size_type i_end;

    // Physical parameters relevant to the simulation
    value_type Mach;
    value_type Re;
    value_type Mach2;		
    value_type gM2;		
    value_type Re_inv;		

    value_type p_inf;		

    value_type gamma;		
    value_type Pr;
    value_type Pr_t;
    value_type gam1;		
    value_type gam1_inv;	
    value_type Pr_inv;		
    value_type Pr_t_inv;
    value_type gPr;

    value_type C_T_inf;		
    value_type C_dt_v;		

    /* ---------------------------------------------------------------------- */

    // Utility functions for converting multi-dimensional indices to one-dimensional indices.
    
    __forceinline__ __device__ size_type get_node_num() {
      return IM * JM * KM;
    }

    __forceinline__ __device__ size_type get_idx_x(size_type i, size_type j, size_type k) {
      return i + IM * (j + JM * k);
    }

    __forceinline__ __device__ size_type get_idx_u(size_type i, size_type j, size_type k) {
      return i + IM_G * (j + JM_G * k);
    }

    __forceinline__ __device__ size_type get_idx(index_type i, index_type j, index_type k) {
      return (NG+i) + IM_G * ((NG+j) + JM_G * (NG+k));
    }

    __forceinline__ __device__ size_type get_idx_Qa(size_type n_eq,
						    size_type i, size_type j, size_type k) {
      return n_eq + NEQ * (i + IM_G * (j + JM_G * k));
    }

    __forceinline__ __device__ size_type get_idx_Q(size_type n_eq,
						   index_type i, index_type j, index_type k) {
      return n_eq + NEQ * ((NG+i) + IM_G * ((NG+j) + JM_G * (NG+k)));
    }

    __forceinline__ __device__ size_type get_idx_Ep(size_type n_eq,
						    size_type i, size_type j, size_type k) {
      return n_eq + NEQ * (i + (IM - 1) * (j + (JM - 2) * k));
    }

    __forceinline__ __device__ size_type get_idx_Fp(size_type n_eq,
						    size_type i, size_type j, size_type k) {
      return n_eq + NEQ * (i + (IM - 2) * (j + (JM - 1) * k));
    }

    __forceinline__ __device__ size_type get_idx_Gp(size_type n_eq,
						    size_type i, size_type j, size_type k) {
      return n_eq + NEQ * (i + (IM - 2) * (j + (JM - 2) * k));
    }

    __forceinline__ __device__ size_type get_idx_Ev(size_type n_eq,
						    size_type i, size_type j, size_type k) {
      return n_eq + (NEQ - 1) * (i + (IM_G - 2) * (j + (JM_G - 2) * k));
    }

    __forceinline__ __device__ size_type get_idx_dfv(size_type n_eq,
						     size_type i, size_type j, size_type k) {
      return n_eq + (NEQ - 1) * (i + (IM - 2) * (j + (JM - 2) * k));
    }

    /* ---------------------------------------------------------------------- */
    
  };


  struct Block3dData {

    // Owns flow field data in device memory (RAII).
    // CUDA kernels will still consume raw pointers obtained via *_ptr() accessors.

    // Mesh coordinate data (temporary for metrics; kept here for simplicity)
    thrust::device_vector<value_type> x;
    thrust::device_vector<value_type> y;
    thrust::device_vector<value_type> z;

    // Geometric metrics for coordinate transformations
    thrust::device_vector<value_type> x_xi;
    thrust::device_vector<value_type> y_xi;
    thrust::device_vector<value_type> z_xi;
    thrust::device_vector<value_type> x_eta;
    thrust::device_vector<value_type> y_eta;
    thrust::device_vector<value_type> z_eta;
    thrust::device_vector<value_type> x_zeta;
    thrust::device_vector<value_type> y_zeta;
    thrust::device_vector<value_type> z_zeta;

    thrust::device_vector<value_type> xi_x;
    thrust::device_vector<value_type> xi_y;
    thrust::device_vector<value_type> xi_z;
    thrust::device_vector<value_type> eta_x;
    thrust::device_vector<value_type> eta_y;
    thrust::device_vector<value_type> eta_z;
    thrust::device_vector<value_type> zeta_x;
    thrust::device_vector<value_type> zeta_y;
    thrust::device_vector<value_type> zeta_z;

    thrust::device_vector<value_type> Jac;

    // Flow field data
    thrust::device_vector<value_type> rho;
    thrust::device_vector<value_type> u;
    thrust::device_vector<value_type> v;
    thrust::device_vector<value_type> w;
    thrust::device_vector<value_type> p;
    
    thrust::device_vector<value_type> Q;
    thrust::device_vector<value_type> Q_p;

    thrust::device_vector<value_type> T;
    thrust::device_vector<value_type> mu;

#ifndef IS_INVISCID
    // Spatial derivatives of flow field variables
    thrust::device_vector<value_type> u_xi;
    thrust::device_vector<value_type> v_xi;
    thrust::device_vector<value_type> w_xi;
    thrust::device_vector<value_type> u_eta;
    thrust::device_vector<value_type> v_eta;
    thrust::device_vector<value_type> w_eta;
    thrust::device_vector<value_type> u_zeta;
    thrust::device_vector<value_type> v_zeta;
    thrust::device_vector<value_type> w_zeta;

    thrust::device_vector<value_type> T_xi;
    thrust::device_vector<value_type> T_eta;
    thrust::device_vector<value_type> T_zeta;

    // Stress tensor
    thrust::device_vector<value_type> tau_xx;
    thrust::device_vector<value_type> tau_yy;
    thrust::device_vector<value_type> tau_zz;
    thrust::device_vector<value_type> tau_xy;
    thrust::device_vector<value_type> tau_xz;
    thrust::device_vector<value_type> tau_yz;

    // Heat flux
    thrust::device_vector<value_type> q_x;
    thrust::device_vector<value_type> q_y;
    thrust::device_vector<value_type> q_z;

    // Viscous flux 
    thrust::device_vector<value_type> Ev;
    thrust::device_vector<value_type> Fv;
    thrust::device_vector<value_type> Gv;
#endif
    
    // Derivative of viscous fluxes
    thrust::device_vector<value_type> diff_flux_vis;

    // Reconstructed numerical flux for inviscid term
    thrust::device_vector<value_type> Ep;
    thrust::device_vector<value_type> Fp;
    thrust::device_vector<value_type> Gp;

    // Local time step
    thrust::device_vector<value_type> dt;

    // Raw pointer helpers (for existing kernels)
    value_type* x_ptr() { return thrust::raw_pointer_cast(x.data()); }
    value_type* y_ptr() { return thrust::raw_pointer_cast(y.data()); }
    value_type* z_ptr() { return thrust::raw_pointer_cast(z.data()); }

    value_type* x_xi_ptr() { return thrust::raw_pointer_cast(x_xi.data()); }
    value_type* y_xi_ptr() { return thrust::raw_pointer_cast(y_xi.data()); }
    value_type* z_xi_ptr() { return thrust::raw_pointer_cast(z_xi.data()); }
    value_type* x_eta_ptr() { return thrust::raw_pointer_cast(x_eta.data()); }
    value_type* y_eta_ptr() { return thrust::raw_pointer_cast(y_eta.data()); }
    value_type* z_eta_ptr() { return thrust::raw_pointer_cast(z_eta.data()); }
    value_type* x_zeta_ptr() { return thrust::raw_pointer_cast(x_zeta.data()); }
    value_type* y_zeta_ptr() { return thrust::raw_pointer_cast(y_zeta.data()); }
    value_type* z_zeta_ptr() { return thrust::raw_pointer_cast(z_zeta.data()); }

    value_type* xi_x_ptr() { return thrust::raw_pointer_cast(xi_x.data()); }
    value_type* xi_y_ptr() { return thrust::raw_pointer_cast(xi_y.data()); }
    value_type* xi_z_ptr() { return thrust::raw_pointer_cast(xi_z.data()); }
    value_type* eta_x_ptr() { return thrust::raw_pointer_cast(eta_x.data()); }
    value_type* eta_y_ptr() { return thrust::raw_pointer_cast(eta_y.data()); }
    value_type* eta_z_ptr() { return thrust::raw_pointer_cast(eta_z.data()); }
    value_type* zeta_x_ptr() { return thrust::raw_pointer_cast(zeta_x.data()); }
    value_type* zeta_y_ptr() { return thrust::raw_pointer_cast(zeta_y.data()); }
    value_type* zeta_z_ptr() { return thrust::raw_pointer_cast(zeta_z.data()); }
    value_type* Jac_ptr() { return thrust::raw_pointer_cast(Jac.data()); }

    value_type* rho_ptr() { return thrust::raw_pointer_cast(rho.data()); }
    value_type* u_ptr() { return thrust::raw_pointer_cast(u.data()); }
    value_type* v_ptr() { return thrust::raw_pointer_cast(v.data()); }
    value_type* w_ptr() { return thrust::raw_pointer_cast(w.data()); }
    value_type* p_ptr() { return thrust::raw_pointer_cast(p.data()); }

    value_type* Q_ptr() { return thrust::raw_pointer_cast(Q.data()); }
    value_type* Q_p_ptr() { return thrust::raw_pointer_cast(Q_p.data()); }

    value_type* T_ptr() { return thrust::raw_pointer_cast(T.data()); }
    value_type* mu_ptr() { return thrust::raw_pointer_cast(mu.data()); }

#ifndef IS_INVISCID
    value_type* u_xi_ptr() { return thrust::raw_pointer_cast(u_xi.data()); }
    value_type* v_xi_ptr() { return thrust::raw_pointer_cast(v_xi.data()); }
    value_type* w_xi_ptr() { return thrust::raw_pointer_cast(w_xi.data()); }
    value_type* u_eta_ptr() { return thrust::raw_pointer_cast(u_eta.data()); }
    value_type* v_eta_ptr() { return thrust::raw_pointer_cast(v_eta.data()); }
    value_type* w_eta_ptr() { return thrust::raw_pointer_cast(w_eta.data()); }
    value_type* u_zeta_ptr() { return thrust::raw_pointer_cast(u_zeta.data()); }
    value_type* v_zeta_ptr() { return thrust::raw_pointer_cast(v_zeta.data()); }
    value_type* w_zeta_ptr() { return thrust::raw_pointer_cast(w_zeta.data()); }

    value_type* T_xi_ptr() { return thrust::raw_pointer_cast(T_xi.data()); }
    value_type* T_eta_ptr() { return thrust::raw_pointer_cast(T_eta.data()); }
    value_type* T_zeta_ptr() { return thrust::raw_pointer_cast(T_zeta.data()); }

    value_type* tau_xx_ptr() { return thrust::raw_pointer_cast(tau_xx.data()); }
    value_type* tau_yy_ptr() { return thrust::raw_pointer_cast(tau_yy.data()); }
    value_type* tau_zz_ptr() { return thrust::raw_pointer_cast(tau_zz.data()); }
    value_type* tau_xy_ptr() { return thrust::raw_pointer_cast(tau_xy.data()); }
    value_type* tau_xz_ptr() { return thrust::raw_pointer_cast(tau_xz.data()); }
    value_type* tau_yz_ptr() { return thrust::raw_pointer_cast(tau_yz.data()); }

    value_type* q_x_ptr() { return thrust::raw_pointer_cast(q_x.data()); }
    value_type* q_y_ptr() { return thrust::raw_pointer_cast(q_y.data()); }
    value_type* q_z_ptr() { return thrust::raw_pointer_cast(q_z.data()); }

    value_type* Ev_ptr() { return thrust::raw_pointer_cast(Ev.data()); }
    value_type* Fv_ptr() { return thrust::raw_pointer_cast(Fv.data()); }
    value_type* Gv_ptr() { return thrust::raw_pointer_cast(Gv.data()); }
#endif

    value_type* diff_flux_vis_ptr() { return thrust::raw_pointer_cast(diff_flux_vis.data()); }

    value_type* Ep_ptr() { return thrust::raw_pointer_cast(Ep.data()); }
    value_type* Fp_ptr() { return thrust::raw_pointer_cast(Fp.data()); }
    value_type* Gp_ptr() { return thrust::raw_pointer_cast(Gp.data()); }

    value_type* dt_ptr() { return thrust::raw_pointer_cast(dt.data()); }
    
  };
    
}

#endif /* _BLOCK3D_CUDA_H_ */
