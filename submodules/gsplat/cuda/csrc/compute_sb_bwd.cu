#include "bindings.h"
#include "spherical_beta.cuh"

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <iostream>
namespace gsplat {

namespace cg = cooperative_groups;

    template <typename T>
    __global__ void compute_sh_bwd_kernel(
        const uint32_t N,
        const uint32_t num_primitives,
        const uint32_t active_primitives,
        const vec3<T> *__restrict__ dirs, // [N, 3]
        const T *__restrict__ c0,         // [3] base color
        const T *__restrict__ coeffs,     // [N, num_primitives, 6] [r, g, b, theta, phi, beta]
        const bool *__restrict__ masks,   // [N]
        const T *__restrict__ v_colors,   // [N, 3]

        T *__restrict__ v_c0,           // [N, 3]
        T *__restrict__ v_coeffs,         // [N, num_primitives, 6]
        T *__restrict__ v_dirs            // [N, 3] optional
    ) {
        // parallelize over N
        uint32_t idx = cg::this_grid().thread_rank();
        if (idx >= N) {
            return;
        }
        if (masks != nullptr && !masks[idx]) {
            return;
        }

        spherical_beta_isotropic_bwd(
            active_primitives,
            3,  // num_colors
            c0 + idx * 3,  // c0 is the first 3 elements of coeffs
            coeffs + idx * num_primitives * 6,
            dirs[idx],
            v_colors + idx * 3,

            v_c0 + idx * 3,
            v_coeffs + idx * num_primitives * 6
        );
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> compute_sb_bwd_tensor(
        const uint32_t num_primitives,
        const uint32_t active_primitives,
        const torch::Tensor &dirs,               // [..., 3]
        const torch::Tensor &c0,                 // [..., 3]
        const torch::Tensor &coeffs,             // [..., num_primitives, 6]
        const at::optional<torch::Tensor> masks, // [...]
        const torch::Tensor &v_colors,           // [..., 3]
        bool compute_v_dirs
    ) {
        GSPLAT_DEVICE_GUARD(dirs);
        GSPLAT_CHECK_INPUT(dirs);
        GSPLAT_CHECK_INPUT(c0);
        GSPLAT_CHECK_INPUT(coeffs);
        GSPLAT_CHECK_INPUT(v_colors);
        if (masks.has_value()) {
            GSPLAT_CHECK_INPUT(masks.value());
        }
        TORCH_CHECK(v_colors.size(-1) == 3, "v_colors must have last dimension 3");
        TORCH_CHECK(c0.size(-1) == 3, "c0 must have last dimension 3");
        TORCH_CHECK(coeffs.size(-1) == 6, "coeffs must have last dimension 6");
        TORCH_CHECK(dirs.size(-1) == 3, "dirs must have last dimension 3");
        const uint32_t N = dirs.numel() / 3;
       
        torch::Tensor v_coeffs = torch::zeros_like(coeffs);
        torch::Tensor v_c0 = torch::zeros_like(c0);
        torch::Tensor v_dirs;
        if (compute_v_dirs) {
            v_dirs = torch::zeros_like(dirs);
        }
        if (N) {
            compute_sh_bwd_kernel<float>
                <<<(N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                GSPLAT_N_THREADS>>>(
                    N,
                    num_primitives,
                    active_primitives,
                    reinterpret_cast<vec3<float> *>(dirs.data_ptr<float>()),
                    c0.data_ptr<float>(),
                    coeffs.data_ptr<float>(),
                    masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
                    v_colors.data_ptr<float>(),
                    v_c0.data_ptr<float>(),
                    v_coeffs.data_ptr<float>(),
                    compute_v_dirs ? v_dirs.data_ptr<float>() : nullptr
            );
        }

        return std::make_tuple(v_c0, v_coeffs, v_dirs); // [..., num_primitives, 6], [..., 3], [..., 3]
    }
}