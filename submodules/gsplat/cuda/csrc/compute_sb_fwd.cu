#include "bindings.h"
#include "spherical_beta.cuh"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

    template <typename T>
    __global__ void compute_sh_fwd_kernel(
        const uint32_t N,
        const uint32_t num_primitives,
        const uint32_t active_primitives,
        const vec3<T> *__restrict__ dirs, // [N, 3]
        const T *__restrict__ c0,         // [3] base color
        const T *__restrict__ coeffs,     // [N, num_primitives, 6] [r, g, b, theta, phi, beta]
        const bool *__restrict__ masks,   // [N]
        T *__restrict__ colors            // [N, 3]
    ) {
        // parallelize over N * 3
        uint32_t idx = cg::this_grid().thread_rank();
        if (idx >= N) {
            return;
        }
        if (masks != nullptr && !masks[idx]) {
            return;
        }
        spherical_beta_isotropic_fwd(
            active_primitives,
            3,
            c0 + idx * 3,
            coeffs + idx * num_primitives * 6 ,
            dirs[idx],
            colors + idx * 3
        );
    }

    torch::Tensor compute_sb_fwd_tensor(
        const uint32_t active_primitives,
        const torch::Tensor &dirs,              // [..., 3]
        const torch::Tensor &c0,                // [3] base color
        const torch::Tensor &coeffs,            // [..., num_primitives, 6]
        const at::optional<torch::Tensor> masks // [...]
    ) {
        GSPLAT_DEVICE_GUARD(dirs);
        GSPLAT_CHECK_INPUT(dirs);
        GSPLAT_CHECK_INPUT(c0);
        GSPLAT_CHECK_INPUT(coeffs);
        if (masks.has_value()) {
            GSPLAT_CHECK_INPUT(masks.value());
        }
        TORCH_CHECK(coeffs.size(-1) == 6, "coeffs must have last dimension 6");
        TORCH_CHECK(dirs.size(-1) == 3, "dirs must have last dimension 3");
        TORCH_CHECK(c0.size(-1) == 3, "c0 must have last dimension 3");
        const uint32_t N = dirs.numel() / 3;
        const uint32_t K = coeffs.size(-2);
        torch::Tensor colors = torch::empty_like(dirs); // [..., 3]
        // parallelize over N * 3
        if (N) {
            compute_sh_fwd_kernel<float>
                <<<(N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                GSPLAT_N_THREADS>>>(
                    N,
                    K,
                    active_primitives,
                    reinterpret_cast<vec3<float> *>(dirs.data_ptr<float>()),
                    c0.data_ptr<float>(),
                    coeffs.data_ptr<float>(),
                    masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
                    colors.data_ptr<float>()
                );
        }

        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error in compute_sb_bwd_tensor: " << cudaGetErrorString(error) << std::endl;
            throw std::runtime_error("CUDA error in compute_sb_bwd_tensor " + std::string(cudaGetErrorString(error)));
        }

        return colors; // [..., 3]
    }

} // namespace gsplat
