#ifndef GSPLAT_SPHERICAL_GAUSSIAN_CUH
#define GSPLAT_SPHERICAL_GAUSSIAN_CUH

#include "bindings.h"
#include "types.cuh"
#include "utils.cuh"

namespace gsplat {
    /**
     * @brief The dot product formulation of spherical gaussians:
     *
     * C = c0 + ∑ᵢ cᵢ dot(μᵢ, v)^(4 * exp(bᵢ))
     *
     *
     * TOTAL PARAMETERS: 3 + 6N, where N is the number of primitives.
     *
     * @param num_primitives
     * @param num_primitives
     * @param num_colors
     * @param dir
     * @param coeffs
     * @param colors
     */
    template <typename T>
    __forceinline__ __device__ void spherical_beta_isotropic_fwd(
        const uint32_t num_primitives, // degree of SH to be evaluated
        const uint32_t num_colors,      // color channel
        const T* c0,           // [3] base color
        const T* primitives,   // [num_primitives, 6]
                               // Each primitive has: r, g, b, theta, phi, beta
        const vec3<T> &dir,    // [3]
        // output
        T *colors // [3]
    ){
        colors[0] = c0[0];
        colors[1] = c0[1];
        colors[2] = c0[2];

        // Normalize the direction vector
        T inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
        vec3<T> dir_norm = {
            dir.x * inorm,
            dir.y * inorm,
            dir.z * inorm
        };

        for (uint32_t i = 0; i < num_primitives; i++){  
            // convert theta and phi to x, y, z
            T r = primitives[i * 6 + 0];
            T g = primitives[i * 6 + 1];
            T b = primitives[i * 6 + 2];
            T theta = primitives[i * 6 + 3];
            T phi = primitives[i * 6 + 4];
            T beta = primitives[i * 6 + 5];
            

            // theta range [0, pi], phi range [0, 2pi]
            vec3<T> dir_beta_mean = {
                sin(theta) * cos(phi),
                sin(theta) * sin(phi),
                cos(theta)
            };

            // Compute the dot product between dir and dir_beta_mean
            T dot = dir_norm.x * dir_beta_mean.x + dir_norm.y * dir_beta_mean.y + dir_norm.z * dir_beta_mean.z;

            // Compute the beta term
            T betaTerm = 0.0f;
            if (dot > 0)
                betaTerm = __powf(dot, 4 * __expf(beta));

            // compute the color
            colors[0] += betaTerm * r;
            colors[1] += betaTerm * g;
            colors[2] += betaTerm * b;
        }
    }

    template <typename T>
    __forceinline__ __device__ void spherical_beta_isotropic_bwd(
        const uint32_t num_primitives, // Number of primitives to evaluate
        const uint32_t num_colors,     // Number of color channels

        const T* c0,                   // [3] Base color: [r, g, b]
        const T* primitives,           // [num_primitives, 6]
                                       // Each primitive has: r, g, b, theta, phi, beta
        const vec3<T> &dir,            // [3] Direction vector

        // Gradient input (w.r.t. output colors)
        const T* v_color_in,           // [3] Gradients: [dL/dr, dL/dg, dL/db]

        // Gradient outputs (w.r.t. input parameters)
        T* v_c0,                       // [3] Gradients for base color
        T* v_primitives                // [num_primitives, 6] Gradients for primitives
    ){
        // Initialize gradients for base color c0
        v_c0[0] = v_color_in[0];
        v_c0[1] = v_color_in[1];
        v_c0[2] = v_color_in[2];

        // Normalize the direction vector
        T inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
        vec3<T> dir_norm = {
            dir.x * inorm,
            dir.y * inorm,
            dir.z * inorm
        };

        // Iterate over each primitive
        for (uint32_t i = 0; i < num_primitives; i++){
            // Extract primitive parameters
            T r = primitives[i * 6 + 0];
            T g = primitives[i * 6 + 1];
            T b = primitives[i * 6 + 2];
            T theta = primitives[i * 6 + 3];
            T phi = primitives[i * 6 + 4];
            T beta = primitives[i * 6 + 5];

            // Compute sine and cosine values for theta and phi
            T sin_theta = sin(theta);
            T cos_theta = cos(theta);
            T sin_phi = sin(phi);
            T cos_phi = cos(phi);

            // Compute direction vector of primitive
            vec3<T> dir_beta_mean = {
                sin_theta * cos_phi,
                sin_theta * sin_phi,
                cos_theta
            };

            // Compute the dot product between dir_norm and dir_beta_mean
            T dot = dir_norm.x * dir_beta_mean.x + dir_norm.y * dir_beta_mean.y + dir_norm.z * dir_beta_mean.z;

            // Compute the beta term
            T betaTerm = 0.0f;
            if (dot > 0)
                betaTerm = __powf(dot, 4 * __expf(beta));

            // Compute gradients w.r.t. color contributions (r, g, b)
            // dL/dr_i = v_color_in[0] * betaTerm
            // dL/dg_i = v_color_in[1] * betaTerm
            // dL/db_i = v_color_in[2] * betaTerm
            v_primitives[i * 6 + 0] = v_color_in[0] * betaTerm; // grad_r
            v_primitives[i * 6 + 1] = v_color_in[1] * betaTerm; // grad_g
            v_primitives[i * 6 + 2] = v_color_in[2] * betaTerm; // grad_b

            // Initialize gradients for theta, phi, beta
            T grad_theta = 0.0f;
            T grad_phi = 0.0f;
            T grad_beta = 0.0f;

            // Proceed only if dot > 0 to avoid invalid gradients
            if (dot > 0.0f){
                // Compute exponent = 4 * exp(beta)
                T exp_beta = __expf(beta);
                T exponent = 4.0f * exp_beta;

                // Compute the common gradient factor: (v_color_in[0] * r + v_color_in[1] * g + v_color_in[2] * b)
                T color_grad = v_color_in[0] * r + v_color_in[1] * g + v_color_in[2] * b;

                // Compute gradients w.r.t. theta and phi
                // d(dot)/d(theta_i) = dir_norm_x * cos(theta_i) * cos(phi_i) + dir_norm_y * cos(theta_i) * sin(phi_i) - dir_norm_z * sin(theta_i)
                T d_dot_d_theta = dir_norm.x * (cos_theta * cos_phi) + dir_norm.y * (cos_theta * sin_phi) - dir_norm.z * sin_theta;

                // d(dot)/d(phi_i) = -dir_norm_x * sin(theta_i) * sin(phi_i) + dir_norm_y * sin(theta_i) * cos(phi_i)
                T d_dot_d_phi = -dir_norm.x * sin_theta * sin_phi + dir_norm.y * sin_theta * cos_phi;

                // Compute d(betaTerm)/d(theta_i) = exponent * dot^{exponent -1} * d_dot_d_theta
                T d_betaTerm_d_theta = exponent * __powf(dot, exponent -1) * d_dot_d_theta;

                // Compute d(betaTerm)/d(phi_i) = exponent * dot^{exponent -1} * d_dot_d_phi
                T d_betaTerm_d_phi = exponent * __powf(dot, exponent -1) * d_dot_d_phi;

                // Compute d(betaTerm)/d(beta_i) = 4 * exp(beta) * dot^{exponent} * ln(dot)
                T ln_dot = logf(dot);
                T d_betaTerm_d_beta = exponent * __powf(dot, exponent) * ln_dot;

                // Compute gradients w.r.t. theta, phi, beta
                grad_theta = color_grad * d_betaTerm_d_theta;
                grad_phi = color_grad * d_betaTerm_d_phi;
                grad_beta = color_grad * d_betaTerm_d_beta;
            }

            // Assign gradients to primitives
            v_primitives[i * 6 + 3] = grad_theta; // grad_theta
            v_primitives[i * 6 + 4] = grad_phi;   // grad_phi
            v_primitives[i * 6 + 5] = grad_beta;  // grad_beta
        }
    }
}
#endif