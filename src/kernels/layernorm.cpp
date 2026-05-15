#include "esm_cpp/kernels.h"

#include <cmath>

namespace esm::kernels {

void LayerNormRef(const float* x, const float* gamma, const float* beta,
                  float eps, float* out, int num_rows, int d) {
  for (int r = 0; r < num_rows; ++r) {
    const float* xr = x + static_cast<long>(r) * d;
    float* yr = out + static_cast<long>(r) * d;
    double mean = 0.0;
    for (int i = 0; i < d; ++i) mean += xr[i];
    mean /= d;
    double var = 0.0;
    for (int i = 0; i < d; ++i) {
      double diff = xr[i] - mean;
      var += diff * diff;
    }
    var /= d;
    float inv_std = 1.0f / std::sqrt(static_cast<float>(var) + eps);
    for (int i = 0; i < d; ++i) {
      float normed = (xr[i] - static_cast<float>(mean)) * inv_std;
      yr[i] = normed * gamma[i] + beta[i];
    }
  }
}

}  // namespace esm::kernels
