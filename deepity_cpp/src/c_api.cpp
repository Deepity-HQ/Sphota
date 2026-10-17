/**
 * @file c_api.cpp
 * @brief Plain C interface to LocalLayer<float> for ctypes / other-language callers.
 */

#include <algorithm>
#include <vector>

#include <xmmintrin.h>

#include "deepity/local/local_layer.hpp"

namespace
{
using Layer = deepity::local::LocalLayer<float>;
}

extern "C"
{
  /** @brief Creates a layer; `plain != 0` selects plain SGD instead of momentum SGD. */
  void* dl_create(int nIn, int nOut, int proj, int queue, int delay, int minNeg, int emaEvery,
                  float tau, float lr, float momentum, float ema, int plain,
                  unsigned long long seed)
  {
    Layer::Config c;
    c.nIn = nIn, c.nOut = nOut, c.proj = proj, c.queue = queue, c.delay = delay;
    c.minNegatives = minNeg, c.emaEvery = emaEvery, c.tau = tau, c.lr = lr;
    c.momentum = momentum, c.ema = ema, c.seed = seed;
    c.update = plain ? Layer::Config::Update::kPlain : Layer::Config::Update::kMomentum;
    return new Layer(c);
  }

  /**
   * @brief Flushes denormal floats to zero for the calling thread (MXCSR FTZ and DAZ).
   *
   * Momentum buffers of rows that stop receiving gradient decay geometrically into the denormal
   * range, where x86 arithmetic is an order of magnitude slower.
   */
  void dl_set_flush_denormals(int on)
  {
    const unsigned bits = 0x8040; // FTZ | DAZ
    _mm_setcsr(on ? (_mm_getcsr() | bits) : (_mm_getcsr() & ~bits));
  }

  /** @brief Copies the encoder weights (input-major `nIn x nOut`) and biases to the given buffers.
   */
  void dl_get_weights(void* layer, float* w, float* b)
  {
    const auto& n = static_cast<Layer*>(layer)->Online();
    std::copy(n.w.begin(), n.w.end(), w);
    std::copy(n.b.begin(), n.b.end(), b);
  }

  void dl_destroy(void* layer)
  {
    delete static_cast<Layer*>(layer);
  }

  /** @brief One frame through one layer; copies the layer output (nOut floats) to `out`. */
  float dl_step(void* layer, const float* x, int learn, float* out, int nOut)
  {
    auto* l = static_cast<Layer*>(layer);
    const float* h = l->Step(x, learn != 0);
    for (int i = 0; i < nOut; ++i)
      out[i] = h[i];
    return l->LastLoss();
  }

  /**
   * @brief Streams `n` frames through a two-layer front end, frame by frame.
   *
   * Layer 2 reads layer 1's output and learns only on frames where `(t0 + t) % every2 == 0`.
   * Returns the mean layer-1 loss over the chunk.
   */
  float dl_train_front(void* l1, void* l2, const float* frames, int n, int nIn, int every2,
                       long long t0)
  {
    auto* a = static_cast<Layer*>(l1);
    auto* b = static_cast<Layer*>(l2);
    double loss = 0;
    for (int t = 0; t < n; ++t)
    {
      const float* h1 = a->Step(frames + static_cast<size_t>(t) * nIn, true);
      b->Step(h1, (t0 + t) % every2 == 0);
      loss += a->LastLoss();
    }
    return n > 0 ? static_cast<float>(loss / n) : 0.f;
  }

  /** @brief Online-net activations `h` for `n` inputs (no learning, no state change). */
  void dl_encode(void* layer, const float* X, int n, float* H, int nIn, int nOut, int proj)
  {
    auto* l = static_cast<Layer*>(layer);
    std::vector<float> a(nOut), p(proj);
    for (int i = 0; i < n; ++i)
      l->Online().Forward(
          X + static_cast<size_t>(i) * nIn, a.data(), H + static_cast<size_t>(i) * nOut, p.data());
  }
}
