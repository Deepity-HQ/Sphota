#pragma once
/**
 * @file local_layer.hpp
 * @brief One layer trained by its own InfoNCE loss on a continuous stream (no cross-layer
 * gradient).
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "key_ring.hpp"

namespace deepity::local
{

/** @brief y += a * x over `n` elements. */
template <class T> inline void Axpy(T a, const T* __restrict x, T* __restrict y, int n)
{
  for (int i = 0; i < n; ++i)
    y[i] += a * x[i];
}

/** @brief Dot product of two length-`n` vectors. */
template <class T> inline T Dot(const T* __restrict x, const T* __restrict y, int n)
{
  T s = 0;
#pragma omp simd reduction(+ : s)
  for (int i = 0; i < n; ++i)
    s += x[i] * y[i];
  return s;
}

/**
 * @brief Encoder `h = relu(x W + b)` and projection head `p = h PT + c`, stored input-major.
 *
 * Row `i` of `w` holds the weights leaving input `i`, so a forward pass over a sparse input is a
 * sum of a few contiguous rows, and the weight update touches only those rows.
 */
template <class T> struct Net
{
  int nIn = 0, nOut = 0, proj = 0;
  std::vector<T> w;  ///< [nIn][nOut]
  std::vector<T> b;  ///< [nOut]
  std::vector<T> pt; ///< [nOut][proj]
  std::vector<T> c;  ///< [proj]

  /** @brief Uniform(-1/sqrt(fanIn), 1/sqrt(fanIn)) init, as torch.nn.Linear. */
  void Init(int in, int out, int pr, std::mt19937_64& rng)
  {
    nIn = in, nOut = out, proj = pr;
    auto fill = [&](std::vector<T>& v, size_t n, int fanIn)
    {
      std::uniform_real_distribution<double> u(-1.0 / std::sqrt(fanIn), 1.0 / std::sqrt(fanIn));
      v.resize(n);
      for (auto& e : v)
        e = static_cast<T>(u(rng));
    };
    fill(w, static_cast<size_t>(in) * out, in);
    fill(b, out, in);
    fill(pt, static_cast<size_t>(out) * pr, out);
    fill(c, pr, out);
  }

  /** @brief Computes pre-activation `a`, activation `h` and projection `p` for input `x`. */
  void Forward(const T* x, T* a, T* h, T* p) const
  {
    std::copy(b.begin(), b.end(), a);
    for (int i = 0; i < nIn; ++i)
      if (x[i] != T(0))
        Axpy(x[i], &w[static_cast<size_t>(i) * nOut], a, nOut);
    std::copy(c.begin(), c.end(), p);
    for (int j = 0; j < nOut; ++j)
    {
      h[j] = a[j] > T(0) ? a[j] : T(0);
      if (h[j] != T(0))
        Axpy(h[j], &pt[static_cast<size_t>(j) * proj], p, proj);
    }
  }
};

/** @brief Layer hyperparameters. */
template <class T> struct LocalConfig
{
  /** @brief How weights are updated from the gradient. */
  enum class Update
  {
    kMomentum, ///< SGD with momentum buffer (dense pass; matches the Python reference)
    kPlain     ///< plain SGD (touches only the rows of nonzero inputs)
  };
  int nIn = 784, nOut = 512, proj = 128;
  int queue = 1024;      ///< maximum negatives
  int delay = 64;        ///< newest keys withheld from the negatives (scene-persistence timescale)
  int minNegatives = 64; ///< no learning before this many negatives exist
  int emaEvery = 1;      ///< momentum-encoder refresh interval (decay is raised to this power)
  T tau = T(0.3), lr = T(1e-3), momentum = T(0.9), ema = T(0.99);
  Update update = Update::kMomentum;
  uint64_t seed = 0;
};

/**
 * @brief A layer that learns by contrasting the current frame against the previous frame (positive)
 *        and a delayed FIFO of older frames (negatives), with a momentum copy producing the keys.
 *
 * The input is never differentiated, so no gradient leaves the layer. All state is private to the
 * layer's thread; only activations cross layer boundaries.
 *
 * @tparam T Scalar type (float for production, double for gradient tests).
 */
template <class T> class LocalLayer
{
public:
  using Config = LocalConfig<T>;

  /** @brief Builds the layer and its momentum copy from `cfg.seed`. */
  explicit LocalLayer(const Config& cfg)
      : cfg_(cfg)
      , ring_(cfg.proj, cfg.queue, cfg.delay)
      , a_(cfg.nOut)
      , h_(cfg.nOut)
      , p_(cfg.proj)
      , q_(cfg.proj)
      , dq_(cfg.proj)
      , dp_(cfg.proj)
      , da_(cfg.nOut)
      , logits_(cfg.queue + 1)
      , lastKey_(cfg.proj)
      , aK_(cfg.nOut)
      , hK_(cfg.nOut)
      , pK_(cfg.proj)
  {
    std::mt19937_64 rng(cfg.seed);
    online_.Init(cfg.nIn, cfg.nOut, cfg.proj, rng);
    target_ = online_;
    vw_.assign(online_.w.size(), T(0));
    vb_.assign(online_.b.size(), T(0));
    vpt_.assign(online_.pt.size(), T(0));
    vc_.assign(online_.c.size(), T(0));
  }

  /**
   * @brief Processes one frame: optionally takes a learning step, then stores the frame's key.
   * @param x     Input, length `nIn`.
   * @param learn Whether this step may update the weights.
   * @return The layer output `h` (length `nOut`) computed before the update; valid until the next
   * call.
   */
  const T* Step(const T* x, bool learn)
  {
    const T* negs = nullptr;
    const int nNeg = ring_.Negatives(&negs);
    if (learn && hasLast_ && nNeg >= cfg_.minNegatives)
    {
      lastLoss_ = LossGrad(x, lastKey_.data(), negs, nNeg);
      Apply(x);
      if (++sinceEma_ >= cfg_.emaEvery)
        UpdateTarget(), sinceEma_ = 0;
    }
    else
      online_.Forward(x, a_.data(), h_.data(), p_.data());
    Key(x, lastKey_.data());
    ring_.Push(lastKey_.data());
    hasLast_ = true;
    return h_.data();
  }

  /**
   * @brief InfoNCE loss and its gradient with respect to the pre-activation and projection.
   *
   * Fills `H()`, `Da()` and `Dp()`; the full weight gradients are `dW = x (x) da`, `db = da`,
   * `dPT = h (x) dp`, `dc = dp`. Does not modify any weight.
   *
   * @param x    Input frame.
   * @param kpos Key of the previous frame (unit norm).
   * @param negs `nNeg` contiguous negative keys (unit norm), each of length `proj`.
   * @return The loss `-log softmax(q.[kpos, negs] / tau)[0]`.
   */
  T LossGrad(const T* x, const T* kpos, const T* negs, int nNeg)
  {
    const int pr = cfg_.proj, no = cfg_.nOut;
    online_.Forward(x, a_.data(), h_.data(), p_.data());
    const T pnorm = std::sqrt(Dot(p_.data(), p_.data(), pr));
    const T nrm = pnorm + T(1e-8);
    for (int k = 0; k < pr; ++k)
      q_[k] = p_[k] / nrm;

    logits_[0] = Dot(q_.data(), kpos, pr) / cfg_.tau;
    T mx = logits_[0];
    for (int j = 0; j < nNeg; ++j)
    {
      logits_[j + 1] = Dot(q_.data(), negs + static_cast<size_t>(j) * pr, pr) / cfg_.tau;
      mx = std::max(mx, logits_[j + 1]);
    }
    T z = 0;
    for (int j = 0; j <= nNeg; ++j)
      z += (logits_[j] = std::exp(logits_[j] - mx));
    const T loss = -std::log(logits_[0] / z);

    // dq = ((pi0 - 1) kpos + sum_j pi_j neg_j) / tau
    const T pi0 = logits_[0] / z;
    for (int k = 0; k < pr; ++k)
      dq_[k] = (pi0 - T(1)) * kpos[k];
    for (int j = 0; j < nNeg; ++j)
      Axpy(logits_[j + 1] / z, negs + static_cast<size_t>(j) * pr, dq_.data(), pr);
    for (int k = 0; k < pr; ++k)
      dq_[k] /= cfg_.tau;

    // through q = p / (|p| + eps)
    const T coef = pnorm > T(0) ? Dot(p_.data(), dq_.data(), pr) / (nrm * nrm * pnorm) : T(0);
    for (int k = 0; k < pr; ++k)
      dp_[k] = dq_[k] / nrm - p_[k] * coef;

    // through h = relu(a), p = h PT + c
    for (int j = 0; j < no; ++j)
      da_[j] = a_[j] > T(0) ? Dot(&online_.pt[static_cast<size_t>(j) * pr], dp_.data(), pr) : T(0);
    return loss;
  }

  Net<T>& Online()
  {
    return online_;
  }
  const T* H() const
  {
    return h_.data();
  }
  const T* Da() const
  {
    return da_.data();
  }
  const T* Dp() const
  {
    return dp_.data();
  }
  T LastLoss() const
  {
    return lastLoss_;
  }
  const KeyRing<T>& Ring() const
  {
    return ring_;
  }

  /** @brief Writes the unit-norm key of `x` from the momentum copy into `out` (length `proj`). */
  void Key(const T* x, T* out)
  {
    target_.Forward(x, aK_.data(), hK_.data(), pK_.data());
    const T nrm = std::sqrt(Dot(pK_.data(), pK_.data(), cfg_.proj)) + T(1e-8);
    for (int k = 0; k < cfg_.proj; ++k)
      out[k] = pK_[k] / nrm;
  }

private:
  /** @brief Applies the stored gradient (from LossGrad) to the online weights. */
  void Apply(const T* x)
  {
    const int ni = cfg_.nIn, no = cfg_.nOut, pr = cfg_.proj;
    const T lr = cfg_.lr, mu = cfg_.momentum;
    T* w = online_.w.data();
    T* pt = online_.pt.data();
    if (cfg_.update == Config::Update::kPlain)
    {
      for (int i = 0; i < ni; ++i)
        if (x[i] != T(0))
          Axpy(-lr * x[i], da_.data(), w + static_cast<size_t>(i) * no, no);
      for (int j = 0; j < no; ++j)
        if (h_[j] != T(0))
          Axpy(-lr * h_[j], dp_.data(), pt + static_cast<size_t>(j) * pr, pr);
      Axpy(-lr, da_.data(), online_.b.data(), no);
      Axpy(-lr, dp_.data(), online_.c.data(), pr);
      return;
    }
    // v = mu v + g ; theta -= lr v   (one fused pass; rows without gradient only decay)
    auto row = [&](T* th, T* v, int n, T gScale, const T* g)
    {
      if (g != nullptr)
        for (int k = 0; k < n; ++k)
          v[k] = mu * v[k] + gScale * g[k], th[k] -= lr * v[k];
      else
        for (int k = 0; k < n; ++k)
          v[k] *= mu, th[k] -= lr * v[k];
    };
    for (int i = 0; i < ni; ++i)
      row(w + static_cast<size_t>(i) * no,
          vw_.data() + static_cast<size_t>(i) * no,
          no,
          x[i],
          x[i] != T(0) ? da_.data() : nullptr);
    for (int j = 0; j < no; ++j)
      row(pt + static_cast<size_t>(j) * pr,
          vpt_.data() + static_cast<size_t>(j) * pr,
          pr,
          h_[j],
          h_[j] != T(0) ? dp_.data() : nullptr);
    row(online_.b.data(), vb_.data(), no, T(1), da_.data());
    row(online_.c.data(), vc_.data(), pr, T(1), dp_.data());
  }

  /** @brief Momentum copy follows the online net: target = m^k target + (1 - m^k) online. */
  void UpdateTarget()
  {
    const T m = std::pow(cfg_.ema, static_cast<T>(cfg_.emaEvery));
    auto mix = [&](std::vector<T>& t, const std::vector<T>& o)
    {
      for (size_t k = 0; k < t.size(); ++k)
        t[k] = m * t[k] + (T(1) - m) * o[k];
    };
    mix(target_.w, online_.w), mix(target_.b, online_.b), mix(target_.pt, online_.pt),
        mix(target_.c, online_.c);
  }

  Config cfg_;
  Net<T> online_, target_;
  KeyRing<T> ring_;
  std::vector<T> a_, h_, p_, q_, dq_, dp_, da_, logits_, lastKey_, aK_, hK_, pK_;
  std::vector<T> vw_, vb_, vpt_, vc_;
  T lastLoss_ = 0;
  bool hasLast_ = false;
  int sinceEma_ = 0;
};

} // namespace deepity::local
