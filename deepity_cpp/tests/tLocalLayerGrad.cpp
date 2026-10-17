/**
 * @file tLocalLayerGrad.cpp
 * @brief Finite-difference check of the hand-derived LocalLayer gradient (double precision).
 */

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "deepity/local/local_layer.hpp"

using namespace deepity::local;

namespace
{
std::vector<double> Unit(std::mt19937_64& rng, int n)
{
  std::normal_distribution<double> g;
  std::vector<double> v(n);
  double s = 0;
  for (auto& e : v)
  {
    e = g(rng);
    s += e * e;
  }
  for (auto& e : v)
    e /= std::sqrt(s);
  return v;
}
} // namespace

int main()
{
  LocalConfig<double> cfg;
  cfg.nIn = 20, cfg.nOut = 12, cfg.proj = 8, cfg.queue = 10, cfg.delay = 3, cfg.tau = 0.3,
  cfg.seed = 7;
  LocalLayer<double> layer(cfg);
  std::mt19937_64 rng(11);
  std::vector<double> x(cfg.nIn, 0.0);
  std::uniform_real_distribution<double> u(0.1, 1.0);
  for (int i = 0; i < cfg.nIn; ++i)
    if (i % 3 != 0)
      x[i] = u(rng); // sparse input, like the real frames
  const auto kpos = Unit(rng, cfg.proj);
  const int nNeg = 7;
  std::vector<double> negs;
  for (int j = 0; j < nNeg; ++j)
  {
    const auto k = Unit(rng, cfg.proj);
    negs.insert(negs.end(), k.begin(), k.end());
  }

  const double loss = layer.LossGrad(x.data(), kpos.data(), negs.data(), nNeg);
  std::vector<double> da(layer.Da(), layer.Da() + cfg.nOut), dp(layer.Dp(), layer.Dp() + cfg.proj),
      h(layer.H(), layer.H() + cfg.nOut);
  auto& net = layer.Online();

  const double eps = 1e-6;
  double worst = 0, worstAbs = 0, maxGrad = 0;
  int checked = 0, significant = 0;
  double sabotage = 1.0; // 1.0 = honest gradient; the control run uses 1.05
  auto check = [&](std::vector<double>& param, size_t idx, double analytic)
  {
    analytic *= sabotage;
    const double keep = param[idx];
    param[idx] = keep + eps;
    const double lp = layer.LossGrad(x.data(), kpos.data(), negs.data(), nNeg);
    param[idx] = keep - eps;
    const double lm = layer.LossGrad(x.data(), kpos.data(), negs.data(), nNeg);
    param[idx] = keep;
    const double numeric = (lp - lm) / (2 * eps);
    const double err =
        std::fabs(numeric - analytic) / std::max(1e-7, std::fabs(numeric) + std::fabs(analytic));
    worst = std::max(worst, std::fabs(numeric - analytic) < 1e-9 ? 0.0 : err);
    worstAbs = std::max(worstAbs, std::fabs(numeric - analytic));
    maxGrad = std::max(maxGrad, std::fabs(numeric));
    significant += std::fabs(numeric) > 1e-4;
    ++checked;
  };
  auto sweep = [&]
  {
    for (int i = 0; i < cfg.nIn; ++i)
      for (int j = 0; j < cfg.nOut; ++j)
        check(net.w, static_cast<size_t>(i) * cfg.nOut + j, x[i] * da[j]);
    for (int j = 0; j < cfg.nOut; ++j)
      check(net.b, j, da[j]);
    for (int j = 0; j < cfg.nOut; ++j)
      for (int k = 0; k < cfg.proj; ++k)
        check(net.pt, static_cast<size_t>(j) * cfg.proj + k, h[j] * dp[k]);
    for (int k = 0; k < cfg.proj; ++k)
      check(net.c, k, dp[k]);
  };
  sweep();
  const double honestWorst = worst;
  const int honestSig = significant;
  const double honestAbs = worstAbs, honestMax = maxGrad;
  worst = 0, sabotage = 1.05, sweep(); // control: a 5% wrong gradient must be caught
  const double sabotagedWorst = worst;

  std::printf(
      "loss %.6f; %d entries checked per sweep, %d with |grad| > 1e-4 (largest |grad| %.3f)\n",
      loss,
      checked / 2,
      honestSig,
      honestMax);
  std::printf("honest gradient: worst relative error %.2e, worst absolute error %.2e\n",
              honestWorst,
              honestAbs);
  std::printf("control (gradient scaled by 1.05): worst relative error %.2e (must be detected)\n",
              sabotagedWorst);
  const bool ok = honestWorst < 1e-5 && honestSig > 50 && sabotagedWorst > 1e-3;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
