/**
 * @file bStep.cpp
 * @brief Single-thread cost of one LocalLayer::Step on realistic sparse frames (19% nonzero).
 */

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "deepity/local/local_layer.hpp"

using namespace deepity::local;

namespace
{
double Run(const char* name, LocalConfig<float> cfg, const std::vector<float>& frames, int nFrames,
           int steps)
{
  LocalLayer<float> layer(cfg);
  for (int t = 0; t < 1500; ++t)
    layer.Step(&frames[(t % nFrames) * (size_t)cfg.nIn], false); // fill the FIFO
  for (int t = 0; t < 2000; ++t)
    layer.Step(&frames[(t % nFrames) * (size_t)cfg.nIn], true); // warm up
  const auto t0 = std::chrono::steady_clock::now();
  for (int t = 0; t < steps; ++t)
    layer.Step(&frames[(t % nFrames) * (size_t)cfg.nIn], true);
  const double us =
      std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() /
      steps;
  std::printf("  %-44s %7.1f us/step   (loss %.3f)\n", name, us, layer.LastLoss());
  return us;
}
} // namespace

int main()
{
  const int nFrames = 4096, steps = 20000;
  std::mt19937_64 rng(1);
  std::uniform_real_distribution<float> u(0.f, 1.f);
  auto frames = [&](int dim, double density)
  {
    std::vector<float> f(static_cast<size_t>(nFrames) * dim, 0.f);
    for (auto& e : f)
      e = u(rng) < density ? u(rng) : 0.f;
    return f;
  };

  std::printf("layer 1 (784 -> 512, 19%% dense input), queue 1024, proj 128\n");
  LocalConfig<float> c;
  const auto f1 = frames(784, 0.19);
  const double base = Run("momentum SGD, EMA every step (Python-equiv.)", c, f1, nFrames, steps);
  c.emaEvery = 8;
  Run("momentum SGD, EMA every 8 steps", c, f1, nFrames, steps);
  c.emaEvery = 1, c.update = LocalConfig<float>::Update::kPlain, c.lr = 0.01f;
  Run("plain SGD (lr/(1-mu)), EMA every step", c, f1, nFrames, steps);
  c.emaEvery = 8;
  const double fast = Run("plain SGD, EMA every 8 steps", c, f1, nFrames, steps);

  std::printf("layer 2 (512 -> 256, 50%% dense input)\n");
  LocalConfig<float> d;
  d.nIn = 512, d.nOut = 256;
  const auto f2 = frames(512, 0.5);
  const double base2 = Run("momentum SGD, EMA every step", d, f2, nFrames, steps);
  d.emaEvery = 8, d.update = LocalConfig<float>::Update::kPlain, d.lr = 0.01f;
  const double fast2 = Run("plain SGD, EMA every 8 steps", d, f2, nFrames, steps);

  std::printf("\nfront end per frame (L1 + L2 every other frame): %.0f us (Python-equiv.) vs %.0f "
              "us (plain/lazy)\n"
              "Python reference: ~3700 us/frame (3 workers) to ~4400-5900 us/frame (5 workers)\n",
              base + 0.5 * base2,
              fast + 0.5 * fast2);
  return 0;
}
