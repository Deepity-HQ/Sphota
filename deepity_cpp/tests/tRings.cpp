/**
 * @file tRings.cpp
 * @brief Correctness of KeyRing (window contents) and SpscRing (ordering, no loss, two threads).
 */

#include <atomic>
#include <cstdio>
#include <thread>

#include "deepity/local/key_ring.hpp"
#include "deepity/local/spsc_ring.hpp"

using namespace deepity::local;

namespace
{
bool TestKeyRing()
{
  const int dim = 2, queue = 5, delay = 3;
  KeyRing<float> ring(dim, queue, delay);
  for (int n = 0; n <= 60; ++n)
  {
    const float* neg = nullptr;
    const int got = ring.Negatives(&neg);
    const int valid = n > delay ? n - delay : 0;
    const int want = valid < queue ? valid : queue;
    if (got != want)
      return std::printf("n=%d: %d negatives, want %d\n", n, got, want), false;
    for (int k = 0; k < got; ++k)
    {
      const float expect =
          static_cast<float>(valid - got + k); // oldest first, newest `delay` withheld
      if (neg[k * dim] != expect || neg[k * dim + 1] != expect)
        return std::printf("n=%d slot %d: %g, want %g\n", n, k, neg[k * dim], expect), false;
    }
    const float key[2] = {static_cast<float>(n), static_cast<float>(n)};
    ring.Push(key);
  }
  return true;
}

struct Slot
{
  uint64_t id;
  float v[16];
};

bool TestSpsc()
{
  constexpr uint64_t kN = 3'000'000;
  SpscRing<Slot, 1024> ring;
  std::atomic<bool> done{false};
  uint64_t received = 0, last = 0;
  bool ok = true, first = true;
  std::thread consumer(
      [&]
      {
        Slot s;
        while (true)
        {
          if (ring.TryPop(s))
          {
            if (!first && s.id <= last)
              ok = false; // strictly increasing, no reordering
            for (float f : s.v)
              ok = ok && f == static_cast<float>(s.id); // no torn slots
            first = false, last = s.id, ++received;
          }
          else if (done.load(std::memory_order_acquire))
          {
            if (!ring.TryPop(s))
              break;
            ++received, last = s.id;
          }
        }
      });
  std::thread producer(
      [&]
      {
        Slot s;
        for (uint64_t i = 0; i < kN; ++i)
        {
          s.id = i;
          for (float& f : s.v)
            f = static_cast<float>(i);
          ring.TryPush(s);
        }
        done.store(true, std::memory_order_release);
      });
  producer.join();
  consumer.join();
  std::printf("spsc: pushed %llu, received %llu, dropped %llu\n",
              (unsigned long long)kN,
              (unsigned long long)received,
              (unsigned long long)ring.Dropped());
  return ok && received + ring.Dropped() == kN;
}
} // namespace

int main()
{
  const bool a = TestKeyRing(), b = TestSpsc();
  std::printf("KeyRing %s, SpscRing %s\n", a ? "PASS" : "FAIL", b ? "PASS" : "FAIL");
  return a && b ? 0 : 1;
}
