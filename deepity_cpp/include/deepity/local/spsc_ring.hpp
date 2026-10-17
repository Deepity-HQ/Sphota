#pragma once
/**
 * @file spsc_ring.hpp
 * @brief Wait-free single-producer single-consumer ring for passing activations between layers.
 */

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace deepity::local
{

/**
 * @brief Bounded wait-free SPSC queue; the producer never blocks.
 *
 * One thread calls TryPush, one thread calls TryPop. When the ring is full the producer drops the
 * new item and counts it (a layer below must never wait for a slower layer above). Slots are
 * preallocated, so the hot path performs no allocation. Head and tail live on separate cache
 * lines, and each side keeps a private cache of the other side's index so that the shared line is
 * only read when the cached value says full/empty.
 *
 * Memory ordering: the producer publishes with a release store, the consumer observes with an
 * acquire load; nothing stronger is needed.
 *
 * @tparam Slot Trivially copyable payload (e.g. a frame id plus an activation vector).
 * @tparam Cap  Capacity, a power of two.
 */
template <class Slot, size_t Cap> class SpscRing
{
  static_assert(Cap >= 2 && (Cap & (Cap - 1)) == 0, "capacity must be a power of two");
  static constexpr size_t kLine = 64;

public:
  SpscRing()
      : slots_(new Slot[Cap])
  {
  }

  /**
   * @brief Producer side. Copies `s` into the ring.
   * @return false (and counts a drop) if the ring is full.
   */
  bool TryPush(const Slot& s)
  {
    const uint64_t h = head_.load(std::memory_order_relaxed);
    if (h - cachedTail_ == Cap)
    {
      cachedTail_ = tail_.load(std::memory_order_acquire);
      if (h - cachedTail_ == Cap)
      {
        ++dropped_;
        return false;
      }
    }
    slots_[h & (Cap - 1)] = s;
    head_.store(h + 1, std::memory_order_release);
    return true;
  }

  /**
   * @brief Consumer side. Moves the oldest item into `out`.
   * @return false if the ring is empty.
   */
  bool TryPop(Slot& out)
  {
    const uint64_t t = tail_.load(std::memory_order_relaxed);
    if (t == cachedHead_)
    {
      cachedHead_ = head_.load(std::memory_order_acquire);
      if (t == cachedHead_)
        return false;
    }
    out = slots_[t & (Cap - 1)];
    tail_.store(t + 1, std::memory_order_release);
    return true;
  }

  /** @brief Items the producer dropped; read it only after the producer has stopped. */
  uint64_t Dropped() const
  {
    return dropped_;
  }

private:
  alignas(kLine) std::atomic<uint64_t> head_{0};
  alignas(kLine) std::atomic<uint64_t> tail_{0};
  alignas(kLine) uint64_t cachedTail_ = 0; ///< producer-private
  uint64_t dropped_ = 0;                   ///< producer-private
  alignas(kLine) uint64_t cachedHead_ = 0; ///< consumer-private
  alignas(kLine) std::unique_ptr<Slot[]> slots_;
};

} // namespace deepity::local
