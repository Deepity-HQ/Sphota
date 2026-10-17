#pragma once
/**
 * @file key_ring.hpp
 * @brief Private FIFO of past contrastive keys with a built-in delay line.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace deepity::local
{

/**
 * @brief Ring buffer of the last `queue + delay` keys, owned by exactly one thread.
 *
 * The newest `delay` keys are withheld: only keys older than that are valid negatives, so frames
 * of the scene being looked at are never their own negatives. Keys are stored twice (at `i` and
 * `i + capacity`), so any window of negatives is one contiguous block and the similarity to all
 * negatives is a single matrix-vector product with no wrap-around handling.
 *
 * Not thread-safe by design: the queue belongs to one layer, so it needs neither locks nor atomics.
 *
 * @tparam T Scalar type (float or double).
 */
template <class T> class KeyRing
{
public:
  /**
   * @param dim   Key dimension.
   * @param queue Maximum number of negatives returned.
   * @param delay Number of newest keys withheld from the negatives.
   */
  KeyRing(int dim, int queue, int delay)
      : dim_(dim)
      , queue_(queue)
      , delay_(delay)
      , cap_(static_cast<size_t>(queue) + delay)
      , buf_(2 * cap_ * static_cast<size_t>(dim), T(0))
  {
  }

  /** @brief Appends a key, overwriting the oldest one once the ring is full. */
  void Push(const T* key)
  {
    const size_t idx = static_cast<size_t>(count_ % cap_);
    std::copy(key, key + dim_, &buf_[idx * dim_]);
    std::copy(key, key + dim_, &buf_[(idx + cap_) * dim_]);
    ++count_;
  }

  /**
   * @brief Returns the current negatives as one contiguous block.
   * @param[out] out Pointer to `n * dim` values, oldest first (null if `n == 0`).
   * @return Number of negatives `n`.
   */
  int Negatives(const T** out) const
  {
    const uint64_t valid = count_ > static_cast<uint64_t>(delay_) ? count_ - delay_ : 0;
    const int n = static_cast<int>(std::min<uint64_t>(valid, static_cast<uint64_t>(queue_)));
    if (n == 0)
    {
      *out = nullptr;
      return 0;
    }
    const uint64_t first = valid - n;
    *out = &buf_[static_cast<size_t>(first % cap_) * dim_];
    return n;
  }

  /** @brief Total number of keys ever pushed. */
  uint64_t Count() const
  {
    return count_;
  }

private:
  int dim_, queue_, delay_;
  size_t cap_;
  uint64_t count_ = 0;
  std::vector<T> buf_;
};

} // namespace deepity::local
