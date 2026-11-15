#include "../poly1305.hpp"

// Test-only C ABI for comparison with an independent big-integer reference.
extern "C" void poly1305_test_tag(const uint8_t *key, const uint8_t *message,
                                size_t length, size_t stride, uint8_t *tag) {
  Poly1305 mac(key);
  size_t offset = 0;
  while (offset < length) {
    const size_t count = stride < length - offset ? stride : length - offset;
    mac.update(message + offset, count);
    offset += count;
  }
  mac.finalize(tag);
}
