#ifndef VIBECODER_POLY1305_HPP
#define VIBECODER_POLY1305_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>

// Streaming Poly1305 (RFC 8439, section 2.5).
// A key must be used for only one message. Call finalize() exactly once.
// AEAD padding and length encoding are the caller's responsibility.
class Poly1305 {
  uint32_t r[5], h[5] = {}, pad[4];
  uint8_t buffer[16] = {};
  size_t buffered = 0;

  static uint32_t le32(const uint8_t *p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
           (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
  }

  static void store_le32(uint8_t *p, uint32_t value) {
    for (unsigned i = 0; i < 4; ++i)
      p[i] = uint8_t(value >> (8 * i));
  }

  // Full blocks have the implicit 2^128 bit. The last partial block
  // instead has a byte containing 1 appended by finalize().
  void block(const uint8_t msg[16], uint32_t high_bit) {
    h[0] += le32(msg) & 0x3ffffff;
    h[1] += (le32(msg + 3) >> 2) & 0x3ffffff;
    h[2] += (le32(msg + 6) >> 4) & 0x3ffffff;
    h[3] += (le32(msg + 9) >> 6) & 0x3ffffff;
    h[4] += (le32(msg + 12) >> 8) | high_bit;

    const uint32_t s1 = r[1] * 5, s2 = r[2] * 5;
    const uint32_t s3 = r[3] * 5, s4 = r[4] * 5;
    uint64_t d0 = uint64_t(h[0])*r[0] + uint64_t(h[1])*s4 +
                  uint64_t(h[2])*s3 + uint64_t(h[3])*s2 + uint64_t(h[4])*s1;
    uint64_t d1 = uint64_t(h[0])*r[1] + uint64_t(h[1])*r[0] +
                  uint64_t(h[2])*s4 + uint64_t(h[3])*s3 + uint64_t(h[4])*s2;
    uint64_t d2 = uint64_t(h[0])*r[2] + uint64_t(h[1])*r[1] +
                  uint64_t(h[2])*r[0] + uint64_t(h[3])*s4 + uint64_t(h[4])*s3;
    uint64_t d3 = uint64_t(h[0])*r[3] + uint64_t(h[1])*r[2] +
                  uint64_t(h[2])*r[1] + uint64_t(h[3])*r[0] + uint64_t(h[4])*s4;
    uint64_t d4 = uint64_t(h[0])*r[4] + uint64_t(h[1])*r[3] +
                  uint64_t(h[2])*r[2] + uint64_t(h[3])*r[1] + uint64_t(h[4])*r[0];

    uint32_t c = uint32_t(d0 >> 26);
    h[0] = uint32_t(d0) & 0x3ffffff; d1 += c;
    c = uint32_t(d1 >> 26);
    h[1] = uint32_t(d1) & 0x3ffffff; d2 += c;
    c = uint32_t(d2 >> 26);
    h[2] = uint32_t(d2) & 0x3ffffff; d3 += c;
    c = uint32_t(d3 >> 26);
    h[3] = uint32_t(d3) & 0x3ffffff; d4 += c;
    c = uint32_t(d4 >> 26);
    h[4] = uint32_t(d4) & 0x3ffffff;
    h[0] += c * 5;
    c = h[0] >> 26;
    h[0] &= 0x3ffffff;
    h[1] += c;
  }

public:
  explicit Poly1305(const uint8_t key[32]) {
    // Decode r into 26-bit limbs and apply the RFC's byte-level clamp.
    r[0] = le32(key) & 0x3ffffff;
    r[1] = (le32(key + 3) >> 2) & 0x3ffff03;
    r[2] = (le32(key + 6) >> 4) & 0x3ffc0ff;
    r[3] = (le32(key + 9) >> 6) & 0x3f03fff;
    r[4] = (le32(key + 12) >> 8) & 0x00fffff;
    for (unsigned i = 0; i < 4; ++i) pad[i] = le32(key + 16 + 4*i);
  }

  void update(const uint8_t *msg, size_t len) {
    if (len == 0) return;
    if (buffered != 0) {
      const size_t want = len < 16 - buffered ? len : 16 - buffered;
      std::memcpy(buffer + buffered, msg, want);
      buffered += want; msg += want; len -= want;
      if (buffered != 16) return;
      block(buffer, 1u << 24);
      buffered = 0;
    }
    while (len >= 16) {
      block(msg, 1u << 24);
      msg += 16;
      len -= 16;
    }
    if (len != 0) {
      std::memcpy(buffer, msg, len);
      buffered = len;
    }
  }

  // Append AAD and its explicit zero padding, not a Poly1305 terminator.
  void update_aad(const uint8_t *aad, size_t len) {
    update(aad, len);
    const size_t padding = (16 - len % 16) % 16;
    if (padding != 0) {
      const uint8_t zeros[16] = {};
      update(zeros, padding);
    }
  }

  void finalize(uint8_t tag[16]) {
    if (buffered != 0) {
      buffer[buffered] = 1;
      std::memset(buffer + buffered + 1, 0, 15 - buffered);
      block(buffer, 0);
      buffered = 0;
    }

    // Fully carry h, then select h or h - (2^130 - 5).
    uint32_t c = h[1] >> 26;
    h[1] &= 0x3ffffff; h[2] += c;
    c = h[2] >> 26; h[2] &= 0x3ffffff; h[3] += c;
    c = h[3] >> 26; h[3] &= 0x3ffffff; h[4] += c;
    c = h[4] >> 26; h[4] &= 0x3ffffff; h[0] += c * 5;
    c = h[0] >> 26; h[0] &= 0x3ffffff; h[1] += c;
    uint32_t g[5];
    g[0] = h[0] + 5; c = g[0] >> 26; g[0] &= 0x3ffffff;
    for (unsigned i = 1; i < 4; ++i) {
      g[i] = h[i] + c; c = g[i] >> 26; g[i] &= 0x3ffffff;
    }
    g[4] = h[4] + c - (1u << 26);
    const uint32_t mask = (g[4] >> 31) - 1;
    for (unsigned i = 0; i < 5; ++i) h[i] = (h[i] & ~mask) | (g[i] & mask);

    // Truncate to 128 bits BEFORE adding s; carry only the additions.
    const uint32_t words[4] = {
      h[0] | (h[1] << 26),
      (h[1] >> 6) | (h[2] << 20),
      (h[2] >> 12) | (h[3] << 14),
      (h[3] >> 18) | (h[4] << 8)
    };
    uint64_t carry = 0;
    for (unsigned i = 0; i < 4; ++i) {
      const uint64_t sum = uint64_t(words[i]) + pad[i] + carry;
      store_le32(tag + 4*i, uint32_t(sum));
      carry = sum >> 32;
    }
  }
};

#endif
