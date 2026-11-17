#include "../poly1305.hpp"
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using Bytes = std::vector<uint8_t>;
using Tag = std::array<uint8_t, 16>;

static Bytes hex(const std::string &s) {
  if (s.size() % 2) throw std::runtime_error("Odd hex length");
  Bytes result;
  for (size_t i = 0; i < s.size(); i += 2)
    result.push_back(uint8_t(std::stoul(s.substr(i, 2), nullptr, 16)));
  return result;
}

static void require_tag(const Tag &actual, const Bytes &expected,
                        const std::string &label) {
  if (expected.size() != actual.size() ||
      std::memcmp(actual.data(), expected.data(), actual.size()) != 0)
    throw std::runtime_error("FAIL: " + label);
}

// Exercise one-shot, every possible two-update split, byte-at-a-time,
// empty updates and unaligned inputs for every known-answer vector.
static void vector_test(const std::string &name, const Bytes &key,
                        const Bytes &msg, const Bytes &expected) {
  if (key.size() != 32 || expected.size() != 16)
    throw std::runtime_error("Bad vector dimensions");
  Bytes unaligned(msg.size() + 1);
  if (!msg.empty()) std::memcpy(unaligned.data() + 1, msg.data(), msg.size());
  const uint8_t *data = unaligned.data() + 1;
  Tag tag{};
  Poly1305 whole(key.data());
  whole.update(data, msg.size());
  whole.finalize(tag.data());
  require_tag(tag, expected, name + " one-shot");
  for (size_t split = 0; split <= msg.size(); ++split) {
    Poly1305 mac(key.data());
    mac.update(nullptr, 0);
    mac.update(data, split);
    mac.update(nullptr, 0);
    mac.update(data + split, msg.size() - split);
    mac.finalize(tag.data());
    require_tag(tag, expected, name + " split " + std::to_string(split));
  }
  Poly1305 bytes(key.data());
  for (size_t i = 0; i < msg.size(); ++i) bytes.update(data + i, 1);
  bytes.finalize(tag.data());
  require_tag(tag, expected, name + " bytewise");
  std::printf("PASS: %s (one-shot, all splits, bytewise)\n", name.c_str());
}

int main() {
  try {
    // Published vectors: https://www.rfc-editor.org/rfc/rfc8439
    const Bytes key = hex(
      "85d6be7857556d337f4452fe42d506a8"
      "0103808afb0db2fd4abff6af4149f51b");
    const std::string message = "Cryptographic Forum Research Group";
    vector_test("RFC 8439 section 2.5.2", key,
                Bytes(message.begin(), message.end()),
                hex("a8061dc1305136c6c22b8baf0c0127a9"));
    std::puts("RFC verified tag: a8061dc1305136c6c22b8baf0c0127a9");
    vector_test("RFC A.3 #1", Bytes(32), Bytes(64), Bytes(16));
    vector_test("RFC A.3 #5", hex(std::string("02") + std::string(62, '0')),
                Bytes(16, 0xff), hex("03000000000000000000000000000000"));
    vector_test("RFC A.3 #6",
                hex(std::string("02") + std::string(30, '0') + std::string(32, 'f')),
                hex("02000000000000000000000000000000"),
                hex("03000000000000000000000000000000"));
    vector_test("RFC A.3 #7", hex(std::string("01") + std::string(62, '0')),
                hex(std::string(32, 'f') + "f0" + std::string(30, 'f') +
                    "11000000000000000000000000000000"),
                hex("05000000000000000000000000000000"));
    vector_test("RFC A.3 #8", hex(std::string("01") + std::string(62, '0')),
                hex(std::string(32, 'f') +
                    "fbfefefefefefefefefefefefefefefe" +
                    "01010101010101010101010101010101"), Bytes(16));
    vector_test("RFC A.3 #9", hex(std::string("02") + std::string(62, '0')),
                hex("fd" + std::string(30, 'f')),
                hex("fa" + std::string(30, 'f')));
    const Bytes reduction_key = hex(
        "0100000000000000040000000000000000000000000000000000000000000000");
    const std::string reduction_data =
        "e33594d7505e43b90000000000000000"
        "3394d7505e4379cd0100000000000000"
        "00000000000000000000000000000000";
    vector_test("RFC A.3 #10", reduction_key,
                hex(reduction_data + "01000000000000000000000000000000"),
                hex("14000000000000005500000000000000"));
    vector_test("RFC A.3 #11", reduction_key, hex(reduction_data),
                hex("13000000000000000000000000000000"));
    // For an empty message the tag is the second half of the key.
    vector_test("Empty message", key, {},
                hex("0103808afb0db2fd4abff6af4149f51b"));

    // Regression for the app's 41-byte AAD submitted separately from padding.
    Bytes aad(41, 0xa5), padded = aad;
    padded.resize(48, 0);
    const Bytes payload(33, 0x5a);
    padded.insert(padded.end(), payload.begin(), payload.end());
    Tag actual{}, expected{};
    Poly1305 helper(key.data()), manual(key.data());
    helper.update_aad(aad.data(), aad.size());
    helper.update(payload.data(), payload.size());
    helper.finalize(actual.data());
    manual.update(padded.data(), padded.size());
    manual.finalize(expected.data());
    require_tag(actual, Bytes(expected.begin(), expected.end()), "AAD helper");
    std::puts("PASS: AAD helper equals explicitly zero-padded input");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
