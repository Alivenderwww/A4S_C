// writer.cpp - Serialize an in-memory Image to the .aecbin file.
//
// C1 spec §10: .aecbin is a RAW AEC 128-bit instruction stream -- no header,
// no data/relocation/symbol section. Each instruction is four little-endian
// 32-bit words written w0, w1, w2, w3. entry_pc is 0 and every label has been
// resolved to an absolute instruction index at compile time.
#include "aec/binfmt.h"

#include <cstdio>

namespace aec {
namespace binfmt {

namespace {

void putU32(std::vector<uint8_t> &v, uint32_t x) {
  v.push_back((uint8_t)(x & 0xff));
  v.push_back((uint8_t)((x >> 8) & 0xff));
  v.push_back((uint8_t)((x >> 16) & 0xff));
  v.push_back((uint8_t)((x >> 24) & 0xff));
}

} // namespace

std::vector<uint8_t> write(const Image &img) {
  std::vector<uint8_t> out;
  out.reserve(img.code.size() * 16);
  for (unsigned i = 0; i < img.code.size(); ++i) {
    putU32(out, img.code[i].word0);
    putU32(out, img.code[i].word1);
    putU32(out, img.code[i].word2);
    putU32(out, img.code[i].word3);
  }
  return out;
}

bool writeFile(const Image &img, const std::string &path) {
  std::vector<uint8_t> bytes = write(img);
  FILE *f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  size_t n = bytes.empty() ? 0 : std::fwrite(&bytes[0], 1, bytes.size(), f);
  std::fclose(f);
  return n == bytes.size();
}

} // namespace binfmt
} // namespace aec
