// reader.cpp - Parse a .aecbin byte buffer back into an Image.
//
// Mirrors writer.cpp. Every structural problem is reported through `err` with
// a short reason so aec-objdump can print a clean diagnostic instead of
// crashing on a malformed file.
#include "aec/binfmt.h"

#include <cstdio>

namespace aec {
namespace binfmt {

namespace {

bool getU32(const std::vector<uint8_t> &b, size_t off, uint32_t &out) {
  if (off + 4 > b.size()) return false;
  out = (uint32_t)b[off] | ((uint32_t)b[off + 1] << 8) |
        ((uint32_t)b[off + 2] << 16) | ((uint32_t)b[off + 3] << 24);
  return true;
}

} // namespace

bool read(const std::vector<uint8_t> &bytes, Image &out, std::string &err) {
  // C1 spec §10: .aecbin is a raw 128-bit instruction stream (no header). The
  // file size must be a non-zero multiple of 16 bytes; entry_pc is 0.
  out = Image();
  if (bytes.empty() || bytes.size() % 16 != 0) {
    err = "file size is not a non-zero multiple of 16 bytes";
    return false;
  }
  const uint32_t count = (uint32_t)(bytes.size() / 16);
  out.code.resize(count);
  for (uint32_t i = 0; i < count; ++i) {
    size_t p = (size_t)i * 16;
    getU32(bytes, p + 0, out.code[i].word0);
    getU32(bytes, p + 4, out.code[i].word1);
    getU32(bytes, p + 8, out.code[i].word2);
    getU32(bytes, p + 12, out.code[i].word3);
  }
  out.header.entryPC = 0;
  out.header.instructionCount = count;
  return true;
}

bool readFile(const std::string &path, Image &out, std::string &err) {
  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) { err = "cannot open file: " + path; return false; }
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (sz < 0) { std::fclose(f); err = "cannot size file"; return false; }
  std::vector<uint8_t> bytes((size_t)sz);
  size_t n = sz ? std::fread(&bytes[0], 1, (size_t)sz, f) : 0;
  std::fclose(f);
  if (n != (size_t)sz) { err = "short read"; return false; }
  return read(bytes, out, err);
}

} // namespace binfmt
} // namespace aec
