// writer.cpp - Serialize an in-memory Image to the .aecbin container.
//
// Layout (all little-endian, see binfmt.h):
//   [FileHeader 32B][SectionEntry*4 16B each][CODE][DATA][RELOC][SYMBOL]
// We always emit the four sections CODE/DATA/RELOC/SYMBOL in that order so the
// reader and the C1 spec's required-section list are both satisfied.
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
  // --- Build each payload blob first. ---
  std::vector<uint8_t> code;
  for (unsigned i = 0; i < img.code.size(); ++i) {
    putU32(code, img.code[i].word0);
    putU32(code, img.code[i].word1);
    putU32(code, img.code[i].word2);
    putU32(code, img.code[i].word3);
  }

  const std::vector<uint8_t> &data = img.data;

  std::vector<uint8_t> reloc;
  putU32(reloc, (uint32_t)img.relocs.size());
  for (unsigned i = 0; i < img.relocs.size(); ++i) {
    putU32(reloc, img.relocs[i].instrIndex);
    putU32(reloc, img.relocs[i].kind);
    putU32(reloc, img.relocs[i].addend);
    putU32(reloc, img.relocs[i].reserved);
  }

  std::vector<uint8_t> symbol;
  putU32(symbol, (uint32_t)img.symbols.size());
  for (unsigned i = 0; i < img.symbols.size(); ++i) {
    const SymbolEntry &s = img.symbols[i];
    putU32(symbol, (uint32_t)s.name.size());
    for (unsigned c = 0; c < s.name.size(); ++c)
      symbol.push_back((uint8_t)s.name[c]);
    putU32(symbol, s.value);
    putU32(symbol, s.kind);
  }

  const uint32_t kSectionCount = 4;
  const uint32_t tableBytes = kSectionCount * kSectionEntryBytes;
  uint32_t off = kHeaderBytes + tableBytes;

  const uint32_t codeOff = off;   off += (uint32_t)code.size();
  const uint32_t dataOff = off;   off += (uint32_t)data.size();
  const uint32_t relocOff = off;  off += (uint32_t)reloc.size();
  const uint32_t symOff = off;    off += (uint32_t)symbol.size();

  // --- Header. ---
  std::vector<uint8_t> out;
  out.reserve(off);
  putU32(out, kMagic);
  putU32(out, kVersion);
  putU32(out, kHeaderBytes);
  putU32(out, kSectionCount);
  putU32(out, img.header.entryPC);
  putU32(out, (uint32_t)img.code.size());
  putU32(out, img.header.paramBytes);
  putU32(out, img.header.flags);

  // --- Section table. ---
  putU32(out, SEC_CODE);   putU32(out, codeOff);  putU32(out, (uint32_t)code.size());   putU32(out, 16u);
  putU32(out, SEC_DATA);   putU32(out, dataOff);  putU32(out, (uint32_t)data.size());   putU32(out, 0);
  putU32(out, SEC_RELOC);  putU32(out, relocOff); putU32(out, (uint32_t)reloc.size());  putU32(out, 16);
  putU32(out, SEC_SYMBOL); putU32(out, symOff);   putU32(out, (uint32_t)symbol.size()); putU32(out, 0);

  // --- Payloads. ---
  out.insert(out.end(), code.begin(), code.end());
  out.insert(out.end(), data.begin(), data.end());
  out.insert(out.end(), reloc.begin(), reloc.end());
  out.insert(out.end(), symbol.begin(), symbol.end());

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
