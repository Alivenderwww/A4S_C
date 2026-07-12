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
  out = Image();
  if (bytes.size() < kHeaderBytes) { err = "file smaller than header"; return false; }

  uint32_t magic = 0, version = 0, headerBytes = 0, sectionCount = 0;
  getU32(bytes, 0, magic);
  getU32(bytes, 4, version);
  getU32(bytes, 8, headerBytes);
  getU32(bytes, 12, sectionCount);
  if (magic != kMagic) { err = "bad magic (not an .aecbin)"; return false; }
  if (version != kVersion) { err = "unsupported .aecbin version"; return false; }

  getU32(bytes, 16, out.header.entryPC);
  getU32(bytes, 20, out.header.instructionCount);
  getU32(bytes, 24, out.header.paramBytes);
  getU32(bytes, 28, out.header.flags);
  out.header.magic = magic;
  out.header.version = version;
  out.header.headerBytes = headerBytes;
  out.header.sectionCount = sectionCount;

  size_t tablePos = headerBytes ? headerBytes : kHeaderBytes;
  for (uint32_t s = 0; s < sectionCount; ++s) {
    size_t base = tablePos + (size_t)s * kSectionEntryBytes;
    uint32_t type = 0, offset = 0, size = 0, entSize = 0;
    if (!getU32(bytes, base, type) || !getU32(bytes, base + 4, offset) ||
        !getU32(bytes, base + 8, size) || !getU32(bytes, base + 12, entSize)) {
      err = "truncated section table";
      return false;
    }
    if ((size_t)offset + (size_t)size > bytes.size()) {
      err = "section payload out of bounds";
      return false;
    }

    if (type == SEC_CODE) {
      if (size % 16 != 0) { err = "code section not a multiple of 16 bytes"; return false; }
      uint32_t count = size / 16;
      out.code.resize(count);
      for (uint32_t i = 0; i < count; ++i) {
        size_t p = (size_t)offset + (size_t)i * 16;
        getU32(bytes, p + 0, out.code[i].word0);
        getU32(bytes, p + 4, out.code[i].word1);
        getU32(bytes, p + 8, out.code[i].word2);
        getU32(bytes, p + 12, out.code[i].word3);
      }
    } else if (type == SEC_DATA) {
      out.data.assign(bytes.begin() + offset, bytes.begin() + offset + size);
    } else if (type == SEC_RELOC) {
      uint32_t count = 0;
      if (size >= 4) getU32(bytes, offset, count);
      for (uint32_t i = 0; i < count; ++i) {
        size_t p = (size_t)offset + 4 + (size_t)i * 16;
        RelocEntry r;
        if (!getU32(bytes, p + 0, r.instrIndex) || !getU32(bytes, p + 4, r.kind) ||
            !getU32(bytes, p + 8, r.addend) || !getU32(bytes, p + 12, r.reserved)) {
          err = "truncated relocation section";
          return false;
        }
        out.relocs.push_back(r);
      }
    } else if (type == SEC_SYMBOL) {
      uint32_t count = 0;
      size_t p = offset;
      if (size >= 4) { getU32(bytes, p, count); p += 4; }
      for (uint32_t i = 0; i < count; ++i) {
        uint32_t nameLen = 0;
        if (!getU32(bytes, p, nameLen)) { err = "truncated symbol table"; return false; }
        p += 4;
        if (p + nameLen + 8 > bytes.size()) { err = "truncated symbol entry"; return false; }
        SymbolEntry sym;
        sym.name.assign(bytes.begin() + p, bytes.begin() + p + nameLen);
        p += nameLen;
        getU32(bytes, p, sym.value); p += 4;
        getU32(bytes, p, sym.kind);  p += 4;
        out.symbols.push_back(sym);
      }
    }
    // Unknown section types are ignored (forward-compatible).
  }

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
