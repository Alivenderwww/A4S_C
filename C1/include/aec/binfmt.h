// binfmt.h - The ".aecbin" container format (our own design).
//
// The C1 spec only mandates that the binary *contain* a Header, Code, Data,
// Relocation and Symbol section, with Code being 128-bit AEC instructions.
// It does not fix the container layout, so we define a small, explicit,
// endian-safe format here. Everything is serialized little-endian regardless
// of host byte order (see writer.cpp / reader.cpp).
//
//   [ FileHeader                 ]  (32 bytes)
//   [ SectionEntry * sectionCount]  (16 bytes each)
//   [ CODE   payload ] 16 * instructionCount bytes
//   [ DATA   payload ] raw bytes (param block / constants)
//   [ RELOC  payload ] RelocEntry array
//   [ SYMBOL payload ] u32 count + SymbolEntry array
#ifndef AEC_BINFMT_H
#define AEC_BINFMT_H

#include <cstdint>
#include <string>
#include <vector>

#include "aec/isa.h"

namespace aec {
namespace binfmt {

static const uint32_t kMagic   = 0x31434541u; // 'A','E','C','1' little-endian.
static const uint32_t kVersion = 1u;
static const uint32_t kHeaderBytes = 32u;
static const uint32_t kSectionEntryBytes = 16u;

enum SectionType : uint32_t {
  SEC_CODE   = 1,
  SEC_DATA   = 2,
  SEC_RELOC  = 3,
  SEC_SYMBOL = 4
};

// Relocation kinds. Only param-block references are used by the scaffold; the
// list is here so the C1 owner can grow it without touching the reader.
enum RelocKind : uint32_t {
  RELOC_NONE       = 0,
  RELOC_PARAM_ADDR = 1  // instruction imm holds a param-block byte offset.
};

struct FileHeader {
  uint32_t magic = kMagic;
  uint32_t version = kVersion;
  uint32_t headerBytes = kHeaderBytes;
  uint32_t sectionCount = 0;
  uint32_t entryPC = 0;            // instruction index of the kernel entry.
  uint32_t instructionCount = 0;
  uint32_t paramBytes = 0;         // size of the param block (in DATA).
  uint32_t flags = 0;
};

struct SectionEntry {
  uint32_t type = 0;               // SectionType.
  uint32_t offset = 0;             // byte offset from file start.
  uint32_t size = 0;               // payload size in bytes.
  uint32_t entSize = 0;            // fixed entry size (16 for CODE), else 0.
};

struct RelocEntry {
  uint32_t instrIndex = 0;         // which instruction to patch.
  uint32_t kind = RELOC_NONE;      // RelocKind.
  uint32_t addend = 0;             // param offset / symbol index.
  uint32_t reserved = 0;
};

struct SymbolEntry {
  std::string name;                // kernel/label name.
  uint32_t value = 0;              // instruction index / address.
  uint32_t kind = 0;               // 0 = kernel entry, 1 = label.
};

// In-memory image assembled by the writer / returned by the reader.
struct Image {
  FileHeader header;
  std::vector<isa::Word128> code;
  std::vector<uint8_t>      data;
  std::vector<RelocEntry>   relocs;
  std::vector<SymbolEntry>  symbols;
};

// Serialize an image to a .aecbin byte buffer.
std::vector<uint8_t> write(const Image &img);

// Serialize + write directly to a file. Returns false on I/O error.
bool writeFile(const Image &img, const std::string &path);

// Parse a .aecbin byte buffer. Returns false (with a message in err) on any
// structural problem so callers can report a clean error.
bool read(const std::vector<uint8_t> &bytes, Image &out, std::string &err);
bool readFile(const std::string &path, Image &out, std::string &err);

} // namespace binfmt
} // namespace aec

#endif // AEC_BINFMT_H
