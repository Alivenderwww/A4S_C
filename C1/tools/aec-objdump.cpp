// aec-objdump.cpp - Disassemble an .aecbin back to readable AEC assembly.
//
//   aec-objdump output.aecbin
//
// Reads the container (binfmt reader), then prints the header summary, the
// decoded instruction stream (with labels), and the relocation/symbol tables.
#include "aec/driver.h"
#include "aec/binfmt.h"

#include <cstdio>
#include <string>

using namespace aec;

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s output.aecbin\n", argv[0]);
    return 2;
  }

  binfmt::Image image;
  std::string err;
  if (!binfmt::readFile(argv[1], image, err)) {
    std::fprintf(stderr, "aec-objdump: %s\n", err.c_str());
    return 1;
  }

  std::string asmText = disassemble(image);
  std::fwrite(asmText.data(), 1, asmText.size(), stdout);
  return 0;
}
