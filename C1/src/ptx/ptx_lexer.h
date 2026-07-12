// ptx_lexer.h - Private token API shared by lexer.cpp and parser.cpp.
#ifndef AEC_PTX_LEXER_H
#define AEC_PTX_LEXER_H

#include <string>
#include <vector>

namespace aec {
namespace ptx {

struct Token {
  enum Kind { Word, Punct, End };
  Kind kind = End;
  std::string text;
  int line = 0;
};

// Split PTX source text into tokens (comments stripped).
std::vector<Token> tokenize(const std::string &src);

} // namespace ptx
} // namespace aec

#endif // AEC_PTX_LEXER_H
