// lexer.cpp - Tokenizer for the PTX subset.
//
// PTX is line/statement oriented. We emit a flat token stream and let the
// parser assemble statements (terminated by ';', '{', '}' or a ':' label).
// Comments (// and /* */) are stripped here.
#include "ptx_lexer.h"

#include <cctype>

namespace aec {
namespace ptx {

static bool isIdentChar(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '%' ||
         c == '.' || c == '$' || c == '<' || c == '>';
}

std::vector<Token> tokenize(const std::string &src) {
  std::vector<Token> toks;
  int line = 1;
  size_t i = 0;
  const size_t n = src.size();

  while (i < n) {
    char c = src[i];

    if (c == '\n') { ++line; ++i; continue; }
    if (std::isspace(static_cast<unsigned char>(c))) { ++i; continue; }

    // Line comment.
    if (c == '/' && i + 1 < n && src[i + 1] == '/') {
      while (i < n && src[i] != '\n') ++i;
      continue;
    }
    // Block comment.
    if (c == '/' && i + 1 < n && src[i + 1] == '*') {
      i += 2;
      while (i + 1 < n && !(src[i] == '*' && src[i + 1] == '/')) {
        if (src[i] == '\n') ++line;
        ++i;
      }
      i += 2;
      continue;
    }

    // Single-character punctuation that matters to the parser.
    if (c == ';' || c == ',' || c == '{' || c == '}' || c == '(' ||
        c == ')' || c == '[' || c == ']' || c == ':' || c == '@') {
      Token t;
      t.kind = Token::Punct;
      t.text = std::string(1, c);
      t.line = line;
      toks.push_back(t);
      ++i;
      continue;
    }

    // Identifier / number / directive / register token.
    if (isIdentChar(c) || c == '-' || c == '!') {
      size_t start = i;
      // '!' only leads a token (guard negation); consume it then the ident.
      if (c == '!') { ++i; }
      while (i < n && isIdentChar(src[i])) ++i;
      Token t;
      t.kind = Token::Word;
      t.text = src.substr(start, i - start);
      t.line = line;
      toks.push_back(t);
      continue;
    }

    // Anything else: skip one char (defensive).
    ++i;
  }

  Token eof;
  eof.kind = Token::End;
  eof.line = line;
  toks.push_back(eof);
  return toks;
}

} // namespace ptx
} // namespace aec
