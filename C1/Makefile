# Makefile - AEC C1 compiler (aec-cc) + disassembler (aec-objdump).
#
# Pinned to -std=c++11: the source is written and tested against C++11 (this dev
# host has g++ 4.9.2), and every modern toolchain -- including the grader's GCC
# 13.3 -- accepts -std=c++11. Pinning it (rather than opting into c++17) keeps
# the build on the exact standard the code was compiled against, avoiding any
# reliance on never-tested C++17 behavior. -finput-charset=UTF-8 makes the UTF-8
# source comments portable across locales. See README.md ("构建").

CXX ?= g++
CXXSTD := c++11

CXXFLAGS ?= -std=$(CXXSTD) -finput-charset=UTF-8 -O2 -Wall -Wextra -Iinclude -MMD -MP
LDFLAGS  ?=

# Windows (MinGW) appends .exe; keep both hosts happy.
ifeq ($(OS),Windows_NT)
EXE := .exe
else
EXE :=
endif

LIB_SRC := $(wildcard src/*.cpp) $(wildcard src/*/*.cpp)
LIB_OBJ := $(patsubst src/%.cpp,obj/%.o,$(LIB_SRC))

CC_OBJ := obj/tools/aec-cc.o
OD_OBJ := obj/tools/aec-objdump.o

CC_BIN := bin/aec-cc$(EXE)
OD_BIN := bin/aec-objdump$(EXE)

.PHONY: all build submit test clean selftest
all: build submit

build: $(CC_BIN) $(OD_BIN)
	@echo "built with -std=$(CXXSTD): $(CC_BIN) $(OD_BIN)"

# Place the required submission entry point (spec §14) at the path the grader
# invokes: compiler/aec-cc. disassembler/aec-objdump is our companion tool.
submit: build
	@mkdir -p compiler disassembler
	@cp $(CC_BIN) compiler/aec-cc$(EXE)
	@cp $(OD_BIN) disassembler/aec-objdump$(EXE)
	@echo "submission entry points ready: compiler/aec-cc  disassembler/aec-objdump"

$(CC_BIN): $(LIB_OBJ) $(CC_OBJ)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(OD_BIN): $(LIB_OBJ) $(OD_OBJ)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

obj/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

obj/tools/%.o: tools/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run the encoder golden self-test after building.
selftest: $(CC_BIN)
	./$(CC_BIN) --selftest && echo "selftest OK"

# Compile + disassemble every public PTX test.
test: build
	bash tests/run_public.sh

clean:
	rm -rf obj bin

-include $(LIB_OBJ:.o=.d) $(CC_OBJ:.o=.d) $(OD_OBJ:.o=.d)
