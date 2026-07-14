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

.PHONY: all build submit test conformance clean selftest
all: build submit

build: $(CC_BIN) $(OD_BIN)
	@echo "built with -std=$(CXXSTD): $(CC_BIN) $(OD_BIN)"

# The submission entry point (spec §14) is compiler/aec-cc, a committed wrapper
# that build-on-first-use + execs bin/aec-cc, so it works with or without a
# grader pre-build. On Windows we also drop a native .exe next to it for local
# testing; on Linux we just keep the wrapper executable (never overwrite it).
submit: build
	@mkdir -p compiler disassembler
	@cp $(OD_BIN) disassembler/aec-objdump$(EXE)
ifeq ($(OS),Windows_NT)
	@cp $(CC_BIN) compiler/aec-cc.exe
else
	@chmod +x compiler/aec-cc
endif
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

# CModel conformance regression: divergence (partial-block bounds guards,
# `cond ? a : b` if-conversion) + ISA coverage (%laneid, SHL.u32, shr, bitwise,
# min/max) -- the scenarios the 5 public cases (verify.py) never exercise.
# Needs the x86-64 golden model (public/aec-cmodel-release); run on an x86 Linux
# host or under WSL.
conformance: build
	python3 sim/cmodel/conformance.py

clean:
	rm -rf obj bin

-include $(LIB_OBJ:.o=.d) $(CC_OBJ:.o=.d) $(OD_OBJ:.o=.d)
