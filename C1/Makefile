# Makefile - AEC C1 compiler (aec-cc) + disassembler (aec-objdump).
#
# The eval image ships GCC 13.3 (C++17); this dev host has g++ 4.9.2 which does
# NOT accept -std=c++17. We therefore auto-detect the newest -std the compiler
# accepts (c++17 -> else c++11). The source is written C++11-safe so it builds
# under both. See README.md ("构建") for the Windows/WSL note.

CXX ?= g++

# Pick c++17 when available, else fall back to c++11 (g++ 4.9.2).
CXXSTD := $(shell echo 'int main(){return 0;}' | $(CXX) -std=c++17 -fsyntax-only -x c++ - 2>/dev/null && echo c++17 || echo c++11)

CXXFLAGS ?= -std=$(CXXSTD) -O2 -Wall -Wextra -Iinclude -MMD -MP
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

# Place the required submission entry points at the paths the grader invokes:
#   compiler/aec-cc  disassembler/aec-objdump  agent/run_agent
submit: build
	@mkdir -p compiler disassembler
	@cp $(CC_BIN) compiler/aec-cc$(EXE)
	@cp $(OD_BIN) disassembler/aec-objdump$(EXE)
	@chmod +x agent/run_agent 2>/dev/null || true
	@echo "submission entry points ready: compiler/aec-cc  disassembler/aec-objdump  agent/run_agent"

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
