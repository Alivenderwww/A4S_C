// aec-cc.cpp - Command-line front end for the AEC compiler.
//
//   aec-cc input.ptx -O2 -o output.aecbin
//
// Flags:
//   -o <file>            output path (default: <input>.aecbin)
//   -O0 | -O2 | -O3      optimization level (default -O2)
//   --report <file>      write a JSON perf report (consumed by the agent)
//   --no-<pass>          disable a single pass: const-prop|dce|cse|licm|
//                        mem-coalesce|pred-opt|dual-issue|gemm
//   --sched-window <n>   list-scheduler lookahead
//   --selftest           run the encoder golden self-test and exit
//   -v | --verbose       dump pipeline progress to stderr
//   -h | --help          usage
#include "aec/driver.h"
#include "aec/target.h"
#include "aec/isa.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace aec;

static void usage(const char *argv0) {
  std::printf(
    "usage: %s input.ptx [-O0|-O2|-O3] [-o out.aecbin] [--report r.json]\n"
    "          [--no-const-prop|--no-dce|--no-cse|--no-licm|--no-mem-coalesce\n"
    "           |--no-pred-opt|--no-dual-issue] [--sched-window N]\n"
    "          [--selftest] [-v|--verbose] [-h|--help]\n", argv0);
}

static std::string defaultOut(const std::string &in) {
  size_t dot = in.find_last_of('.');
  size_t slash = in.find_last_of("/\\");
  if (dot != std::string::npos && (slash == std::string::npos || dot > slash))
    return in.substr(0, dot) + ".aecbin";
  return in + ".aecbin";
}

int main(int argc, char **argv) {
  std::string input, output, reportPath;
  Options opt;
  opt.applyOptLevel(OptLevel::O2);

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
    else if (a == "--selftest") { return isa::selfTest() ? 0 : 1; }
    else if (a == "-v" || a == "--verbose") { opt.verbose = true; }
    else if (a == "--lenient") { opt.lenient = true; }
    else if (a == "-O0") { opt.applyOptLevel(OptLevel::O0); }
    else if (a == "-O2") { opt.applyOptLevel(OptLevel::O2); }
    else if (a == "-O3") { opt.applyOptLevel(OptLevel::O3); }
    else if (a == "-O1") { opt.applyOptLevel(OptLevel::O2); } // alias
    else if (a == "-o") { if (i + 1 < argc) output = argv[++i]; }
    else if (a == "--report") { if (i + 1 < argc) reportPath = argv[++i]; }
    else if (a == "--sched-window") { if (i + 1 < argc) opt.sched_window = std::atoi(argv[++i]); }
    else if (a == "--no-const-prop") { opt.const_prop = false; }
    else if (a == "--no-dce") { opt.dce = false; }
    else if (a == "--no-cse") { opt.cse = false; }
    else if (a == "--no-licm") { opt.licm = false; }
    else if (a == "--no-mem-coalesce") { opt.mem_coalesce = false; }
    else if (a == "--no-pred-opt") { opt.pred_opt = false; }
    else if (a == "--no-dual-issue") { opt.dual_issue = false; }
    else if (a == "--unroll") { opt.unroll = true; }
    else if (a == "--no-unroll") { opt.unroll = false; }
    else if (a == "--unroll-factor") { if (i + 1 < argc) opt.unroll_factor = std::atoi(argv[++i]); }
    else if (!a.empty() && a[0] == '-') {
      std::fprintf(stderr, "aec-cc: unknown option '%s'\n", a.c_str());
      usage(argv[0]);
      return 2;
    } else {
      input = a;
    }
  }

  if (input.empty()) { usage(argv[0]); return 2; }
  if (output.empty()) output = defaultOut(input);

  if (opt.verbose) isa::selfTest();

  CompileReport rep;
  std::string err;
  if (!compileFile(input, output, opt, rep, err)) {
    std::fprintf(stderr, "aec-cc: %s\n", err.c_str());
    return 1;
  }

  const char *olv = (opt.opt == OptLevel::O0) ? "O0"
                  : (opt.opt == OptLevel::O3) ? "O3" : "O2";
  char json[512];
  std::snprintf(json, sizeof(json),
      "{\"kernel\":\"%s\",\"opt\":\"%s\",\"instruction_count\":%u,"
      "\"spill_count\":%u,\"dual_issue_pairs\":%u,\"param_bytes\":%u,"
      "\"est_cycles\":%llu}\n",
      rep.kernel.c_str(), olv, rep.instructionCount, rep.spillCount,
      rep.dualIssuePairs, rep.paramBytes, (unsigned long long)rep.estCycles);

  if (!reportPath.empty()) {
    FILE *rf = std::fopen(reportPath.c_str(), "wb");
    if (rf) { std::fputs(json, rf); std::fclose(rf); }
    else std::fprintf(stderr, "aec-cc: warning: cannot write report %s\n", reportPath.c_str());
  }

  std::fprintf(stderr, "aec-cc: wrote %s  (%u instructions, %s)\n",
      output.c_str(), rep.instructionCount, olv);
  if (opt.verbose) std::fputs(json, stderr);
  return 0;
}
