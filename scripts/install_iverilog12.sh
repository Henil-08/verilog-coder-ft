#!/usr/bin/env bash
# VerilogEval's testbenches need Icarus Verilog v12 (v13 rejects them).
# Builds v12 into $PREFIX without touching any system iverilog.
# macOS deps: brew install autoconf bison flex gperf
set -euo pipefail
PREFIX="${PREFIX:-$HOME/.local/opt/iverilog-12}"
SRC="$(mktemp -d)"
git clone --depth 1 --branch v12-branch https://github.com/steveicarus/iverilog.git "$SRC/iverilog"
cd "$SRC/iverilog"
# v12's driver calls _NSGetExecutablePath without including its header (fails on recent macOS clang)
if [[ "$(uname)" == "Darwin" ]] && ! grep -q "mach-o/dyld.h" driver/main.c; then
  printf '#ifdef __APPLE__\n#include <mach-o/dyld.h>\n#endif\n' | cat - driver/main.c > driver/main.c.new
  mv driver/main.c.new driver/main.c
fi
# macOS ships an ancient bison; prefer Homebrew's if present
if command -v brew >/dev/null; then export PATH="$(brew --prefix bison)/bin:$PATH"; fi
sh autoconf.sh
./configure --prefix="$PREFIX"
make -j"$(getconf _NPROCESSORS_ONLN)"
make install
echo "installed: $PREFIX/bin/iverilog ($("$PREFIX/bin/iverilog" -V 2>&1 | head -1))"
echo "export IVERILOG_BIN_DIR=$PREFIX/bin"
