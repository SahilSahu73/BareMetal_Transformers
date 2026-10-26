#!/usr/bin/env bash
# Disassemble an object file / archive and fail if it contains a data-path multiplier.
#
# Forbidden (always): floating-point multiply / fused multiply-add, SIMD integer multiplies and
# multiply-accumulates (x86 pmul*/pmadd*, ARM mul/mla/smull/umull/sdot/... on vector registers).
# Tolerated and merely counted: SCALAR integer multiplies (x86 `imul`; AArch64 `mul`/`madd`/`msub`/
# `smull`... on x/w registers). Compilers emit those for array-index and loop-partition arithmetic
# (row * row_bytes); they are address generation, not weight x activation work, and neither ISA has a
# separate instruction for it. The distinction on AArch64 is the register class: a `v` register
# operand (e.g. `v0.4s`) means NEON data-path.
#
#   scripts/check_multiplierless.sh <object-or-archive> [objdump-command]
# Exit status 0: clean, 1: multiplier instruction found, 2: usage / tool error.
set -u
obj="${1:-}"
objdump_cmd="${2:-${OBJDUMP:-objdump}}"
[ -n "$obj" ] && [ -e "$obj" ] || { echo "usage: $0 <object-or-archive> [objdump]" >&2; exit 2; }
command -v "$objdump_cmd" >/dev/null 2>&1 || { echo "objdump not found: $objdump_cmd" >&2; exit 2; }

disassembly="$("$objdump_cmd" -d --no-show-raw-insn "$obj" 2>/dev/null)" || { echo "objdump failed on $obj" >&2; exit 2; }
[ -n "$disassembly" ] || { echo "no instructions found in $obj" >&2; exit 2; }

printf '%s\n' "$disassembly" | awk -F'\t' '
  function is_vector(ops) { return ops ~ /(^|[ ,\[])v[0-9]+\./ }
  NF >= 2 {
    split($2, a, " "); m = a[1]
    ops = substr($2, length(m) + 1) (NF >= 3 ? " " $3 : "")
    total++
    if (m ~ /^v?(fmul|fmla|fmls|fmadd|fmsub|fnmadd|fnmsub|fnmul|mulss|mulsd|mulps|mulpd)/ ||
        m ~ /^v?pmul[a-z0-9]*$/ || m ~ /^v?pmadd[a-z0-9]*$/ || m ~ /^vfn?m(add|sub)[0-9a-z]*$/ ||
        m ~ /^(mulx|sdot|udot|usdot|sqdmulh|sqrdmulh|pmull2?|smmla|ummla|usmmla|bfmmla)$/ ||
        m ~ /^fmul/) {
      bad[m]++; nbad++
    } else if (m == "mul" && ops ~ /%/) {            # one-operand x86 mul
      bad[m]++; nbad++
    } else if (m == "imul") {
      scalar++
    } else if (m ~ /^(mul|mla|mls|madd|msub|mneg|smull2?|umull2?|smlal2?|umlal2?|smlsl2?|umlsl2?|smaddl|umaddl|smsubl|umsubl|smulh|umulh|smnegl|umnegl)$/) {
      if (is_vector(ops)) { bad[m " (vector)"]++; nbad++ } else scalar++
    } else if (m ~ /^v[a-z0-9]+$/ || m ~ /^(p(add|sub|and|or|xor|sad|shuf|cmp|abs|sign|alignr|unpck|ack)[a-z0-9]*|add[pv]?|sub|and|eor|orr|tbl|cmeq|saddlp|sadalp|uaddlp|uadalp|saddlv|addv|addp|sabd|uabd)$/) {
      simd[m] = 1
    }
  }
  END {
    printf "checked: %d instructions, scalar integer multiplies (index arithmetic): %d\n", total, scalar
    if (nbad) {
      print "FOUND multiplier-class instructions:"
      for (k in bad) printf "    %4d %s\n", bad[k], k
      exit 1
    }
    print "no data-path multiplier (no FP/SIMD multiply, FMA, pmadd, widening or vector integer multiply)"
    s = ""; for (k in simd) s = s " " k
    print "add/logic-class SIMD ops used:" s
  }'
exit $?   # the awk verdict: 0 clean, 1 multiplier found
