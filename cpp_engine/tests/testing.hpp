// Tiny assertion harness (no gtest dependency). Exit code 0 = pass, 77 = skipped (CTest convention).
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>

static int g_failures = 0;

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      std::fprintf(stderr, "  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);     \
      ++g_failures;                                                              \
    }                                                                            \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                                            \
  do {                                                                                   \
    const double va = (a), vb = (b);                                                     \
    if (!(std::fabs(va - vb) <= (tol))) {                                                \
      std::fprintf(stderr, "  FAIL %s:%d  |%s - %s| = %g > %g\n", __FILE__, __LINE__, #a, #b, \
                   std::fabs(va - vb), static_cast<double>(tol));                        \
      ++g_failures;                                                                      \
    }                                                                                    \
  } while (0)

#define RUN(test)                          \
  do {                                     \
    const int before = g_failures;         \
    test();                                \
    std::printf("[%s] %s\n", g_failures == before ? " ok " : "FAIL", #test); \
  } while (0)

inline int finish() { return g_failures ? 1 : 0; }
