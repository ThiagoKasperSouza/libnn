#ifndef NC_TEST_COMMON_H
#define NC_TEST_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int nc_tests_run = 0;
static int nc_tests_failed = 0;

#define NC_CHECK(cond)                                                        \
    do {                                                                      \
        ++nc_tests_run;                                                       \
        if (!(cond)) {                                                        \
            fprintf(stderr, "FALHOU %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++nc_tests_failed;                                                \
        }                                                                     \
    } while (0)

#define NC_CHECK_NEAR(actual, expected, tolerance)                            \
    do {                                                                      \
        ++nc_tests_run;                                                       \
        const double nc_a_ = (double)(actual);                                \
        const double nc_e_ = (double)(expected);                              \
        if (fabs(nc_a_ - nc_e_) > (tolerance)) {                              \
            fprintf(stderr, "FALHOU %s:%d: %s (%.8f) != %s (%.8f)\n",        \
                    __FILE__, __LINE__, #actual, nc_a_, #expected, nc_e_);    \
            ++nc_tests_failed;                                                \
        }                                                                     \
    } while (0)

#define NC_TEST_SUMMARY(title)                                                \
    do {                                                                      \
        printf("%s: %d/%d verificações passaram\n", (title),                  \
               nc_tests_run - nc_tests_failed, nc_tests_run);                 \
        return nc_tests_failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;             \
    } while (0)

#endif /* NC_TEST_COMMON_H */
