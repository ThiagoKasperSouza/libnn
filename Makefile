CC      ?= cc
AR      ?= ar

# Flags obrigatórias: preservadas mesmo quando CFLAGS/CPPFLAGS vêm da
# linha de comando ou do ambiente (o `+=` comum é ignorado nesses casos).
NC_REQUIRED_CFLAGS := -std=c11 -Wall -Wextra -Wpedantic
CFLAGS  ?= -O2
override CFLAGS := $(NC_REQUIRED_CFLAGS) $(CFLAGS)
override CPPFLAGS += -Iinclude

SANFLAGS = -fsanitize=address,undefined -fno-omit-frame-pointer -g -O1

SRCS := $(wildcard src/*.c)
OBJS := $(SRCS:.c=.o)
HDRS := $(wildcard include/*.h)
LIB  := libnn.a

TESTS := tests/test_arena tests/test_tensor tests/test_autograd tests/test_zoo
SAN_TESTS := $(TESTS:tests/%=tests/%_asan)
EXAMPLES := examples/marco1_basics examples/xor_perceptron examples/dae_example

.PHONY: all lib tests test asan valgrind examples run_example run_examples clean

all: lib

lib: $(LIB)

$(LIB): $(OBJS)
	$(AR) rcs $@ $^

src/%.o: src/%.c $(HDRS)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

tests/test_%: tests/test_%.c $(LIB) $(HDRS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) -lm -o $@

tests/test_%_asan: tests/test_%.c $(SRCS) $(HDRS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SANFLAGS) $< $(SRCS) -lm -o $@

examples/%: examples/%.c $(LIB) $(HDRS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) -lm -o $@

examples: $(EXAMPLES)

run_example: run_examples

run_examples: $(EXAMPLES)
	@for e in $(EXAMPLES); do ./$$e || exit 1; done

tests: $(TESTS)

test: tests
	@for t in $(TESTS); do ./$$t || exit 1; done

asan: $(SAN_TESTS)
	@for t in $(SAN_TESTS); do \
		ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 ./$$t || exit 1; \
	done

valgrind: tests
	@if command -v valgrind >/dev/null 2>&1; then \
		for t in $(TESTS); do \
			valgrind --error-exitcode=42 --leak-check=full --show-leak-kinds=all ./$$t || exit 1; \
		done; \
	else \
		echo "valgrind não instalado; use 'make asan'."; \
	fi

clean:
	rm -f $(OBJS) $(LIB) $(TESTS) $(SAN_TESTS) $(EXAMPLES)
