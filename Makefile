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

TESTS := tests/test_arena tests/test_tensor
SAN_TESTS := $(TESTS:tests/%=tests/%_asan)
EXAMPLES := examples/marco1_basics

.PHONY: all lib tests test asan valgrind examples run_example clean

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

run_example: examples
	./examples/marco1_basics

tests: $(TESTS)

test: tests
	./tests/test_arena
	./tests/test_tensor

asan: $(SAN_TESTS)
	ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 ./tests/test_arena_asan
	ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 ./tests/test_tensor_asan

valgrind: tests
	@if command -v valgrind >/dev/null 2>&1; then \
		valgrind --error-exitcode=42 --leak-check=full --show-leak-kinds=all ./tests/test_arena && \
		valgrind --error-exitcode=42 --leak-check=full --show-leak-kinds=all ./tests/test_tensor; \
	else \
		echo "valgrind não instalado; use 'make asan'."; \
	fi

clean:
	rm -f $(OBJS) $(LIB) $(TESTS) $(SAN_TESTS) $(EXAMPLES)
