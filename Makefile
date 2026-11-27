CC      ?= cc
AR      ?= ar

# Flags obrigatórias: preservadas mesmo quando CFLAGS/CPPFLAGS vêm da
# linha de comando ou do ambiente (o `+=` comum é ignorado nesses casos).
NC_REQUIRED_CFLAGS := -std=c11 -Wall -Wextra -Wpedantic
CFLAGS  ?= -O2
override CFLAGS := $(NC_REQUIRED_CFLAGS) $(CFLAGS)
override CPPFLAGS += -Iinclude

SANFLAGS = -fsanitize=address,undefined -fno-omit-frame-pointer -g -O1
COVFLAGS = --coverage -O0 -g

SRCS := $(wildcard src/*.c)
OBJS := $(SRCS:.c=.o)
HDRS := $(wildcard include/*.h)
LIB  := libnn.a
SHLIB := libnn_shared.so

TESTS := tests/test_arena tests/test_tensor tests/test_autograd tests/test_zoo \
         tests/test_spatial tests/test_sequence tests/test_energy tests/test_render
SAN_TESTS := $(TESTS:tests/%=tests/%_asan)
EXAMPLES := examples/apps/tensor-basics/tensor-basics \
            examples/apps/xor-classifier/xor-classifier \
            examples/apps/denoising-autoencoder/denoising-autoencoder \
            examples/apps/image-classifier/image-classifier \
            examples/apps/gan-toy/gan-toy \
            examples/apps/sequence-copy/sequence-copy \
            examples/apps/ntm-memory/ntm-memory \
            examples/apps/hopfield-memory/hopfield-memory \
            examples/apps/rbm-cd/rbm-cd \
            examples/apps/reservoir-predictor/reservoir-predictor \
            examples/apps/kohonen-map/kohonen-map \
            examples/apps/render-graph/render-graph

.PHONY: all lib shared tests test asan coverage valgrind examples apps run_example run_examples run_apps clean

all: lib shared

lib: $(LIB) $(SHLIB)

shared: $(SHLIB)

$(LIB): $(OBJS)
	$(AR) rcs $@ $^

# Biblioteca compartilhada para os bindings Python (ctypes).
$(SHLIB): $(SRCS) $(HDRS)
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -shared $(SRCS) -lm -o $@
	cp $@ python/libnn/

src/%.o: src/%.c $(HDRS)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

tests/test_%: tests/test_%.c $(LIB) $(HDRS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) -lm -o $@

tests/test_%_asan: tests/test_%.c $(SRCS) $(HDRS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SANFLAGS) $< $(SRCS) -lm -o $@

examples/apps/tensor-basics/tensor-basics: examples/apps/tensor-basics/main.c $(LIB) $(HDRS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) -lm -o $@
examples/apps/xor-classifier/xor-classifier: examples/apps/xor-classifier/main.c $(LIB) $(HDRS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) -lm -o $@
examples/apps/denoising-autoencoder/denoising-autoencoder: examples/apps/denoising-autoencoder/main.c $(LIB) $(HDRS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) -lm -o $@
examples/apps/image-classifier/image-classifier: examples/apps/image-classifier/main.c $(LIB) $(HDRS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) -lm -o $@
examples/apps/gan-toy/gan-toy: examples/apps/gan-toy/main.c $(LIB) $(HDRS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) -lm -o $@
examples/apps/sequence-copy/sequence-copy: examples/apps/sequence-copy/main.c $(LIB) $(HDRS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) -lm -o $@
examples/apps/ntm-memory/ntm-memory: examples/apps/ntm-memory/main.c $(LIB) $(HDRS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) -lm -o $@
examples/apps/hopfield-memory/hopfield-memory: examples/apps/hopfield-memory/main.c $(LIB) $(HDRS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) -lm -o $@
examples/apps/rbm-cd/rbm-cd: examples/apps/rbm-cd/main.c $(LIB) $(HDRS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) -lm -o $@
examples/apps/reservoir-predictor/reservoir-predictor: examples/apps/reservoir-predictor/main.c $(LIB) $(HDRS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) -lm -o $@
examples/apps/kohonen-map/kohonen-map: examples/apps/kohonen-map/main.c $(LIB) $(HDRS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) -lm -o $@
examples/apps/render-graph/render-graph: examples/apps/render-graph/main.c $(LIB) $(HDRS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) -lm -o $@

examples apps: $(EXAMPLES)

run_example: run_examples

run_examples run_apps: $(EXAMPLES)
	@if [ -n "$(APP)" ]; then \
		./examples/apps/$(APP)/$(APP); \
	else \
		for e in $(EXAMPLES); do ./$$e || exit 1; done; \
	fi

tests: $(TESTS)

test: tests
	@for t in $(TESTS); do ./$$t || exit 1; done

asan: $(SAN_TESTS)
	@for t in $(SAN_TESTS); do \
		ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 ./$$t || exit 1; \
	done

# Cobertura de código: instrumenta com gcov, executa a suíte, captura com lcov
# e converte para o formato SonarQube Generic Coverage (coverage.xml).
coverage:
	$(MAKE) clean
	$(MAKE) lib test CFLAGS="$(COVFLAGS)" LDFLAGS="$(COVFLAGS)"
	lcov --capture --rc lcov_branch_coverage=1 --directory . --output-file coverage.raw.info
	lcov --remove coverage.raw.info --rc lcov_branch_coverage=1 '/usr/*' '*/tests/*' --output-file coverage.info
	rm -f coverage.raw.info
	python3 ci/lcov_to_sonar.py coverage.info coverage.xml .
	@echo "Cobertura gerada: coverage.xml"

valgrind: tests
	@if command -v valgrind >/dev/null 2>&1; then \
		for t in $(TESTS); do \
			valgrind --error-exitcode=42 --leak-check=full --show-leak-kinds=all ./$$t || exit 1; \
		done; \
	else \
		echo "valgrind não instalado; use 'make asan'."; \
	fi

clean:
	rm -f $(OBJS) $(LIB) $(SHLIB) python/libnn/$(SHLIB) $(TESTS) $(SAN_TESTS) $(EXAMPLES) \
	      coverage.info coverage.raw.info coverage.xml *.gcda *.gcno src/*.gcda src/*.gcno tests/*.gcda tests/*.gcno
