#include "test_common.h"
#include "neurochart.h"

#include <stdint.h>
#include <string.h>

static void test_init_and_alignment(void)
{
    NC_Arena arena = nc_arena_init(1024);
    NC_CHECK(arena.buffer != NULL);
    NC_CHECK(arena.capacity == 1024);
    NC_CHECK(arena.offset == 0);
    NC_CHECK(((uintptr_t)arena.buffer % NC_ARENA_ALIGNMENT) == 0);

    /* Memória zerada apenas na inicialização. */
    bool all_zero = true;
    for (size_t i = 0; i < arena.capacity; ++i) {
        if (arena.buffer[i] != 0) {
            all_zero = false;
            break;
        }
    }
    NC_CHECK(all_zero);
    NC_CHECK(nc_arena_bytes_used(&arena) == 0);
    NC_CHECK(nc_arena_bytes_available(&arena) == 1024);
    NC_CHECK(nc_arena_contains(&arena, arena.buffer));

    nc_arena_free(&arena);
    NC_CHECK(arena.buffer == NULL);
    NC_CHECK(arena.capacity == 0);
}

static void test_aligned_allocations_do_not_overlap(void)
{
    NC_Arena arena = nc_arena_init(4096);
    void *ptrs[8] = { 0 };
    const size_t alignments[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };

    for (size_t i = 0; i < 8; ++i) {
        ptrs[i] = nc_arena_alloc(&arena, 37, alignments[i]);
        NC_CHECK(ptrs[i] != NULL);
        NC_CHECK(((uintptr_t)ptrs[i] % alignments[i]) == 0);
        NC_CHECK(nc_arena_contains(&arena, ptrs[i]));
        memset(ptrs[i], (int)(i + 1), 37);
    }
    for (size_t i = 0; i < 8; ++i) {
        if (ptrs[i] == NULL) {
            continue;
        }
        const unsigned char *bytes = (const unsigned char *)ptrs[i];
        for (size_t j = 0; j < 37; ++j) {
            NC_CHECK(bytes[j] == (unsigned char)(i + 1));
        }
    }
    NC_CHECK(nc_arena_bytes_used(&arena) == arena.offset);
    nc_arena_free(&arena);
}

static void test_exhaustion_and_reset(void)
{
    NC_Arena arena = nc_arena_init(256);

    void *first = nc_arena_alloc(&arena, 100, NC_ARENA_ALIGNMENT);
    NC_CHECK(first != NULL);
    void *second = nc_arena_alloc(&arena, 100, NC_ARENA_ALIGNMENT);
    NC_CHECK(second != NULL);

    const size_t before = arena.offset;
    NC_CHECK(nc_arena_alloc(&arena, 100, NC_ARENA_ALIGNMENT) == NULL);
    NC_CHECK(arena.offset == before); /* falha não consome espaço */
    NC_CHECK(nc_arena_alloc(&arena, SIZE_MAX, 32) == NULL);

    nc_arena_reset(&arena);
    NC_CHECK(arena.offset == 0);
    NC_CHECK(nc_arena_bytes_available(&arena) == 256);
    void *again = nc_arena_alloc(&arena, 100, NC_ARENA_ALIGNMENT);
    NC_CHECK(again == first); /* reutilização em O(1) */

    nc_arena_free(&arena);
}

static void test_stress_many_small_allocations(void)
{
    NC_Arena arena = nc_arena_init(1024 * 1024);
    void *blocks[8192] = { 0 };
    size_t count = 0;

    while (count < 8192) {
        void *p = nc_arena_alloc(&arena, 24, 32);
        if (p == NULL) {
            break;
        }
        memset(p, 0xA5, 24);
        blocks[count++] = p;
    }
    NC_CHECK(count > 0);

    for (size_t i = 0; i < count; ++i) {
        const unsigned char *bytes = (const unsigned char *)blocks[i];
        for (size_t j = 0; j < 24; ++j) {
            if (bytes[j] != 0xA5) {
                NC_CHECK(bytes[j] == 0xA5);
                break;
            }
        }
    }

    const size_t first_count = count;
    nc_arena_reset(&arena);
    count = 0;
    while (count < 8192) {
        void *p = nc_arena_alloc(&arena, 24, 32);
        if (p == NULL) {
            break;
        }
        blocks[count++] = p;
    }
    NC_CHECK(count == first_count); /* reset determinístico */

    nc_arena_free(&arena);
}

static void test_edge_cases(void)
{
    NC_Arena empty = nc_arena_init(0);
    NC_CHECK(empty.buffer == NULL);
    NC_CHECK(nc_arena_alloc(&empty, 1, 8) == NULL);
    nc_arena_reset(&empty);
    nc_arena_free(&empty);

    NC_Arena arena = nc_arena_init(64);
    NC_CHECK(nc_arena_alloc(NULL, 8, 8) == NULL);
    NC_CHECK(nc_arena_alloc(&arena, 0, 8) == NULL);
    NC_CHECK(nc_arena_alloc(&arena, 8, 0) != NULL); /* alinhamento 0 => 1 */
    NC_CHECK(!nc_arena_contains(&arena, NULL));
    nc_arena_free(&arena);
    nc_arena_free(NULL);
    nc_arena_reset(NULL);
    NC_CHECK(nc_arena_bytes_used(NULL) == 0);
}

static void test_alignment_contract_and_contains(void)
{
    NC_Arena arena = nc_arena_init(1024);

    const size_t before = arena.offset;
    NC_CHECK(nc_arena_alloc(&arena, 8, 3) == NULL);
    NC_CHECK(nc_arena_alloc(&arena, 8, 6) == NULL);
    NC_CHECK(nc_arena_alloc(&arena, 8, 100) == NULL);
    NC_CHECK(arena.offset == before); /* alinhamento inválido não consome espaço */

    void *p = nc_arena_alloc(&arena, 16, 8);
    NC_CHECK(p != NULL);

    /* nc_arena_contains descreve a região física, não blocos alocados. */
    NC_CHECK(nc_arena_contains(&arena, p));
    NC_CHECK(nc_arena_contains(&arena, arena.buffer + 500)); /* não alocado */
    NC_CHECK(!nc_arena_contains(&arena, arena.buffer + arena.capacity));

    nc_arena_free(&arena);
    nc_arena_free(&arena); /* idempotente */
    NC_CHECK(!nc_arena_contains(&arena, p));
}

int main(void)
{
    printf("=== test_arena (Marco 1) ===\n");
    test_init_and_alignment();
    test_aligned_allocations_do_not_overlap();
    test_exhaustion_and_reset();
    test_stress_many_small_allocations();
    test_edge_cases();
    test_alignment_contract_and_contains();
    NC_TEST_SUMMARY("test_arena");
}
