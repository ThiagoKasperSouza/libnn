#include "nc_arena.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/*
 * A arena guarda o ponteiro bruto do malloc() logo antes do buffer alinhado.
 * Assim nc_arena_free() sempre devolve exatamente o ponteiro recebido de
 * malloc(), mesmo quando o alinhamento base exigiu um deslocamento.
 */
NC_Arena nc_arena_init(size_t capacity_in_bytes)
{
    NC_Arena arena = { NULL, 0, 0 };

    if (capacity_in_bytes == 0) {
        return arena;
    }

    const size_t align = NC_ARENA_ALIGNMENT;

    /* capacity + (align - 1) + sizeof(void*) com verificação de overflow. */
    if (capacity_in_bytes > SIZE_MAX - (align - 1) - sizeof(void *)) {
        return arena;
    }
    const size_t raw_size = capacity_in_bytes + (align - 1) + sizeof(void *);

    uint8_t *raw = (uint8_t *)malloc(raw_size);
    if (raw == NULL) {
        return arena;
    }

    const uintptr_t unaligned = (uintptr_t)raw + sizeof(void *);
    const uintptr_t aligned =
        (unaligned + (align - 1)) & ~(uintptr_t)(align - 1);

    arena.buffer = (uint8_t *)aligned;
    arena.capacity = capacity_in_bytes;
    arena.offset = 0;

    memcpy(arena.buffer - sizeof(void *), &raw, sizeof(raw));

    /* Zeramos apenas na inicialização: reset() permanece O(1). */
    memset(arena.buffer, 0, capacity_in_bytes);
    return arena;
}

void *nc_arena_alloc(NC_Arena *arena, size_t size, size_t alignment)
{
    if (arena == NULL || arena->buffer == NULL || size == 0) {
        return NULL;
    }
    if (alignment == 0) {
        alignment = 1;
    }
    if ((alignment & (alignment - 1)) != 0) {
        return NULL; /* contrato: alinhamento deve ser potência de 2 */
    }

    const uintptr_t address = (uintptr_t)(arena->buffer + arena->offset);
    const size_t padding = (size_t)((alignment - (address % alignment)) % alignment);

    if (padding > arena->capacity - arena->offset) {
        return NULL;
    }
    const size_t start = arena->offset + padding;

    if (size > arena->capacity - start) {
        return NULL;
    }

    void *ptr = arena->buffer + start;
    arena->offset = start + size;
    return ptr;
}

void nc_arena_reset(NC_Arena *arena)
{
    if (arena != NULL) {
        arena->offset = 0;
    }
}

void nc_arena_free(NC_Arena *arena)
{
    if (arena == NULL || arena->buffer == NULL) {
        return;
    }

    void *raw = NULL;
    memcpy(&raw, arena->buffer - sizeof(void *), sizeof(raw));
    free(raw);

    arena->buffer = NULL;
    arena->capacity = 0;
    arena->offset = 0;
}

size_t nc_arena_bytes_used(const NC_Arena *arena)
{
    return arena != NULL ? arena->offset : 0;
}

size_t nc_arena_bytes_available(const NC_Arena *arena)
{
    if (arena == NULL) {
        return 0;
    }
    return arena->capacity - arena->offset;
}

bool nc_arena_contains(const NC_Arena *arena, const void *ptr)
{
    if (arena == NULL || arena->buffer == NULL || ptr == NULL) {
        return false;
    }

    const uintptr_t base = (uintptr_t)arena->buffer;
    const uintptr_t addr = (uintptr_t)ptr;
    return addr >= base && addr < base + arena->capacity;
}
