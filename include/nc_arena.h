#ifndef NC_ARENA_H
#define NC_ARENA_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/* Alinhamento padrão das alocações da arena (32 bytes = AVX2). */
#ifndef NC_ARENA_ALIGNMENT
#define NC_ARENA_ALIGNMENT 32u
#endif

/*
 * Alocador sequencial (arena / bump allocator).
 *
 * - buffer   : início da região de memória alinhada (endereço alinhado a
 *              NC_ARENA_ALIGNMENT ou superior).
 * - capacity : capacidade total, em bytes.
 * - offset   : próximo byte livre; resetar para 0 é O(1).
 *
 * Toda a memória é devolvida de uma vez por nc_arena_free(). Não é possível
 * liberar alocações individuais — use arenas separadas para dados persistentes
 * (model_arena) e temporários (scratch_arena).
 */
typedef struct {
    uint8_t *buffer;
    size_t capacity;
    size_t offset;
} NC_Arena;

/* Aloca `capacity_in_bytes` alinhados. Retorna uma arena zerada em caso de falha. */
NC_Arena nc_arena_init(size_t capacity_in_bytes);

/*
 * Aloca `size` bytes alinhados a `alignment` (potência de 2; 0 é tratado
 * como 1). Valores que não são potência de 2 retornam NULL. Retorna NULL
 * também se não houver espaço. A memória NÃO é zerada (exceto na
 * inicialização da arena).
 */
void *nc_arena_alloc(NC_Arena *arena, size_t size, size_t alignment);

/* Libera todos os bytes alocados em O(1), mantendo a arena utilizável. */
void nc_arena_reset(NC_Arena *arena);

/* Devolve a memória da arena ao sistema e zera a estrutura. */
void nc_arena_free(NC_Arena *arena);

/* Consultas auxiliares. */
size_t nc_arena_bytes_used(const NC_Arena *arena);
size_t nc_arena_bytes_available(const NC_Arena *arena);

/*
 * Retorna true se o endereço estiver dentro da REGIÃO FÍSICA da arena
 * ([buffer, buffer+capacity)), inclusive bytes ainda não alocados ou
 * restos de alocações anteriores a um reset(). Não indica que o ponteiro
 * corresponde a um bloco ativo — a arena não rastreia alocações individuais.
 */
bool nc_arena_contains(const NC_Arena *arena, const void *ptr);

#endif /* NC_ARENA_H */
