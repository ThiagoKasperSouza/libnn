# libnn

SDK de redes neurais em **C11 puro, zero dependências externas** (apenas `stdlib`, `math`, `string`, `stdint`, `stddef`, `stdbool`). A arquitetura completa e o roadmap estão em [`Plano de Arquitetura_ SDK de Redes Neurais em C Puro (Zero Dependências).md`](./Plano%20de%20Arquitetura_%20SDK%20de%20Redes%20Neurais%20em%20C%20Puro%20%28Zero%20Depend%C3%AAncias%29.md).

**Status:** Marco 1 (infraestrutura base) implementado. Detalhes em [`docs/MARCO1.md`](./docs/MARCO1.md).

## Marcos

| Marco | Escopo | Estado |
| :-- | :-- | :-- |
| 1 | Arena, tensores, matmul com tiling, RNG, testes de memória | ✅ implementado |
| 2 | Tape de autograd, Adam, P/FF/DFF/AE/DAE/SAE/DRN | ⏳ |
| 3 | `im2col` + GEMM, conv transposta, pooling, DCN/DN/DCIGN/GAN | ⏳ |
| 4 | Unrolling temporal, BPTT, LSTM/GRU/NTM | ⏳ |
| 5 | Energia, spiking e reservatórios | ⏳ |
| 6 | Renderizador SVG e certificação | ⏳ |

## Build e testes

```sh
make          # compila libnn.a
make test     # testes funcionais e de estresse de memória
make asan     # mesmos testes com AddressSanitizer + UBSan
make valgrind # se valgrind estiver disponível
make examples # compila os exemplos
make run_example
make clean
```

Requer GCC/Clang com suporte a C11. Não há dependências de terceiros.

## Estrutura atual

```
libnn/
├── include/
│   ├── neurochart.h   # header público unificado
│   ├── nc_arena.h     # arena allocator
│   └── nc_tensor.h    # tensores, RNG e álgebra linear
├── src/
│   ├── nc_arena.c
│   └── nc_tensor.c
├── examples/
│   └── marco1_basics.c
├── tests/
│   ├── test_common.h
│   ├── test_arena.c
│   └── test_tensor.c
├── docs/
│   └── MARCO1.md
├── Makefile
└── README.md
```

## Modelo de memória

Duas arenas com papéis distintos:

- **`model_arena`** — pesos, vieses e estruturas que vivem durante todo o ciclo do modelo.
- **`scratch_arena`** — ativações e buffers temporários; `nc_arena_reset()` devolve tudo em **O(1)** ao final de cada iteração.

A alocação é sequencial e alinhada (32 bytes por padrão, AVX2). Não há `free` individual; `nc_arena_free()` devolve a região inteira. As funções de criação fazem rollback do offset em caso de falta de memória.

## Exemplo rápido

```c
#include "neurochart.h"

NC_Arena model = nc_arena_init(16u * 1024 * 1024);
NC_Arena scratch = nc_arena_init(4u * 1024 * 1024);

nc_rng_seed(42); /* determinístico */
size_t shape[2] = { 2, 3 };
NC_Tensor *w = nc_tensor_create(&model, shape, 2, true);
nc_tensor_fill_random_normal(w, 0.0f, 0.1f);

/* ... montar x e y na scratch ... */
nc_matmul(w, x, y);

nc_arena_reset(&scratch); /* O(1) por iteração */
nc_arena_free(&scratch);
nc_arena_free(&model);
```

Exemplo completo e executável: [`examples/marco1_basics.c`](./examples/marco1_basics.c).

## API do Marco 1

### Arena (`nc_arena.h`)

| Função | Contrato |
| :-- | :-- |
| `nc_arena_init(capacity)` | Aloca região alinhada e zerada; arena zerada em falha |
| `nc_arena_alloc(a, size, align)` | Alinhamento deve ser potência de 2 (`0 ⇒ 1`); `NULL` se inválido/sem espaço |
| `nc_arena_reset(a)` | `offset = 0` em O(1), não zera a memória |
| `nc_arena_free(a)` | Devolve a região; idempotente |
| `nc_arena_bytes_used/available` | Consultas |
| `nc_arena_contains(a, ptr)` | Verdadeiro se o endereço está na **região física**, inclusive bytes não alocados |

### Tensores, RNG e álgebra (`nc_tensor.h`)

| Função | Contrato |
| :-- | :-- |
| `nc_tensor_create(arena, shape, ndim, requires_grad)` | 1–4 dimensões contíguas, `grad` opcional, dados zerados |
| `nc_tensor_fill_zeros / random_normal / random_uniform` | Preenchimento de `data` (e `grad`, nos zeros) |
| `nc_rng_seed / next_u64 / next_double / next_normal` | Xoshiro256\*\* + splitmix64 + Box-Muller, global e determinístico |
| `nc_matmul(A, B, C)` | GEMM com tiling; lida com aliasing entre `C` e as entradas |
| `nc_conv2d / conv2d_transpose / max_pool2d` | Operações 2D sobre tensores `[N,C,H,W]` |

## Validação

`make test` executa testes de alinhamento, esgotamento, reset, estresse com milhares de blocos, RNG, GEMM (comparado com referência ingênua), aliasing, views com strides não-unitários, convolução, transposta, pooling e casos-limite. `make asan` roda a mesma suíte com ASan/UBSan e detecção de vazamentos.

## Limitações conhecidas

- `nc_conv2d` usa loops diretos; `im2col + GEMM` está previsto para o Marco 3.
- O RNG é estado global, não thread-safe.
- Alias entre `C` e as entradas de `nc_matmul` usa `malloc` temporário (caminho excepcional, fora do hot path).
- O módulo `nc_render` e as topologias do `nc_zoo` ainda não existem.
