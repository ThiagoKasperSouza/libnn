# Marco 1 — Infraestrutura Base de Memória e Álgebra Linear

Este documento detalha o que foi entregue no Marco 1, os contratos de cada módulo, as decisões de implementação e a estratégia de validação.

## 1. Escopo entregue

| Item do plano | Entrega |
| :-- | :-- |
| `NC_Arena` (alocação e resets O(1)) | `include/nc_arena.h`, `src/nc_arena.c` |
| `NC_Tensor` (matmul, conv2d, RNG) | `include/nc_tensor.h`, `src/nc_tensor.c` |
| Matmul com tiling | GEMM em blocos 16×32×16 |
| Gerador Box-Muller | Xoshiro256\*\* + splitmix64 + Box-Muller |
| Testes de estresse com ASan/Valgrind | `tests/test_arena.c`, `tests/test_tensor.c`, alvos `make asan` / `make valgrind` |
| Header público | `include/neurochart.h` |
| Exemplo e documentação | `examples/marco1_basics.c`, este documento e `README.md` |

Fora de escopo (marcos seguintes): autograd, otimizadores, zoo de topologias, `im2col + GEMM`, memória recorrente/energia/spiking e renderização SVG.

## 2. Contratos e invariantes

### Arena

- O buffer é alinhado a `NC_ARENA_ALIGNMENT` (32 bytes por padrão); o ponteiro bruto do `malloc` fica logo antes do buffer alinhado, para que `nc_arena_free()` devolva exatamente o ponteiro original.
- `nc_arena_alloc` aceita apenas alinhamentos potência de 2 (`0` é tratado como `1`); qualquer outro valor retorna `NULL` sem consumir espaço.
- A aritmética de capacidade, padding e tamanho é verificada contra overflow; falhas de alocação devolvem `offset` ao valor anterior (rollback).
- `nc_arena_reset` zera apenas `offset` — a memória antiga permanece, o que mantém a operação O(1). Por isso os testes não assumem dados zerados após reset.
- `nc_arena_contains` responde sobre a **região física** `[buffer, buffer+capacity)`, não sobre blocos alocados. A arena não rastreia alocações individuais.

### Tensor

- `shape[0..ndim-1]` são as dimensões ativas, interpretadas como `[N, C, H, W]`; strides são contíguos row-major e `size` é o produto das dimensões.
- `nc_tensor_create` rejeita `ndim` fora de 1–4, dimensão zero e produtos que transbordem; `grad` é alocado somente quando `requires_grad`.
- As operações 2D validam o layout (strides coerentes com shape/size) antes de calcular offsets. Isso garante que todo índice válido gere um offset menor que `size`, mesmo para tensores montados fora do construtor.
- `nc_conv2d`: `input [N,C_in,H,W]`, `kernel [C_out,C_in,kH,kW]`, `output [N,C_out,oH,oW]` com `oH = (H+2p-kH)/stride + 1`.
- `nc_conv2d_transpose`: `kernel [C_in,C_out,kH,kW]`, `output [N,C_out,(H-1)*stride-2p+kH, ...]`.
- `nc_max_pool2d`: janelas `k×k`, sem padding.
- Parâmetros extremos (padding/stride que estouram `size_t`, kernels maiores que a entrada, shapes incompatíveis) são rejeitados sem tocar na saída.

### RNG

- Estado global Xoshiro256\*\* semeado por splitmix64; `nc_rng_seed(0)` é válido e determinístico. Sem semeadura explícita, usa-se uma semente padrão fixa.
- `nc_rng_next_double` usa os 53 bits altos; `nc_rng_next_normal` usa Box-Muller com o segundo valor da par em cache. O caso `u1 == 0` é evitado.
-Estado global implica não thread-safety (aceito neste marco).

### `nc_matmul`

- Suporta `ndim` 2–4; as duas últimas dimensões são as da matriz e as anteriores são batches.
- Tiling 16×32×16 com acumulador local; apenas a região `imax×jmax` do tile é zerada por bloco (≈2 KiB de stack).
- Se `C` compartilha memória com `A` ou `B` (total ou parcialmente), o GEMM é calculado em buffer temporário e copiado no layout real de `C`. O caminho sem aliasing não faz alocação.

## 3. Decisões de implementação

1. **Duas arenas explícitas em vez de um heap geral.** O reset O(1) da scratch arena é o mecanismo central para evitar `malloc`/`free` dentro dos loops de treino.
2. **Alinhamento por prefixo.** Guardar o ponteiro bruto antes do buffer alinhado preserva a estrutura pública `NC_Arena` com apenas três campos e mantém `free()` correto.
3. **Tiling pequeno.** O acumulador 16×32 cabe em ~2 KiB de stack, adequado também a threads com stack reduzida; blocos de K de 16 favorecem reuso em cache L1.
4. **Convolução direta neste marco.** O plano reserva `im2col + GEMM` para o Marco 3; a implementação atual prioriza correção e validação de limites.
5. **Hardening com validação de layout.** Em vez de checar overflow em cada expressão de offset dentro dos loops (custo no hot path), valida-se uma vez que o tensor é contíguo e coerente; a partir disso, índices válidos implicam offsets válidos.

## 4. Validação

```
make test  → test_arena: 357/357    test_tensor: 234124/234124
make asan  → ASan + UBSan sem erros, sem vazamentos
```

Cobertura principal: alinhamento (inclusive inválido), esgotamento e rollback, reset determinístico, estresse com milhares de blocos, `free` duplo, repetibilidade de semente (inclusive 0), distribuição normal, GEMM vs. referência ingênua, aliasing (`C==A`, `C==B`, batched), views com strides não-unitários, convolução com stride/batch/padding, transposta, pooling com janelas sobrepostas e parâmetros extremos.

`make valgrind` está disponível quando a ferramenta estiver instalada; neste ambiente ela não está, e o ASan cobre a verificação de vazamentos.

## 5. Como executar o exemplo

```sh
make run_example
```

O exemplo [`examples/marco1_basics.c`](../examples/marco1_basics.c) mostra o ciclo completo: inicializar as duas arenas, semear o RNG, criar/preencher tensores, multiplicar, demonstrar aliasing, convoluir, fazer pooling e resets da scratch.
