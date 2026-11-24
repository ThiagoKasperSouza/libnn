# libnn

SDK de redes neurais em **C11 puro, zero dependências externas** (apenas `stdlib`, `math`, `string`, `stdint`, `stddef`, `stdbool`). A arquitetura completa e o roadmap estão em [`Plano de Arquitetura_ SDK de Redes Neurais em C Puro (Zero Dependências).md`](./Plano%20de%20Arquitetura_%20SDK%20de%20Redes%20Neurais%20em%20C%20Puro%20%28Zero%20Depend%C3%AAncias%29.md).

**Status:** Marcos 1 a 3 implementados. Detalhes em [`docs/MARCO1.md`](./docs/MARCO1.md), [`docs/MARCO2.md`](./docs/MARCO2.md) e [`docs/MARCO3.md`](./docs/MARCO3.md).

## Marcos

| Marco | Escopo | Estado |
| :-- | :-- | :-- |
| 1 | Arena, tensores, matmul com tiling, RNG, testes de memória | ✅ implementado |
| 2 | Tape de autograd, Adam, P/FF/DFF/AE/DAE/SAE/DRN | ✅ implementado |
| 3 | `im2col` + GEMM, conv transposta, pooling, DCN/DN/DCIGN/GAN | ✅ implementado |
| 4 | Unrolling temporal, BPTT, LSTM/GRU/NTM | ⏳ |
| 5 | Energia, spiking e reservatórios | ⏳ |
| 6 | Renderizador SVG e certificação | ⏳ |

## Build e testes

```sh
make          # compila libnn.a
make test     # testes funcionais e de estresse de memória
make asan     # mesmos testes com AddressSanitizer + UBSan
make valgrind # se valgrind estiver disponível
make examples     # compila os exemplos
make run_examples # executa os três exemplos
make clean
```

Requer GCC/Clang com suporte a C11. Não há dependências de terceiros.

## Estrutura atual

```
libnn/
├── include/
│   ├── neurochart.h   # header público unificado
│   ├── nc_arena.h     # arena allocator
│   ├── nc_tensor.h    # tensores, RNG e álgebra linear
│   ├── nc_autograd.h  # fita e operações diferenciáveis
│   ├── nc_cells.h     # 13 tipos de célula e ativações
│   ├── nc_topology.h  # NC_Graph, arestas e forward/backward
│   ├── nc_trainers.h  # Adam e MSE
│   └── nc_zoo.h       # construtores P/FF/DFF/AE/DAE/SAE/DRN + DCN/DN/DCIGN/GAN
├── src/                # um .c por header acima (exceto neurochart.h)
├── examples/
│   ├── marco1_basics.c
│   ├── xor_perceptron.c
│   ├── dae_example.c
│   ├── dcn_synthetic.c
│   └── gan_toy.c
├── tests/
│   ├── test_common.h
│   ├── test_arena.c
│   ├── test_tensor.c
│   ├── test_autograd.c
│   ├── test_zoo.c
│   └── test_spatial.c
├── docs/
│   ├── MARCO1.md
│   ├── MARCO2.md
│   └── MARCO3.md
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
nc_rng_seed(42);

NC_Graph *net = nc_zoo_create_ff(&model, 2, 8, 1);
NC_Tensor *x = /* ... entradas na model arena ... */;
NC_Tensor *y = /* ... alvos na model arena ... */;
NC_OptimizerAdam opt = { 0.05f, 0.9f, 0.999f, 1e-8f };

for (size_t epoch = 0; epoch < 1000; ++epoch) {
    nc_arena_reset(&scratch);                       /* O(1) */
    NC_Tensor *pred = nc_graph_forward(net, x, &scratch);
    nc_autograd_record_mse(net->tape, pred, y);     /* semente do gradiente */
    nc_graph_backward(net);
    nc_train_backprop_step(net, &opt);
}

nc_graph_destroy(net);
nc_arena_free(&scratch);
nc_arena_free(&model);
```

Exemplos completos: [`examples/marco1_basics.c`](./examples/marco1_basics.c),
[`examples/xor_perceptron.c`](./examples/xor_perceptron.c),
[`examples/dae_example.c`](./examples/dae_example.c),
[`examples/dcn_synthetic.c`](./examples/dcn_synthetic.c) e
[`examples/gan_toy.c`](./examples/gan_toy.c).

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
| `nc_conv2d / conv2d_transpose / max_pool2d` | Operações 2D diretas sobre tensores `[N,C,H,W]` |
| `nc_conv2d_im2col / conv2d_transpose_im2col` | Mesmas operações via im2col + GEMM (buffer na scratch) |
| `nc_avg_pool2d / nc_tensor_flatten` | Pooling médio e achatamento `[N,C,H,W] → [N,C*H*W]` |

## API do Marco 2

### Autograd e células

| Função | Contrato |
| :-- | :-- |
| `nc_autograd_tape_init(arena, capacity)` | Fita na arena persistente; `NULL` em falha |
| `nc_autograd_tape_reset / backward` | Reinicia a fita / roda os nós em ordem reversa |
| `nc_nn_matmul / add / activation / dropout` | Forward + gravação do nó na fita |
| `nc_compute_mse_loss / nc_autograd_record_mse` | Erro quadrático médio e semente de gradiente |
| `nc_cell_create / nc_cell_init_params / nc_cell_forward` | Célula tipada com pesos Xavier e ativação |

### Grafo, treino e zoo

| Função | Contrato |
| :-- | :-- |
| `nc_graph_create / add_cell / connect` | Monta o DAG na model arena |
| `nc_graph_forward(g, input, scratch)` | Achata a entrada para `[batch, features]` e executa as células |
| `nc_graph_backward(g)` | Aplica sparsity (SAE) e propaga gradientes |
| `nc_train_backprop_step(g, opt)` | Adam com correção de viés; estados criados na primeira chamada |
| `nc_zoo_create_{perceptron,ff,dff,ae,dae,sae,drn}` | Fábricas das 7 topologias do Marco 2 |

## API do Marco 3

| Função | Contrato |
| :-- | :-- |
| `nc_nn_conv2d / conv2d_transpose / max_pool2d / avg_pool2d` | Forward espacial com gravação na fita |
| `nc_cell_init_conv_params` | Inicializa kernel 4D `[c_out,c_in,kH,kW]` com Xavier |
| `nc_graph_backward_no_zero` | Backward sem zerar gradientes (usado pelo gerador da GAN) |
| `nc_zoo_create_{dcn,dn,dcign}` | DCN, DN e autoencoder convolucional |
| `nc_zoo_create_gan / nc_train_gan_step` | Par gerador/discriminador e passo minimax |

## Validação

`make test` cobre cinco binários: arena/tensores (Marco 1), gradientes analíticos, estrutura das 7 topologias feedforward, treino de XOR e reconstrução AE/DAE/SAE/DRN (Marco 2) e, no Marco 3, im2col vs. convolução direta, gradientes espaciais por diferenças finitas, DCN/DN/DCIGN com perda decrescente e 200 passos de GAN. `make asan` repete tudo com ASan/UBSan e detecção de vazamentos.

## Limitações conhecidas

- Backward de convolução usa laços diretos; o forward é que usa im2col + GEMM.
- O latente do DCIGN é determinístico (`mu = x`); amostragem/KL chegam no Marco 5.
- A GAN usa MSE como objective adversarial, não BCE.
- O RNG é estado global, não thread-safe.
- Alias entre `C` e as entradas de `nc_matmul` usa `malloc` temporário (caminho excepcional, fora do hot path).
- Células spiking e recorrentes chegam nos Marcos 4–5; `nc_render` (SVG) no Marco 6.
