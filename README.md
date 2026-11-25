# libnn

SDK de redes neurais em **C11 puro, zero dependências externas** (apenas `stdlib`, `math`, `string`, `stdint`, `stddef`, `stdbool`). A arquitetura completa e o roadmap estão em [`Plano de Arquitetura_ SDK de Redes Neurais em C Puro (Zero Dependências).md`](./Plano%20de%20Arquitetura_%20SDK%20de%20Redes%20Neurais%20em%20C%20Puro%20%28Zero%20Depend%C3%AAncias%29.md).

**Status:** Marcos 1 a 5 implementados. Detalhes em [`docs/MARCO1.md`](./docs/MARCO1.md) a [`docs/MARCO5.md`](./docs/MARCO5.md).

## Marcos

| Marco | Escopo | Estado |
| :-- | :-- | :-- |
| 1 | Arena, tensores, matmul com tiling, RNG, testes de memória | ✅ implementado |
| 2 | Tape de autograd, Adam, P/FF/DFF/AE/DAE/SAE/DRN | ✅ implementado |
| 3 | `im2col` + GEMM, conv transposta, pooling, DCN/DN/DCIGN/GAN | ✅ implementado |
| 4 | Unrolling temporal, BPTT, RNN/LSTM/GRU/NTM | ✅ implementado |
| 5 | Energia, spiking e reservatórios | ✅ implementado |
| 6 | Renderizador SVG e certificação | ⏳ |

## Build e testes

```sh
make          # compila libnn.a
make test     # testes funcionais e de estresse de memória
make asan     # mesmos testes com AddressSanitizer + UBSan
make valgrind # se valgrind estiver disponível
make apps                         # compila todas as aplicações
make run_apps                     # executa todas
make run_apps APP=xor-classifier  # executa apenas uma
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
│   ├── nc_trainers.h  # Adam, MSE e treino GAN
│   ├── nc_sequence.h  # unrolling temporal e BPTT
│   └── nc_zoo.h       # construtores das 11 topologias implementadas
├── src/                # um .c por header acima (exceto neurochart.h)
├── examples/
│   ├── README.md       # índice das aplicações
│   └── apps/            # uma pasta por aplicação, com main.c + README.md
│       ├── tensor-basics/
│       ├── xor-classifier/
│       ├── denoising-autoencoder/
│       ├── image-classifier/
│       ├── gan-toy/
│       ├── sequence-copy/
│       ├── ntm-memory/
│       ├── hopfield-memory/
│       ├── rbm-cd/
│       ├── reservoir-predictor/
│       └── kohonen-map/
├── tests/
│   ├── test_common.h
│   ├── test_arena.c
│   ├── test_tensor.c
│   ├── test_autograd.c
│   ├── test_zoo.c
│   ├── test_spatial.c
│   └── test_sequence.c
├── docs/
│   ├── MARCO1.md
│   ├── MARCO2.md
│   ├── MARCO3.md
│   ├── MARCO4.md
│   └── MARCO5.md
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

Exemplos completos: o índice em [`examples/README.md`](./examples/README.md) lista as sete aplicações, cada uma com `main.c` e um README explicando a arquitetura, os dados e como treinar.

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

## API do Marco 4

| Função | Contrato |
| :-- | :-- |
| `nc_nn_mul / nc_nn_scale / nc_nn_softmax` | Operações diferenciáveis usadas por portes e atenção |
| `nc_sequence_forward(g, x_seq[3D], T, scratch)` | Desdobra T passos numa única fita; retorna as saídas por passo |
| `nc_sequence_backward(g)` | Zera gradiente e roda o BPTT na fita completa |
| `nc_sequence_reset_state(g)` | Limpa h/c/memória antes de nova sequência |
| `nc_cell_init_gated_params` | Pesos de entrada/recorrência/viés por porte |
| `nc_zoo_create_{rnn,lstm,gru,ntm}` | Fábricas recorrentes |

## API do Marco 5

| Função | Contrato |
| :-- | :-- |
| `nc_train_contrastive_divergence` | CD-k para RBM/BM/Hopfield estocástico |
| `nc_train_hebbian_step / nc_train_stdp_step` | Aprendizado Hebbiano e STDP para LIF |
| `nc_train_reservoir_ridge_regression` | Pseudoinversa ridge para ELM/ESN/LSM/RBF |
| `nc_train_kohonen_bmu_step` | Atualização competitiva do BMU |
| `nc_train_kernel_hinge_loss` | Hinge loss linear (SVM) |
| `nc_zoo_create_{markov_chain,hopfield,boltzmann,rbm,dbn,vae,lsm,elm,esn,kohonen,rbf,svm}` | Fábricas de energia/reservatório |

## Validação

`make test` cobre sete binários: arena/tensores (Marco 1), gradientes analíticos, topologias feedforward (Marco 2), operações espaciais e GAN (Marco 3), BPTT e RNN/LSTM/GRU/NTM (Marco 4) e, no Marco 5, CD-k, Hebb/Hopfield, STDP/LIF, ridge (ELM/RBF), BMU e hinge loss — incluindo os 12 construtores. `make asan` repete tudo com ASan/UBSan e detecção de vazamentos.

## Limitações conhecidas

- Backward de convolução usa laços diretos; o forward é que usa im2col + GEMM.
- O latente do DCIGN é determinístico (`mu = x`); amostragem/KL chegam no Marco 5.
- A GAN usa MSE como objective adversarial, não BCE.
- O RNG é estado global, não thread-safe.
- Alias entre `C` e as entradas de `nc_matmul` usa `malloc` temporário (caminho excepcional, fora do hot path).
- A NTM usa produto escalar não normalizado e apenas batch 1; a escrita não propaga gradiente entre passos.
- Células spiking/energéticas chegam no Marco 5; `nc_render` (SVG) no Marco 6.
