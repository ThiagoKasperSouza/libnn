# Classificador de imagens (CNN)

## O que esta aplicação faz

Classifica imagens sintéticas 8×8 em duas classes: faixa horizontal ou faixa vertical. Demonstra o caminho convolucional completo do Marco 3: `im2col + GEMM`, pooling e uma cabeça densa.

## Arquitetura

```
INPUT [1,1,8,8]
  → KERNEL/CONV [8 filtros 3×3, padding 1, ReLU]   → [1,8,8,8]
  → CONV_OR_POOL [maxpool 2×2, stride 2]           → [1,8,4,4]
  → OUTPUT denso [8*4*4 → 2 classes]
```

Construtor: `nc_zoo_create_dcn(&model, 1, 8, 8, 2)`.

O mapa de características 4D é achatado para `[N, C*H*W]` apenas quando a camada densa o consome; o gradiente volta até os filtros porque a view compartilha o buffer de gradiente.

## Dados

Quatro imagens 8×8 geradas em memória: duas com faixa horizontal e duas com faixa vertical. Alvos em one-hot de 2 classes.

## Como treinar

```sh
make run_apps APP=image-classifier
```

O loop é o mesmo dos exemplos anteriores, com a diferença de que a entrada é um tensor 4D `[N, C, H, W]`:

```c
nc_arena_reset(&scratch);
NC_Tensor *pred = nc_graph_forward(net, x, &scratch);
nc_autograd_record_mse(net->tape, pred, y);
nc_graph_backward(net);
nc_train_backprop_step(net, &opt);
```

### Hiperparâmetros

| Parâmetro | Valor |
| :-- | :-- |
| Épocas | 800 |
| Taxa de aprendizado | 0.01 |
| Filtros / kernel | 8 / 3×3 |
| Pooling | 2×2, stride 2 |
| Semente RNG | 2026 |

## Saída esperada

A loss vai de ~0.39 para ~0.000 e a acurácia no conjunto sintético é **4/4**.

## Como adaptar

- Mude `channels`, `h` e `w` no construtor para outras dimensões de imagem (o pooling assume dimensões pares).
- O plano prevê um exemplo `mnist_dcn`; carregar MNIST exigiria um leitor de arquivos, o que foge da política de zero dependências. Substitua `pixels` pelos seus próprios dados 8×8.
