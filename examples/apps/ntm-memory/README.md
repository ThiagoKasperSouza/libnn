# NTM — memória endereçável por conteúdo

## O que esta aplicação faz

Demonstra uma **Neural Turing Machine**: em vez de guardar informação apenas em estados ocultos, a rede possui um banco de memória externo com 4 posições de 5 dimensões. A cada passo ela calcula uma chave a partir da entrada, encontra as posições mais similares por conteúdo, lê uma combinação delas e escreve de volta no banco.

## Arquitetura

```
INPUT(3) → DIFFERENT_MEMORY/NTM(4 slots × 5 dims) → OUTPUT(2)
```

Construtor: `nc_zoo_create_ntm(&model, 3, 4, 5, 2)`.

Mecanismo:

1. `k = tanh(x · W_key)` — chave de busca;
2. `scores = M · kᵀ` e `w = softmax(scores)` — atenção por conteúdo;
3. `read = wᵀ · M` — leitura;
4. `M ← M + w · tanh(x · W_write)` — escrita.

A transposição usada nos passos 2 e 3 é diferenciável (`nc_nn_transpose`), de modo que `W_key` também recebe gradiente.

## Dados

Cinco vetores 3D (três one-hot e duas combinações) apresentados como uma sequência de 5 passos.

## Como executar e como treinar

A aplicação incluída faz apenas o **forward** e imprime o conteúdo da memória e as leituras:

```sh
make run_apps APP=ntm-memory
```

Para transformá-la em treino, conecte as saídas a um alvo e use o mesmo fluxo de BPTT da aplicação de sequência:

```c
nc_arena_reset(&scratch);
nc_sequence_reset_state(ntm);
NC_Tensor **outs = nc_sequence_forward(ntm, x, T, &scratch);
for (size_t t = 0; t < T; ++t)
    nc_autograd_record_mse(ntm->tape, outs[t], target_t);
nc_sequence_backward(ntm);
nc_train_backprop_step(ntm, &opt);
```

### Hiperparâmetros sugeridos

| Parâmetro | Valor |
| :-- | :-- |
| Passos | 5–20 |
| Slots / dimensão da memória | 4 / 5 |
| Taxa de aprendizado | 0.005–0.01 |
| Semente RNG | 42 |

## Saída esperada

A memória final com 4 linhas de 5 valores, a leitura final (5 dimensões) e as saídas do decodificador em cada passo.

## Limitações

- Similaridade por produto escalar, sem normalização cosine completa.
- Apenas `batch = 1`.
- A escrita atualiza o estado da memória, mas não propaga gradiente persistente entre passos.
