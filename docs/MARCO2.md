# Marco 2 — Motor Autograd e Topologias Feedforward

Detalhes do que foi entregue no Marco 2, os contratos adotados, as decisões de projeto e a validação.

## 1. Escopo entregue

| Item do plano | Entrega |
| :-- | :-- |
| Fita de diferenciação (`NC_Tape`) | `include/nc_autograd.h`, `src/nc_autograd.c` |
| Operações diferenciáveis | matmul, add, ReLU, Tanh, Sigmoid, GELU, dropout, MSE |
| Células | `include/nc_cells.h`, `src/nc_cells.c` (13 tipos declarados; 5 comportamentais neste marco) |
| Topologia e grafo | `include/nc_topology.h`, `src/nc_topology.c` |
| Treinadores | `include/nc_trainers.h`, `src/nc_trainers.c` — Adam + MSE |
| Zoo | `include/nc_zoo.h`, `src/nc_zoo.c` — P, FF, DFF, AE, DAE, SAE, DRN |
| Testes | `tests/test_autograd.c`, `tests/test_zoo.c` |
| Exemplos | `examples/xor_perceptron.c`, `examples/dae_example.c` |

## 2. Fluxo de uma iteração

```c
nc_arena_reset(&scratch);                       /* libera ativações em O(1) */
NC_Tensor *pred = nc_graph_forward(g, x, &scratch);
float loss = nc_compute_mse_loss(pred, target);
nc_autograd_record_mse(g->tape, pred, target);  /* semeia pred->grad */
nc_graph_backward(g);                           /* zera grads, aplica L1, roda a fita */
nc_train_backprop_step(g, &opt);                /* Adam */
```

A fita é alocada uma única vez na `model_arena` com capacidade fixa; `nc_graph_forward` reinicia `count = 0`. Os nós apontam para ativações alocadas na `scratch_arena`, que permanecem válidas até o `backward` da mesma iteração.

## 3. Contratos e invariantes

### Fita e operações

- `NC_TapeNode` guarda até três entradas, a saída, uma `backward_fn` e parâmetros específicos da operação (`NC_OpParams`).
- Cada operação diferenciável calcula o forward e registra o nó. O backward acumula (`+=`) nos gradientes das entradas, permitindo que uma mesma ativação seja usada por mais de uma operação (ex.: skip connections).
- Gradientes de parâmetros são zerados no início de `nc_graph_backward`, não no forward.
- `nc_nn_*` opera sobre tensores 2D `[batch, features]` e exige uma fita válida. `nc_nn_matmul` rejeita `A.cols != B.rows`; `nc_nn_add` rejeita contagens de linhas incompatíveis (só aceita iguais ou broadcasting de um dos lados).
- O grafo achata entradas **contíguas** de qualquer `ndim` para `[batch, features]`; entradas não-contíguas são rejeitadas via `nc_tensor_layout_valid()` em vez de serem lidas incorretamente.
- `nc_autograd_record_mse` não aloca: o alvo viaja em `inputs[1]` do nó e o fator `2/|pred|` é calculado no backward. Não há crescimento da model arena por iteração.
- `nc_nn_dropout` usa *inverted dropout* (escala `1/keep_prob` no forward e no backward). A máscara vive na scratch arena.

### Células e grafo

- `NC_Cell` mantém os 13 tipos do infográfico e a union de estados do plano; neste marco são comportamentais `INPUT`, `NOISY_INPUT` (ruído gaussiano **ou** dropout), `HIDDEN`, `OUTPUT` e `MATCH_INPUT_OUTPUT`.
- A transformação de uma célula com parâmetros é `z = W·x + b`, seguida da ativação configurada. Pesos são inicializados com distribuição normal de desvio `sqrt(2/(fan_in+fan_out))`.
- As células são executadas em ordem de id (o zoo constrói apenas DAGs). Múltiplas arestas entrada em uma célula são somadas com `nc_nn_add`, o que propaga gradiente para todas elas — é assim que funcionam os skips do DRN.
- `NC_EDGE_FROZEN` está declarado, mas o congelamento ainda não é aplicado (nenhuma topologia do Marco 2 o usa).

### Adam e sparsity

- Um `NC_AdamState` por parâmetro treinável (`m`, `v` e contador de passo), criado na primeira chamada de `nc_train_backprop_step` e reutilizado depois.
- Atualização com correção de viés: `m̂ = m/(1-β₁ᵗ)`, `v̂ = v/(1-β₂ᵗ)`, `θ -= lr·m̂/(√v̂ + ε)`.
- A SAE injeta a penalidade L1 diretamente no gradiente da ativação oculta (`β·sinal(h)/|h|`) **antes** de rodar a fita, para que ela se propague aos pesos. `sparsity_target` fica registrado no grafo para uso futuro.

### Zoo

| Fábrica | Topologia |
| :-- | :-- |
| `nc_zoo_create_perceptron` | `INPUT → OUTPUT` |
| `nc_zoo_create_ff` | `INPUT → HIDDEN(ReLU) → OUTPUT` |
| `nc_zoo_create_dff` | `INPUT → H≥1 HIDDEN(ReLU) → OUTPUT` |
| `nc_zoo_create_ae` | `INPUT → HIDDEN → MATCH_INPUT_OUTPUT` |
| `nc_zoo_create_dae` | `NOISY_INPUT → HIDDEN → MATCH_INPUT_OUTPUT` |
| `nc_zoo_create_sae` | AE com `sparsity_beta` e `sparse_cell` |
| `nc_zoo_create_drn` | DFF com skips de identidade entre camadas de mesma largura |

## 4. Decisões de implementação e desvios do plano

1. **`nc_autograd_record` leva o ponteiro da fita e `params`.** O esboço do plano mostra quatro argumentos sem fita; sem ela não é possível suportar múltiplas fitas nem passar a máscara de dropout ou o alvo do MSE. A semântica pretendida foi preservada.
2. **Fita na arena persistente.** A alternativa — alocá-la na scratch — exigiria realocá-la a cada iteração. Os nós são reaproveitados via `reset`.
3. **Entrada achatada para 2D.** O exemplo do plano usa tensores 4D; o autograd do Marco 2 trabalha com matrizes `[batch, features]`. O grafo converte qualquer entrada contígua para esse formato, preservando compatibilidade com o exemplo conceitual.
4. **`nc_compute_mse_loss` e `nc_autograd_record_mse` são separados.** Assim é possível inspecionar a perda sem poluir a fita (útil em avaliação).
5. **Perceptron puro não resolve XOR.** O exemplo `xor_perceptron.c` usa uma FF com camada oculta e documenta o motivo.

## 5. Validação

```
make test  → arena 357/357 · tensor 234124/234124 · autograd 44/44 · zoo 47/47
make asan  → ASan + UBSan sem erros nem vazamentos nos quatro binários
```

Destaques: gradiente analítico de `matmul → add → ReLU` comparado com diferenças finitas para todos os elementos de `W` e `b`; derivadas de ativações, dropout e MSE; estrutura das 7 arquiteturas; treino de XOR com perda final ~0; reconstrução AE/DAE/SAE/DRN com queda de perda em 1500 épocas.

Regressões da revisão independente: shapes incompatíveis em `nc_nn_matmul`/`nc_nn_add` são rejeitados sem gravar nó; 1000 gravações de MSE não consomem a model arena; `tape == NULL` é rejeitado; acumulação de gradiente com a mesma entrada somada duas vezes no DRN foi conferida por diferenças finitas; entrada não-contígua é rejeitada pelo grafo.

## 6. Limitações / próximos marcos

- Células probabilísticas, spiking, recorrentes, memória/kernel e `im2col` ficam para os Marcos 3–5.
- O estado do RNG e os gradientes são globais por grafo: não há paralelismo de threads.
- `nc_render` (SVG) chega no Marco 6.
