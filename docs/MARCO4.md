# Marco 4 — Motores Temporais e Memórias Recorrentes

Detalhes do desdobramento temporal, do BPTT e das quatro arquiteturas recorrentes.

## 1. Escopo entregue

| Item do plano | Entrega |
| :-- | :-- |
| Unrolling temporal | `nc_sequence.h/.c` — `nc_sequence_forward()` |
| BPTT | fita única com todos os passos + `nc_sequence_backward()` |
| Célula RNN | `NC_CELL_RECURRENT` |
| Célula LSTM | `NC_CELL_MEMORY` (4 portes) |
| Célula GRU | `NC_CELL_DIFFERENT_MEMORY` (3 portes) |
| NTM | `NC_CELL_DIFFERENT_MEMORY` (2 projeções) com memória externa |
| Zoo | `nc_zoo_create_{rnn,lstm,gru,ntm}` |
| Testes | `tests/test_sequence.c` |
| Exemplos | `examples/lstm_sequence.c`, `examples/ntm_demo.c` |

## 2. Operações diferenciáveis adicionadas

- `nc_nn_mul` — produto elemento a elemento (Hadamard), com `dA += dY·B` e `dB += dY·A`.
- `nc_nn_scale` — multiplicação por escalar (gradiente escalado).
- `nc_nn_softmax` — normalização exponencial com o Jacobiano padrão no backward.

Sem elas, as equações de portes LSTM/GRU e o endereçamento da NTM não seriam diferenciáveis pela fita existente.

## 3. Unrolling e BPTT

`nc_sequence_forward(graph, x_seq, T, scratch)` recebe `x_seq [T,N,F]`, cria uma view 2D `[N,F]` por passo e chama `nc_graph_step()` sem resetar a fita. As ativações de cada passo ficam vivas na scratch arena até o backward. O estado recorrente (`h`, `c`, banco de memória) vive na célula e é atualizado a cada passo.

Como cada passo registra seus nós na **mesma** fita, `nc_autograd_backward()` propaga o erro desde o último passo até o primeiro — o BPTT surge naturalmente, sem um mecanismo separado. `nc_sequence_backward()` apenas zera os gradientes acumulados e dispara a fita.

`nc_sequence_reset_state()` deve ser chamado no início de cada sequência. A scratch arena também deve ser resetada pelo chamador a cada iteração de treino.

## 4. Células

- **RNN** — `h_t = tanh(x_t·W_xh + h_{t-1}·W_hh + b)`.
- **LSTM** — quatro portes (`i`, `f`, `o`, candidato) com pesos próprios de entrada/recorrência/viés; `c_t = f·c_{t-1} + i·c~` e `h_t = o·tanh(c_t)`.
- **GRU** — portes `z` e `r`, candidato `c~ = tanh(x·W + r·(h·W_hh) + b)` e `h_t = h_{t-1} + z·(c~ - h_{t-1})`.
- **NTM** — chave `k = tanh(x·W_key)`, similaridade `M·kᵀ`, pesos de atenção por softmax, leitura `wᵀ·M`, escrita `M ← M + w·tanh(x·W_write)`. O endereçamento é por conteúdo e o caminho de leitura é diferenciável; a escrita atualiza o estado sem gradientes persistentes entre passos.

Cada célula mantém seus estados na union `NC_Cell.state`. `nc_graph_ensure_adam_states()` registra todos os pesos de portes como parâmetros treináveis.

## 5. Decisões e limitações

- Pesos de portes são tensors separados (não fatias de uma matriz maior), o que evita um operador de slice e mantém o gradiente direto.
- A NTM usa produto escalar não normalizado como similaridade; normalização cosine completa ficaria para uma versão futura.
- Batch > 1 não é suportado na NTM nesta versão (documentado em `ntm_forward`).
- Células probabilísticas/spiking e o renderizador SVG permanecem para o Marco 5/6.

## 6. Validação

```
make test → arena 357/357 · tensor 234124/234124 · autograd 62/62 · zoo 65/65
            spatial 789/789 · sequence 587/587
make asan → seis binários sem erros nem vazamentos
```

Destaques: gradiente de `W_xh` e `W_hh` da RNN conferido por diferenças finitas ao longo de 4 passos; as quatro arquiteturas com forward/backward e gradientes não nulos; treino de GRU em cópia com atraso com perda de `1.29` para `0.00`.
