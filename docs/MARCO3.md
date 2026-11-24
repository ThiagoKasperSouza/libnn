# Marco 3 — Operações Espaciais e Grafos Convolucionais

Detalhes das operações 2D otimizadas, da integração com o autograd e das quatro arquiteturas adicionadas.

## 1. Escopo entregue

| Item do plano | Entrega |
| :-- | :-- |
| `im2col + GEMM` para convolução | `nc_conv2d_im2col()` em `src/nc_tensor.c` |
| Convolução transposta otimizada | `nc_conv2d_transpose_im2col()` (GEMM + scatter) |
| Max/Average pooling | `nc_avg_pool2d()`, `nc_nn_max_pool2d()`, `nc_nn_avg_pool2d()` |
| Backward espacial | backward de conv, conv transposta e pooling na fita |
| Células convolucionais | `NC_CELL_KERNEL`, `NC_CELL_CONV_OR_POOL`, latente probabilístico determinístico |
| Zoo | `nc_zoo_create_dcn`, `nc_zoo_create_dn`, `nc_zoo_create_dcign`, `nc_zoo_create_gan` |
| Treino adversarial | `nc_train_gan_step()` + `nc_graph_backward_no_zero()` |
| Testes | `tests/test_spatial.c` |
| Exemplos | `examples/dcn_synthetic.c`, `examples/gan_toy.c` |

## 2. im2col + GEMM

`nc_conv2d_im2col` monta uma matriz coluna `[C_in*kH*kW, N*oH*oW]` cujas colunas são as janelas da entrada (com zeros no padding). O kernel é visto como `[C_out, C_in*kH*kW]` e o resultado `[C_out, N*oH*oW]` é obtido com `nc_matmul` (que já usa tiling). Um último passo espalha o resultado para o layout `[N, C_out, oH, oW]`. O buffer im2col é alocado na `scratch_arena`, que é reaproveitada via `reset` entre iterações.

A transposta usa o caminho inverso: o kernel é reorganizado como `[C_in, C_out*kH*kW]`, a entrada como `[C_in, N*H*W]`, e um GEMM produz `[C_out*kH*kW, N*H*W]`, espalhado de volta com acumulação (`col2im`).

A implementação direta do Marco 1 foi mantida para referência e comparação nos testes.

## 3. Autograd espacial

Novos nós na fita: `NC_OP_CONV2D`, `NC_OP_CONV_TRANSPOSE`, `NC_OP_MAXPOOL`, `NC_OP_AVGPOOL`.

- **conv2d**: `dK` por correlação entre `dy` e `im2col(input)`; `dInput` por convolução transposta de `dy` com o kernel.
- **conv transposta**: `dInput = conv2d(dy, kernel, stride, padding)`; `dKernel` por correlação direta.
- **max pooling**: um vetor de índices `size_t` na scratch guarda o offset exato do máximo, evitando perda de precisão de offsets em `float`.
- **avg pooling**: o gradiente é dividido por `k²` e distribuído pelas posições da janela.

O indexamento genérico `flat_index()` do autograd foi estendido para qualquer `ndim`, corrigindo MSE e gradientes em tensores 4D.

## 4. Caminho 4D no grafo

Grafos com células `KERNEL`/`CONV_OR_POOL` preservam a entrada 4D. Quando uma célula densa consome um mapa de características, `nc_tensor_flatten()` converte `[N,C,H,W]` em `[N, C*H*W]`. Com `requires_grad=true` ele exige que o tensor original já tenha `grad` alocado e compartilha esse buffer, para que o erro volte até os kernels; sem `grad`, retorna `NULL` em vez de criar um buffer desconectado.

A célula `MATCH_INPUT_OUTPUT` sem pesos atua como pass-through (saída de DN/DCIGN). A célula `PROBABILISTIC_HIDDEN` reserva `mu`, `log_var` e `eps` e atualmente passa `mu = x` — amostragem e KL pertencem ao Marco 5.

## 5. Arquiteturas

| Fábrica | Topologia |
| :-- | :-- |
| `nc_zoo_create_dcn` | `INPUT [1,C,H,W] → CONV(8 filtros 3x3) → MAXPOOL 2x2 → OUTPUT denso` |
| `nc_zoo_create_dn` | `INPUT [1,C,H,W] → CONV_TRANSPOSE(2x2, stride 2) → MATCH` (upsample) |
| `nc_zoo_create_dcign` | `INPUT → CONV → latente determinístico → CONV_TRANSPOSE → MATCH` |
| `nc_zoo_create_gan` | par gerador `ruído → HIDDEN → dados` e discriminador `dados → HIDDEN → 1` |

`nc_train_gan_step` executa: (1) discriminador com reais (alvo 1) e falsos (alvo 0), com atualização; (2) gerador com o erro do discriminador (alvo 1), usando `nc_graph_backward_no_zero()` para preservar a semente de gradiente e atualizando somente o gerador. O objective é MSE em vez de BCE, para manter o SDK sem funções de perda adicionais.

## 6. Validação

```
make test  → arena 357/357 · tensor 234124/234124 · autograd 62/62 · zoo 65/65 · spatial 789/789
make asan  → cinco binários sem erros nem vazamentos
```

A suíte espacial compara im2col com a convolução direta (incluindo a transposta com `C_in=2, C_out=4` e kernel 3×3), verifica gradientes por diferenças finitas (conv, transposta, max/avg pooling) com `N>1` e `C_in>1`, confere o contrato do flatten e o gradiente do maxpool em múltiplos canais, verifica shapes das quatro arquiteturas, treina DCN/DN/DCIGN com perda decrescente e executa 200 passos adversariales conferindo que os pesos do gerador mudam e permanecem finitos.

## 7. Limitações

- O latente probabilístico ainda é determinístico; VAE/KL ficam para o Marco 5.
- Backward de convolução usa laços diretos (o forward é o caminho otimizado com GEMM).
- GAN usa MSE como objective adversarial.
- `NC_EDGE_FROZEN` e `nc_render` permanecem para marcos futuros.
