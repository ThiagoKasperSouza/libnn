# Marco 5 — Modelos de Energia, Spiking e Reservatórios

## 1. Escopo entregue

| Item do plano | Entrega |
| :-- | :-- |
| Divergência contrastiva CD-k | `nc_train_contrastive_divergence()` |
| Hebbiano e STDP | `nc_train_hebbian_step()`, `nc_train_stdp_step()` |
| LIF spiking | `NC_CELL_SPIKING_HIDDEN` com integração, limiar, reset e refratariedade |
| Ridge / pseudoinversa | `nc_train_reservoir_ridge_regression()` |
| Kohonen BMU | `nc_train_kohonen_bmu_step()` |
| Hinge loss (SVM) | `nc_train_kernel_hinge_loss()` |
| 12 arquiteturas | MC, HN, BM, RBM, DBN, VAE, LSM, ELM, ESN, KN, RBF, SVM |
| Testes | `tests/test_energy.c` |
| Aplicações | `hopfield-memory`, `rbm-cd`, `reservoir-predictor`, `kohonen-map` |

## 2. Treinadores não-diferenciáveis

- **CD-k**: amostra `h ~ p(h|v)` na fase positiva, reconstrói `v' ~ p(v|h)`, calcula `h' ~ p(h|v')` e acumula `ΔW ∝ v hᵀ - v' h'ᵀ`, com vieses visível (`recurrent_weights`) e oculto (`bias`). As amostras vêm do RNG determinístico da libnn; os dados binários são tratados como probabilidades.
- **Hebbian**: `ΔW += lr · x_pre · y_pos`, usando a aresta de entrada; para `BACKFED_INPUT` (Hopfield) usa o próprio padrão como pré e pós.
- **STDP**: para neurônios LIF, `ΔW += lr · sinal(Δt) · exp(-|Δt|/τ)` quando há spike pré e pós; `last_pre_time` e `last_post_time` são mantidos no estado da célula.
- **Ridge**: resolve `(ΦᵀΦ + αI)W = ΦᵀY` com Gauss-Jordan e pivoteamento parcial; memória temporária alocada com `malloc` e liberada ao fim.
- **BMU**: procura o protótipo mais próximo e move a vizinhança gaussiana por `W_i += lr · exp(-d²/2σ²) · (x - W_i)`.
- **Hinge**: atualização de margem linear `w += lr·C·y·x` para amostras com `y·(w·x) < 1`.

## 3. Células e grafos

- `PROBABILISTIC_HIDDEN` com `gate_weights[0/1]` implementa o truque da reparametrização: `mu`, `log_var`, `eps ~ N(0,1)` e `z = mu + eps·exp(log_var/2)`, todos diferenciáveis via fita. Sem esses pesos, mantém o comportamento determinístico do DCIGN.
- `SPIKING_HIDDEN` integra `v = decay·v + W·x`, dispara spike quando `v >= limiar`, zera em `v_reset` e respeita passos refratários.
- `BACKFED_INPUT` soma feedback recorrente à entrada antes da projeção.
- `KERNEL` com pesos 2D calcula features RBF `exp(-||x-c||²)`; com pesos 4D mantém convolução.
- `HIDDEN` com `competitive=true` é um mapa de Kohonen: os pesos são protótipos, não matriz de projeção.
- `NC_Graph` guarda `last_input`, usado pelo Hebbiano e pela recuperação Hopfield.

## 4. Arquiteturas

| Fábrica | Treinador típico |
| :-- | :-- |
| `markov_chain`, `boltzmann`, `rbm`, `dbn` | CD-k |
| `hopfield` | Hebbiano |
| `vae` | Adam (reconstrução; KL simplificada) |
| `lsm` | STDP + ridge |
| `elm`, `esn`, `rbf` | Ridge |
| `kohonen` | BMU |
| `svm` | Hinge loss |

## 5. Limitações

- O endereçamento RBF usa `exp(-||x-c||²)` sem largura configurável.
- O SVM treinado é linear; o módulo `KERNEL` fornece as features RBF para serem usadas antes do hinge.
- A KL da VAE não é adicionada explicitamente à perda (apenas reconstrução).
- A esn/lsm em mini-batches dependem de como as ativações são coletadas pelo chamador.

## 6. Validação

```
make test → arena 357/357 · tensor 234124/234124 · autograd 62/62 · zoo 65/65
            spatial 789/789 · sequence 615/615 · energy 40/40
make asan → sete binários sem erros nem vazamentos
```

Destaques: CD-k altera os pesos da RBM (ΔW L1 ≈ 2.83), Hopfield recupera os seis sinais corretamente, STDP altera a matriz LIF, ridge atinge MSE ≈ 0 no ELM e RBF, BMU move os protótipos para as amostras e o hinge separa as duas classes lineares.
