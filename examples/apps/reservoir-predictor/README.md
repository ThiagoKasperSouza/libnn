# Reservatório com regressão ridge (ELM/ESN/LSM)

## O que faz

Treina uma **Extreme Learning Machine**: a camada oculta é aleatória e congelada; apenas a camada de saída é resolvida analiticamente pela pseudoinversa ridge. É o mesmo mecanismo usado por ESN e LSM.

## Arquitetura

```
INPUT(2) → HIDDEN(12, aleatória e congelada) → OUTPUT(1)
```

Construtor: `nc_zoo_create_elm(&model, 2, 12, 1)`. Os pesos de saída são preenchidos pelo treinador ridge.

## Dados

Quatro pontos 2D com alvo `y = x₀ + x₁`.

## Como treinar

```sh
make run_apps APP=reservoir-predictor
```

Não há loop de epoch: basta um forward para obter as features do reservatório e uma chamada analítica:

```c
nc_graph_forward(elm, x, &scratch);
nc_train_reservoir_ridge_regression(elm, elm->cells[1].activity, y, 1e-3f);
```

Internamente o treinador resolve `(ΦᵀΦ + αI)W = ΦᵀY` com Gauss-Jordan e pivoteamento.

### Hiperparâmetros

| Parâmetro | Valor |
| :-- | :-- |
| Neurônios do reservatório | 12 |
| α (ridge) | 1e-3 |
| Semente | 9 |

## Saída esperada

Previsões iguais aos alvos (`0, 1, 1, 2`) com MSE próximo de zero.

## Como adaptar

- `nc_zoo_create_esn` troca a camada oculta por um reservatório recorrente congelado; use as ativações ao longo do tempo como `states`.
- `nc_zoo_create_lsm` usa neurônios LIF; as spikes viram `states` (com `nc_train_stdp_step` opcional).
- Aumente `neurons` ou ajuste `alpha` para controlar overfitting.
