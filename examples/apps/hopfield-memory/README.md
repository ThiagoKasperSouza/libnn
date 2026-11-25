# Memória associativa Hopfield

## O que faz

Armazena um padrão binário (±1) numa matriz de pesos simétrica pela **regra Hebbiana** e depois o recupera a partir de uma versão ruidosa.

## Arquitetura

```
BACKFED_INPUT(6, tanh) → MATCH_INPUT_OUTPUT(6)
```

Construtor: `nc_zoo_create_hopfield(&model, 6)`. Os pesos começam pequenos e aleatórios; cada passada Hebbiana reforça `W += lr · x · xᵀ`.

## Dados

Padrão de 6 sinais `[+1,+1,-1,-1,+1,-1]`, apresentado com ruído multiplicativo de 30% na recuperação.

## Como treinar

```sh
make run_apps APP=hopfield-memory
```

O treino é não-supervisionado:

```c
nc_arena_reset(&scratch);
nc_graph_forward(net, pattern, &scratch);  // ativa o padrão
nc_train_hebbian_step(net, 0.5f);           // W += lr · x · xᵀ
```

Repita 20 vezes. A recuperação é um único `nc_graph_forward` com o padrão ruidoso: a saída `tanh(W·x_ruidoso)` deve ter os mesmos sinais do original.

### Hiperparâmetros

| Parâmetro | Valor |
| :-- | :-- |
| Passadas Hebbianas | 20 |
| lr | 0.5 |
| Ruído | 30% |
| Semente | 11 |

## Saída esperada

Os seis sinais recuperados com os mesmos sinais do padrão original (acerto 6/6).
