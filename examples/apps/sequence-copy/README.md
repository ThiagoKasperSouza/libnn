# Cópia de sequência com LSTM

## O que esta aplicação faz

Treina uma LSTM numa tarefa que exige **memória temporal**: prever, em cada passo `t`, o valor da entrada no passo `t-1` (cópia com atraso 1). Uma rede feedforward não consegue resolver isso sem acesso ao histórico.

## Arquitetura

```
INPUT(2) → MEMORY/LSTM(8) → OUTPUT(1)
```

Construtor: `nc_zoo_create_lstm(&model, 2, 8, 1)`.

Quatro portes (entrada, esquecimento, saída e candidato) com pesos próprios de entrada, recorrência e viés. O estado `h` e o estado de célula `c` são atualizados a cada passo.

## Dados

Sequência de 6 passos com entradas 2D:

```
x = [1,0] [0,1] [1,1] [0,0] [1,0] [0,1]
y = [ 0  ] [ 1  ] [ 0  ] [ 1  ] [ 0  ] [ 1  ]   (y[t] = x[t-1][0])
```

## Como treinar

```sh
make run_apps APP=sequence-copy
```

O gradiente flui do último passo até o primeiro porque todos os passos são gravados na **mesma fita**:

```c
nc_arena_reset(&scratch);
nc_sequence_reset_state(net);              // limpa h e c antes da sequência
NC_Tensor **outs = nc_sequence_forward(net, x, T, &scratch);
for (size_t t = 0; t < T; ++t)
    nc_autograd_record_mse(net->tape, outs[t], target_t);
nc_sequence_backward(net);                 // BPTT
nc_train_backprop_step(net, &opt);          // Adam uma vez por sequência
```

### Hiperparâmetros

| Parâmetro | Valor |
| :-- | :-- |
| Épocas (sequências) | 600 |
| Passos por sequência | 6 |
| Neurônios ocultos | 8 |
| Taxa de aprendizado | 0.01 |
| Semente RNG | 2026 |

## Saída esperada

A loss cai de ~2.87 para ~0.000; as previsões finais são `0.000 1.000 0.000 1.000 0.000 1.000`, idênticas aos alvos.

## Como adaptar

- Troque `nc_zoo_create_lstm` por `nc_zoo_create_gru` ou `nc_zoo_create_rnn` para comparar arquiteturas.
- Aumente o atraso da tarefa: o LSTM precisa reter informação por mais passos.
- Defina `T` maior, desde que a fita suporte os nós extras (`ZOO_TAPE_CAPACITY = 8192`; um transbordo gera erro explícito).
