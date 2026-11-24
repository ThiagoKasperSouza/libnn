# Classificador XOR

## O que esta aplicação faz

Aprende a função lógica XOR, que **não é linearmente separável**: um Perceptron puro (sem camada oculta) não consegue resolvê-la. Por isso a aplicação usa uma rede **Feed Forward (FF)** com uma camada oculta ReLU.

## Arquitetura

```
INPUT(2) → HIDDEN(8, ReLU) → OUTPUT(1, linear)
```

Construtor: `nc_zoo_create_ff(&model, 2, 8, 1)`.

## Dados

Quatro amostras sintéticas, uma para cada combinação de bits:

```
x = [0,0] [0,1] [1,0] [1,1]
y = [  0] [  1] [  1] [  0]
```

## Como treinar

```sh
make run_apps APP=xor-classifier
```

O loop em `main.c` segue o padrão canônico da libnn:

```c
nc_arena_reset(&scratch);                        // 1. limpa ativações
NC_Tensor *pred = nc_graph_forward(net, x, &scratch);  // 2. forward
float loss = nc_compute_mse_loss(pred, y);       // 3. perda
nc_autograd_record_mse(net->tape, pred, y);      // 4. semente do gradiente
nc_graph_backward(net);                          // 5. backpropagation
nc_train_backprop_step(net, &opt);               // 6. Adam
```

### Hiperparâmetros

| Parâmetro | Valor |
| :-- | :-- |
| Épocas | 3000 |
| Taxa de aprendizado | 0.05 |
| β₁ / β₂ / ε (Adam) | 0.9 / 0.999 / 1e-8 |
| Semente RNG | 42 |

## Saída esperada

A loss cai de ~0.77 para ~0.000 ainda nas primeiras centenas de épocas, e as previsões finais são `0.000 1.000 1.000 0.000`, iguais aos alvos.

## Como adaptar

- Troque `8` por outro número de neurônios ocultos em `nc_zoo_create_ff`.
- Use `nc_zoo_create_dff` com várias camadas para um classificador mais profundo.
- Para classes múltiplas, aumente `output_dim` e use um one-hot como alvo.
