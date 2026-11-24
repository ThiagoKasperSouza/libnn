# Autoencoder denoising

## O que esta aplicação faz

Aprende a **reconstruir sinais limpos a partir de entradas ruidosas**. É o exemplo canônico de autoencoder: a rede comprime a entrada num código latente menor e depois a reconstrói, sendo treinada para minimizar o erro contra a versão original sem ruído.

## Arquitetura

```
NOISY_INPUT(8, σ=0.2) → HIDDEN(4, ReLU) → MATCH_INPUT_OUTPUT(8, linear)
```

Construtor: `nc_zoo_create_dae(&model, 8, 4, 0.2f)`.

A célula `NOISY_INPUT` injeta ruído gaussiano **apenas no forward**; o gradiente passa por ela sem alterar os dados de entrada.

## Dados

Quatro padrões esparsos de 8 dimensões, cada um com um valor 1.0 e um valor 0.5 em posições fixas. O alvo é o próprio padrão original (sem ruído).

## Como treinar

```sh
make run_apps APP=denoising-autoencoder
```

A cada época o código:

1. reseta a scratch arena;
2. faz `nc_graph_forward` (que aplica ruído novo a cada passo);
3. calcula `nc_compute_mse_loss(x, reconstrução)`;
4. semeia o gradiente com `nc_autograd_record_mse`;
5. roda `nc_graph_backward` e `nc_train_backprop_step`.

### Hiperparâmetros

| Parâmetro | Valor |
| :-- | :-- |
| Épocas | 2000 |
| Taxa de aprendizado | 0.02 |
| σ do ruído | 0.2 |
| Dimensão latente | 4 |
| Semente RNG | 123 |

## Saída esperada

A loss cai de ~0.27 para ~0.005. Na reconstrução final, os valores principais aproximam-se de 1.0 e 0.5, enquanto as posições zero ficam próximas de zero.

## Como adaptar

- Aumente `noise_level` para um treino mais difícil.
- Em vez de ruído gaussiano, use dropout definindo `state.noisy.dropout_rate` na célula `NOISY_INPUT`.
- Troque `nc_zoo_create_dae` por `nc_zoo_create_ae` para autoencoder sem ruído, ou `nc_zoo_create_sae` para penalidade de sparsity.
