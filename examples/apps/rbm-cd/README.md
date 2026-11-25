# RBM com divergência contrastiva

## O que faz

Treina uma **Restricted Boltzmann Machine** com CD-k: alterna amostragem de Gibbs ("fase positiva" com os dados, "fase negativa" reconstruindo a partir da amostra) e atualiza pesos e vieses pela diferença das duas fases.

## Arquitetura

```
INPUT(6) → PROBABILISTIC_HIDDEN(3)
```

Construtor: `nc_zoo_create_rbm(&model, 6, 3)`. Os vieses visível e oculto ficam em `recurrent_weights` e `bias` da célula oculta.

## Dados

Quatro padrões binários de 6 bits gerados em memória.

## Como treinar

```sh
make run_apps APP=rbm-cd
```

Cada chamada executa um passo CD-k completo:

```c
nc_train_contrastive_divergence(rbm, data, 1, 0.05f);
```

O treinador usa `nc_rng_next_double()` para amostrar as ativações binárias, portanto é determinístico dado `nc_rng_seed()`.

### Hiperparâmetros

| Parâmetro | Valor |
| :-- | :-- |
| Passos CD | 200 |
| k (fases Gibbs) | 1 |
| lr | 0.05 |
| Unidades ocultas | 3 |
| Semente | 7 |

## Saída esperada

A norma de `W` estabiliza ao longo dos passos, indicando que os pesos aprenderam estatísticas dos padrões.

## Como adaptar

- Aumente `k` para uma aproximação mais precisa da divergência (mais custosa).
- Use `nc_zoo_create_dbn` para empilhar várias RBMs.
- Troque os dados binários por vetores contínuos em `[0,1]` (a amostragem ainda é binária).
