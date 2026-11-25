# Mapa de Kohonen (SOM)

## O que faz

Organiza 6 protótipos numa grade 1D para representar quatro padrões 2D. A cada amostra, encontra o **BMU** (best matching unit, menor distância euclidiana) e move os protótipos da vizinhança em direção à amostra.

## Arquitetura

```
INPUT(2) → HIDDEN competitiva(6 protótipos de 2 dimensões)
```

Construtor: `nc_zoo_create_kohonen(&model, 2, 6)`. Os pesos da célula oculta são os protótipos, atualizados diretamente pelo treinador (não por gradiente).

## Dados

Os quatro cantos do quadrado unitário: `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)`.

## Como treinar

```sh
make run_apps APP=kohonen-map
```

```c
nc_train_kohonen_bmu_step(som, sample, lr, radius);
```

O raio controla a vizinhança gaussiana em torno do BMU; `lr` escala o passo.

### Hiperparâmetros

| Parâmetro | Valor |
| :-- | :-- |
| Épocas (passagens pelos 4 padrões) | 200 |
| lr | 0.3 |
| radius | 2.0 |
| Neurônios | 6 |
| Semente | 42 |

## Saída esperada

Os protótipos convergem para perto dos quatro cantos; alguns neurônios especializam-se em cada padrão.

## Como adaptar

- Aumente o número de neurônios para um mapa mais fino.
- Diminua `radius` ao longo do treino (decaimento) para refinar a organização.
- Troque os padrões 2D por vetores de maior dimensão (ajuste `input_dim`).
