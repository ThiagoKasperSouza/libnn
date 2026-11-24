# Noções básicas de tensores

## O que esta aplicação faz

Não é um modelo treinável: é uma demonstração da infraestrutura do Marco 1. Mostra o ciclo de vida completo da memória e das operações numéricas:

- duas arenas (`model` e `scratch`);
- criação e preenchimento de tensores com RNG determinístico;
- multiplicação de matrizes, incluindo o caso em que a saída reaproveita o buffer de uma entrada (aliasing);
- convolução 2D e max pooling;
- reset O(1) da scratch arena.

Arquivo: `main.c`.

## Como executar

```sh
make run_apps APP=tensor-basics
```

## Como "treinar" a rede

Esta aplicação não treina. Ela estabelece os conceitos usados por todas as outras:

- os **pesos** ficam na `model_arena` e sobrevivem a múltiplas iterações;
- as **ativações** ficam na `scratch_arena`, que deve ser resetada no início de cada passo;
- `nc_rng_seed()` torna a inicialização reproduzível.

Para ver um loop de treino completo, continue em [`../xor-classifier/README.md`](../xor-classifier/README.md).

## Saída esperada

O programa imprime o resultado de `W @ x`, o mesmo produto com `C == A`, os mapas de convolução/pooling e a quantidade de bytes da scratch antes e depois do reset (que volta a zero).
