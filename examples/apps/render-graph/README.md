# Renderização SVG de grafos neurais

## O que faz

Exporta a topologia de qualquer `NC_Graph` para um arquivo `.svg` vetorial no estilo do infográfico *"A mostly complete chart of Neural Networks"* (Fjodor van Veen, Asimov Institute). Cada célula é desenhada como um círculo colorido conforme seu tipo, com marcadores de borda/triângulo para as variantes da legenda, e as arestas são linhas com setas indicando o fluxo de dados.

## Arquitetura

A aplicação renderiza dois grafos:
- `nc_zoo_create_dae(&model, 8, 4, 0.2f)` — Denoising Autoencoder (8 entradas, 4 ocultas)
- `nc_zoo_create_lstm(&model, 3, 4, 3)` — LSTM (3 entradas, 4 memórias, 3 saídas)

## Como usar

```sh
make run_apps APP=render-graph
```

A API é simples:

```c
int nc_render_svg(const NC_Graph *graph, const char *filename);
```

Retorna 0 em sucesso, -1 em erro. Existe também `nc_render_svg_string` para renderização em buffer de memória (útil para testes e uso em servidores web).

### Estética do infográfico

O SVG reproduz o estilo do guia *"A mostly complete chart of Neural Networks"*:

| Tipo de célula | Cor | Hex |
| :-- | :-- | :-- |
| Entrada (Input, Backfed, Noisy) | Amarelo | `#FFD700` |
| Oculta (Hidden, Probabilistic, Spiking) | Verde | `#33CC33` |
| Saída (Output, Match I/O) | Laranja | `#FF6600` |
| Recorrente (Recurrent, Memory, Different Memory) | Azul | `#3399FF` |
| Convolucional (Kernel, Conv/Pool) | Rosa | `#FF66B2` |

Cada camada é expandida em múltiplos neurônios com conexões totalmente conectadas. Variantes "com borda" têm um círculo concêntrico interno; variantes "com triângulo" têm um triângulo dentro da célula — idêntico à legenda do guia. A legenda completa com 13 variantes é incluída no rodapé.

## Saída esperada

Dois arquivos SVG (`dae_architecture.svg` e `lstm_architecture.svg`) que podem ser abertos em qualquer navegador ou editor de imagens vetoriais.

## Como adaptar

- Passe qualquer `NC_Graph*` criado pelas fábricas do `nc_zoo` para `nc_render_svg`.
- Use `nc_render_svg_string` para obter o SVG em memória (ex.: servir via HTTP).
- A função lê a topologia (células e arestas) do grafo; não requer pesos treinados.
