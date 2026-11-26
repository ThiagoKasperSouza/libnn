# Renderização SVG de grafos neurais

## O que faz

Exporta a topologia de qualquer `NC_Graph` para um arquivo `.svg` vetorial no estilo do infográfico *"A mostly complete chart of Neural Networks"* (Fjodor van Veen, Asimov Institute). Cada célula é desenhada como um círculo colorido conforme seu tipo, com marcadores de borda/triângulo para as variantes da legenda, e as arestas são linhas com setas indicando o fluxo de dados.

## Arquitetura

A aplicação renderiza dois grafos:
- `nc_zoo_create_dae(&model, 8, 4, 0.2f)` — Denoising Autoencoder (8 entradas, 4 ocultas)
- `nc_zoo_create_lstm(&model, 4, 6, 2)` — LSTM (4 entradas, 6 ocultas, 2 saídas)

## Como usar

```sh
make run_apps APP=render-graph
```

A API é simples:

```c
int nc_render_svg(const NC_Graph *graph, const char *filename);
```

Retorna 0 em sucesso, -1 em erro. Existe também `nc_render_svg_string` para renderização em buffer de memória (útil para testes e uso em servidores web).

### Paleta de cores

| Tipo de célula | Cor | Hex |
| :-- | :-- | :-- |
| Entrada (Input, Backfed, Noisy) | Amarelo | `#FFD700` |
| Oculta (Hidden, Probabilistic, Spiking) | Verde | `#90EE90` |
| Saída (Output, Match I/O) | Vermelho | `#FF6347` |
| Recorrente (Recurrent, Memory, Different Memory) | Azul | `#87CEEB` |
| Convolucional (Kernel, Conv/Pool) | Rosa | `#FFB6C1` |

Variantes (com borda, com triângulo) recebem marcadores adicionais conforme a legenda do infográfico.

## Saída esperada

Dois arquivos SVG (`dae_architecture.svg` e `lstm_architecture.svg`) que podem ser abertos em qualquer navegador ou editor de imagens vetoriais.

## Como adaptar

- Passe qualquer `NC_Graph*` criado pelas fábricas do `nc_zoo` para `nc_render_svg`.
- Use `nc_render_svg_string` para obter o SVG em memória (ex.: servir via HTTP).
- A função lê a topologia (células e arestas) do grafo; não requer pesos treinados.
