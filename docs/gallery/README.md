# Galeria interativa das 27 arquiteturas

Página estática que mostra o diagrama SVG de cada uma das 27 redes do
infográfico *A mostly complete chart of Neural Networks*, com filtros por
família e painel de detalhe (células, treinador e código de criação em Python).

## Como abrir

Basta abrir `docs/gallery/index.html` no navegador. Não precisa de servidor,
Python nem build — os SVGs estão commitados em `svgs/`.

```sh
xdg-open docs/gallery/index.html
```

## Como regenerar

Os SVGs e o arquivo `data.js` são gerados pelos bindings Python a partir da
topologia real de cada `NC_Graph`:

```sh
cd /home/thiago/Documents/libnn
make lib
python3 python/examples/generate_gallery.py
```

O script usa `nc_render_svg_string()` via ctypes, portanto qualquer mudança no
renderizador C se reflete na galeria após a regeneração.

## Estrutura

```
docs/gallery/
├── index.html          # página com filtros e painel de detalhe
├── data.js             # metadados das 27 arquiteturas (gerado)
├── README.md
└── svgs/
    ├── P.svg  FF.svg  RBF.svg  DFF.svg ...
    └── (27 arquivos)
```

## Famílias

| Família | Arquiteturas |
| :-- | :-- |
| Feedforward | P, FF, DFF, DRN |
| Autoencoders | AE, DAE, SAE, VAE, DCIGN |
| Modelos de energia | MC, HN, BM, RBM, DBN |
| Convolucionais | DCN, DN |
| Recorrentes e memória | RNN, LSTM, GRU, NTM |
| Reservatórios | LSM, ELM, ESN |
| Competitivos e kernel | KN, RBF, SVM |
| Generativos | GAN |

## Notas

- O GAN é renderizado pelo seu **gerador**; o discriminador é um segundo grafo.
- Cores e marcadores seguem a legenda do infográfico: amarelo (entrada),
  verde (oculta), laranja (saída), azul (estado) e rosa (convolucional).
- Cada célula do grafo é expandida em neurônios; conexões são totalmente
  conectadas entre camadas adjacentes.
