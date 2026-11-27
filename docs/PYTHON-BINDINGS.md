# Bindings Python (ctypes) — nova iniciativa

## 1. Decisões governantes

| Decisão | Escolha | Consequência |
| :-- | :-- | :-- |
| Mecanismo | `ctypes` | Carrega `libnn_shared.so` em runtime; zero dependências novas, sem compilação no install. |
| Superfície | API Pythonic de alto nível | Classes `Arena`, `Tensor`, `Graph`, `Adam`; ponteiros C escondidos. |
| Distribuição | Dentro do repo | `python/libnn/` com `pyproject.toml`; `make lib && pip install -e python`. Sem PyPI. |
| Fatia da API | Ciclo essencial | Construtores do zoo, forward, treinadores e render. Tensores de baixo nível ficam para depois. |
| Arenas | Context manager, grafo empresta | Fechar `with Arena` com grafo vivo gera `LibNNError`. |
| Erros | Exceções para falhas | `NULL`/`false` do C viram exceções; `None` só para ausência legítima. |

## 2. Estrutura

```
python/
├── pyproject.toml        # pacote libnn 0.1.0, setuptools, sem dependências
├── README.md             # instalação, uso, API
├── libnn/
│   ├── __init__.py       # API Pythonic
│   └── _core.py          # carga da .so, structs ctypes, assinaturas
├── tests/
│   └── test_bindings.py  # 10 testes
└── examples/
    └── xor.py            # exemplo completo: treino + SVG
```

A biblioteca compartilhada `libnn_shared.so` é gerada na raiz por `make lib`
(junto com `libnn.a`). O nome diferente de `libnn.so` evita colisão com o
pacote Python `libnn` — o Python tentaria carregar o `.so` como módulo de
extensão.

## 3. Instalação e uso

```sh
make lib
python3 -m pip install -e python      # ou apenas sys.path.insert(0, "python")
python3 python/examples/xor.py
```

```python
import libnn as nn

nn.seed(7)
with nn.Arena(16 << 20) as model, nn.Arena(4 << 20) as scratch:
    net = nn.Graph.ff(model, 2, 8, 1)
    opt = nn.Adam(lr=0.05)
    for _ in range(300):
        scratch.reset()
        x = nn.Tensor(scratch, [4,2], data=[[0,0],[0,1],[1,0],[1,1]])
        y = nn.Tensor(scratch, [4,1], data=[[0],[1],[1],[0]])
        pred = net.forward(x, scratch)
        loss = net.train_backprop(pred, y, opt, scratch)
    net.render_svg("xor.svg")
    net.destroy()
```

## 4. Cobertura da API

- **Arena**: `bytes_used`, `bytes_available`, `reset()`, `close()`, context manager.
- **Tensor**: construtor com `data`, `tolist()`, `fill_zeros()`, `fill_random_normal()`.
- **Graph**: 13 fábricas do zoo no ciclo essencial — `perceptron`, `ff`, `ae`, `dae`, `vae`, `hopfield`, `rbm`, `kohonen`, `elm`, `svm`, `rnn`, `lstm`, `gru`, `ntm`.
- **Treinadores**: `train_backprop`, `contrastive_divergence`, `hebbian`, `stdp`, `ridge`, `kohonen_step`, `hinge`.
- **Render**: `render_svg(path)` e `to_svg()`.
- **RNG**: `seed()` / `set_seed()`.

## 5. Validação

```
make lib                          → libnn.a + libnn_shared.so
python3 -m pytest python/tests -q → 10 passed
python3 python/examples/xor.py    → loss 0.59463 → 0.00000, SVG gerado
make test / make asan             → suíte C intacta (8 binários)
```

## 6. Limitações

- Tensores de baixo nível (matmul, conv, im2col) não têm wrapper Python nesta versão.
- `NC_Graph` é parcialmente opaco no ctypes: apenas os campos até `tape` são
  expostos, pois o wrapper precisa do tape para gravar a perda MSE.
- O tensor retornado por `forward` vive na arena scratch; chamadas a
  `scratch.reset()` o invalidam. O wrapper não rastreia esse ciclo de vida.
- A API cobre 14 fábricas do zoo; as demais (DCN, DN, DCIGN, DBN, DRN, DFF,
  LSM, ESN, RBF, Boltzmann, Markov) podem ser adicionadas incrementalmente.

## 7. Próximos passos

1. Cobrir as fábricas restantes do zoo.
2. Expor tensores de baixo nível e operações de álgebra linear.
3. Gerenciar o ciclo de vida do tensor de saída em relação ao `scratch.reset()`.
