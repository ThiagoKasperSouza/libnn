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
- **Graph**: **todas as 27 arquiteturas do zoo** estão disponíveis:

  | Grupo | Fábricas Python |
  | :-- | :-- |
  | Feedforward | `perceptron`, `ff`, `dff`, `drn` |
  | Autoencoders | `ae`, `dae`, `sae`, `vae`, `dcign` |
  | Energia | `boltzmann`, `markov_chain`, `rbm`, `dbn`, `hopfield` |
  | Competitivo | `kohonen`, `rbf`, `svm` |
  | Reservatório | `elm`, `esn`, `lsm` |
  | Recorrentes | `rnn`, `lstm`, `gru`, `ntm` |
  | Convolucionais | `dcn`, `dn` |
  | Adversarial | `GAN` (classe separada, com `train_step`) |

  Fábricas com dimensões ocultas (`dff`, `drn`, `dbn`) recebem uma sequência
  de inteiros, por exemplo `Graph.dff(model, 4, [8, 6], 2)`.
- **Sequências**: `sequence_forward(x_seq, steps, scratch)` → lista de tensores, `sequence_backward()`, `reset_state()`.
- **Treinadores**: `train_backprop`, `contrastive_divergence`, `hebbian`, `stdp`, `ridge`, `kohonen_step`, `hinge`, `GAN.train_step`.
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
- `nc_sequence_forward` retorna `NC_Tensor **`; o binding usa
  `POINTER(POINTER(NC_Tensor))` e a API devolve uma lista de `Tensor`.
- O tensor retornado por `forward` vive na arena scratch. `scratch.reset()`
  invalida esses tensores e qualquer acesso posterior levanta `LibNNError`
  (em vez de use-after-free silencioso). Use o tensor antes do próximo reset.
- Não são expostos em Python: `nc_arena_alloc`, `nc_arena_contains`,
  `nc_graph_step`, `nc_graph_zero_grads`, `nc_graph_ensure_adam_states`,
  `nc_graph_cell_count`, operações `nc_nn_*`, `nc_matmul` e
  `nc_tensor_flatten`. As 27 fábricas do zoo, por outro lado, estão todas
  cobertas.

## 7. Próximos passos

1. Expor tensores de baixo nível e operações de álgebra linear.
3. Gerenciar o ciclo de vida do tensor de saída em relação ao `scratch.reset()`.
