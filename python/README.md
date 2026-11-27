# libnn — bindings Python

Wrapper **ctypes** sobre a biblioteca compartilhada do SDK `libnn`. O núcleo C
continua sendo a única implementação dos algoritmos; aqui há apenas uma camada
fina de ligação com uma API Pythonic.

## Instalação

A biblioteca C precisa ser compilada antes:

```sh
cd /home/thiago/Documents/libnn
make lib                 # gera libnn.a e libnn_shared.so
python3 -m pip install -e python
```

Para ambiente sem instalação (apenas importar do repo):

```sh
cd /home/thiago/Documents/libnn
python3 -c "import sys; sys.path.insert(0, 'python'); import libnn; print(libnn.version())"
```

Se a biblioteca estiver em outro local, defina `LIBNN_SO=/caminho/libnn_shared.so`.

## Uso rápido

```python
import libnn as nn

nn.seed(7)
with nn.Arena(16 << 20) as model, nn.Arena(4 << 20) as scratch:
    net = nn.Graph.ff(model, input_dim=2, hidden_dim=8, output_dim=1)
    opt = nn.Adam(lr=0.05)

    for step in range(300):
        scratch.reset()
        x = nn.Tensor(scratch, [4, 2], data=[[0,0], [0,1], [1,0], [1,1]])
        y = nn.Tensor(scratch, [4, 1], data=[[0], [1], [1], [0]])
        pred = net.forward(x, scratch)
        loss = net.train_backprop(pred, y, opt, scratch)

    net.render_svg("xor.svg")
    net.destroy()
```

## API

| Classe / função | Finalidade |
| :-- | :-- |
| `Arena(capacity)` | Arena de memória; context manager; `reset()`, `bytes_used` |
| `Tensor(arena, shape, data=...)` | Tensor contíguo; `tolist()`, `fill_random_normal()` |
| `Graph.ff/ae/dae/vae/...` / `GAN` | Construtores do ciclo essencial (inclui GAN) |
| `Graph.forward(x, scratch)` | Forward; retorna `Tensor` |
| `Graph.train_backprop(pred, y, opt, scratch)` | MSE + backward + Adam; retorna a perda |
| `Graph.contrastive_divergence/hebbian/stdp/ridge/kohonen_step/hinge` | Treinadores não-diferenciáveis |
| `Graph.render_svg(path)` / `to_svg()` | Exportação SVG |
| `Graph.sequence_forward/backward/reset_state` | BPTT e estados recorrentes |
| `Adam(lr, beta1, beta2, epsilon)` | Hiperparâmetros do otimizador |
| `seed(value)` | Semente determinística do RNG |
| `LibNNError` | Exceção para falhas da biblioteca C |

### Ciclo de vida

O grafo **empresta** a arena do modelo. Fechar `with Arena(...)` com grafos
ainda vivos levanta `LibNNError`. Use `graph.destroy()` ou o próprio grafo como
context manager. Tensores retornados por `forward` vivem na arena scratch:
`scratch.reset()` os invalida e qualquer uso posterior levanta `LibNNError`.

```python
with nn.Arena(8 << 20) as model:
    with nn.Graph.perceptron(model, 2, 1) as net:
        ...
```

## Escopo da primeira versão

Inclui o ciclo essencial: 15 construtores (com GAN), forward e BPTT, treinadores
(inclusive o passo adversarial) e render. Tensores de baixo nível (matmul,
conv, im2col) e as operações `nc_nn_*` permanecem disponíveis apenas em C.

## Testes

```sh
python3 -m pytest python/tests -q
```
