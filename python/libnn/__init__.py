"""Bindings Python para libnn.

Uso típico::

    import libnn as nn

    nn.seed(7)
    with nn.Arena(16 << 20) as model, nn.Arena(4 << 20) as scratch:
        net = nn.Graph.ff(model, input_dim=3, hidden_dim=4, output_dim=1)
        x = nn.Tensor(scratch, [2, 3], data=[[0, 1, 0], [1, 0, 1]])
        y = nn.Tensor(scratch, [2, 1], data=[[0], [1]])
        for _ in range(100):
            pred = net.forward(x, scratch)
            net.train_backprop(pred, y, nn.Adam(lr=0.05), scratch)
        net.render_svg("rede.svg")

O ciclo de vida das arenas segue o modelo C: o grafo "empresta" a arena do
modelo. Fechar um ``with Arena`` com grafos ainda vivos gera ``LibNNError``.
"""

from __future__ import annotations

import ctypes
import weakref
from ctypes import byref, c_bool, c_float, c_int, c_size_t, c_void_p
from pathlib import Path
from typing import Iterable, Sequence

from . import _core
from ._core import LibNNError

__version__ = "0.1.0"

# Reexporta o caminho da biblioteca C carregada, útil para diagnóstico.
lib_path = _core.lib_path

__all__ = [
    "Adam",
    "Arena",
    "GAN",
    "Graph",
    "LibNNError",
    "Tensor",
    "lib_path",
    "seed",
    "set_seed",
    "version",
]


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def _c_size_array(values: Sequence[int]):
    return (c_size_t * len(values))(*values)


def _flatten(data) -> list[float]:
    """Achata uma sequência aninhada qualquer em uma lista de floats."""
    if isinstance(data, (int, float)):
        return [float(data)]
    out: list[float] = []
    for item in data:
        if isinstance(item, (list, tuple, Sequence)) and not isinstance(item, (str, bytes)):
            out.extend(_flatten(item))
        else:
            out.append(float(item))
    return out


# ---------------------------------------------------------------------------
# Arena
# ---------------------------------------------------------------------------


class Arena:
    """Arena de memória alinhada da libnn.

    Use como context manager::

        with nn.Arena(16 << 20) as arena:
            ...

    O grafo criado com esta arena a referencia; ao sair do ``with``, se ainda
    houver grafos vivos, um ``LibNNError`` é levantado.
    """

    def __init__(self, capacity_bytes: int = 8 << 20):
        if capacity_bytes <= 0:
            raise ValueError("capacidade deve ser positiva")
        self._arena = _core.nc_arena_init(c_size_t(capacity_bytes))
        if not self._arena.buffer:
            raise LibNNError("falha ao alocar arena")
        self._closed = False
        self._live_graphs: set[int] = set()
        self._live_tensors = weakref.WeakSet()

    @property
    def bytes_used(self) -> int:
        self._ensure_open()
        return int(_core.nc_arena_bytes_used(byref(self._arena)))

    @property
    def bytes_available(self) -> int:
        self._ensure_open()
        return int(_core.nc_arena_bytes_available(byref(self._arena)))

    def reset(self) -> None:
        """Libera todas as alocações em O(1), mantendo a arena utilizável."""
        self._ensure_open()
        if self._live_graphs:
            raise LibNNError(
                "não é possível resetar arena com grafos vivos: "
                f"{len(self._live_graphs)} grafo(s)"
            )
        for tensor in list(self._live_tensors):
            tensor._alive = False
        self._live_tensors.clear()
        _core.nc_arena_reset(byref(self._arena))

    def close(self) -> None:
        if self._closed:
            return
        if self._live_graphs:
            raise LibNNError(
                "arena fechada com grafos ainda vivos. Destrua-os antes: "
                f"{len(self._live_graphs)} grafo(s)."
            )
        for tensor in list(self._live_tensors):
            tensor._alive = False
        self._live_tensors.clear()
        _core.nc_arena_free(byref(self._arena))
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise LibNNError("arena fechada")

    def __enter__(self) -> "Arena":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.close()
        else:
            # Não mascara a exceção original, mas libera a memória C.
            try:
                self.close()
            except LibNNError:
                pass
        return False

    def __del__(self):
        try:
            if not self._closed:
                self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------


class Tensor:
    """Tensor contíguo 1D–4D alocado em uma :class:`Arena`.

    O tensor não possui a arena: ela deve permanecer viva enquanto o tensor
    for usado.
    """

    def __init__(
        self,
        arena: Arena,
        shape: Sequence[int],
        *,
        data=None,
        requires_grad: bool = False,
    ):
        arena._ensure_open()
        if not 1 <= len(shape) <= 4:
            raise ValueError("shape deve ter entre 1 e 4 dimensões")
        if any(d <= 0 for d in shape):
            raise ValueError("dimensões devem ser positivas")

        self._arena = arena
        self._ptr = _core.nc_tensor_create(
            byref(arena._arena), _c_size_array(shape), c_size_t(len(shape)), c_bool(requires_grad)
        )
        if not self._ptr:
            raise LibNNError("falha ao criar tensor (arena sem espaço?)")
        self._alive = True
        arena._live_tensors.add(self)

        flat = _flatten(data) if data is not None else None
        if flat is not None:
            expected = 1
            for d in shape:
                expected *= d
            if len(flat) != expected:
                raise ValueError(
                    f"dados têm {len(flat)} elementos, mas shape {tuple(shape)} espera {expected}"
                )
            self._copy_in(flat)

    def _ensure_valid(self) -> None:
        if not getattr(self, "_alive", False):
            raise LibNNError("tensor inválido: arena scratch foi resetada/fechada")
        self._arena._ensure_open()

    def _copy_in(self, flat: Sequence[float]) -> None:
        self._ensure_valid()
        buf = (c_float * len(flat))(*[float(v) for v in flat])
        ctypes.memmove(self._ptr.contents.data, buf, len(flat) * ctypes.sizeof(c_float))

    def tolist(self):
        """Retorna os dados como lista aninhada segundo o shape."""
        self._ensure_valid()
        shape = tuple(self._ptr.contents.shape[i] for i in range(self._ptr.contents.ndim))
        flat = [self._ptr.contents.data[i] for i in range(self._ptr.contents.size)]

        def nest(dims, values, offset=0):
            if not dims:
                return values[offset]
            step = 1
            for d in dims[1:]:
                step *= d
            return [nest(dims[1:], values, offset + i * step) for i in range(dims[0])]

        return nest(shape, flat)

    @property
    def shape(self) -> tuple[int, ...]:
        self._ensure_valid()
        return tuple(self._ptr.contents.shape[i] for i in range(self._ptr.contents.ndim))

    @property
    def data(self):
        """Acesso direto ao buffer (somente leitura é garantida pela API)."""
        self._ensure_valid()
        return self._ptr.contents.data

    def fill_random_normal(self, mean: float = 0.0, stddev: float = 1.0) -> None:
        self._ensure_valid()
        _core.nc_tensor_fill_random_normal(self._ptr, c_float(mean), c_float(stddev))

    def fill_zeros(self) -> None:
        self._ensure_valid()
        _core.nc_tensor_fill_zeros(self._ptr)

    def close(self) -> None:
        """Marca o tensor como inutilizável. A memória é liberada pela arena."""
        self._alive = False

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, arena={hex(id(self._arena))})"


# ---------------------------------------------------------------------------
# Otimizador
# ---------------------------------------------------------------------------


class Adam:
    """Hiperparâmetros do otimizador Adam (Marco 2)."""

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        if lr <= 0 or not 0 < beta1 < 1 or not 0 < beta2 < 1 or epsilon <= 0:
            raise ValueError("parâmetros inválidos para Adam")
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)

    def _as_struct(self):
        class _Adam(ctypes.Structure):
            _fields_ = [
                ("learning_rate", c_float),
                ("beta1", c_float),
                ("beta2", c_float),
                ("epsilon", c_float),
            ]

        return _Adam(c_float(self.lr), c_float(self.beta1), c_float(self.beta2), c_float(self.epsilon))


# ---------------------------------------------------------------------------
# Grafo
# ---------------------------------------------------------------------------


class Graph:
    """Grafo computacional da libnn.

    Construtores de classe mapeiam as fábricas do ``nc_zoo``. O grafo empresta
    a arena do modelo; a arena scratch é passada em cada operação.
    """

    def __init__(self, ptr, arena: Arena):
        if not ptr:
            raise LibNNError("falha ao criar grafo")
        self._ptr = ptr
        self._arena = arena
        self._alive = True
        arena._live_graphs.add(id(self))

    def _ensure_alive(self) -> None:
        if not self._alive:
            raise LibNNError("grafo já destruído")
        self._arena._ensure_open()

    @classmethod
    def _factory(cls, func, arena: Arena, *sizes: int) -> "Graph":
        """Chama uma fábrica cujos argumentos são todos ``size_t``."""
        arena._ensure_open()
        return cls(func(byref(arena._arena), *[c_size_t(v) for v in sizes]), arena)

    @classmethod
    def _factory_hidden(cls, func, arena: Arena, input_dim: int,
                        hidden_dims: Sequence[int], output_dim: int) -> "Graph":
        """Chama uma fábrica com array `const size_t hidden_dims[]`."""
        arena._ensure_open()
        dims = [int(d) for d in hidden_dims]
        if not dims:
            raise ValueError("hidden_dims não pode ser vazio")
        if any(d <= 0 for d in dims):
            raise ValueError("dimensões ocultas devem ser positivas")
        arr = _c_size_array(dims)
        return cls(
            func(
                byref(arena._arena),
                c_size_t(input_dim),
                arr,
                c_size_t(len(dims)),
                c_size_t(output_dim),
            ),
            arena,
        )

    # -- Fábricas feedforward ------------------------------------------------
    @classmethod
    def perceptron(cls, arena, input_dim, output_dim):
        return cls._factory(_core.lib.nc_zoo_create_perceptron, arena, input_dim, output_dim)

    @classmethod
    def ff(cls, arena, input_dim, hidden_dim, output_dim):
        return cls._factory(_core.lib.nc_zoo_create_ff, arena, input_dim, hidden_dim, output_dim)

    @classmethod
    def dff(cls, arena, input_dim, hidden_dims, output_dim):
        return cls._factory_hidden(
            _core.lib.nc_zoo_create_dff, arena, input_dim, hidden_dims, output_dim
        )

    @classmethod
    def drn(cls, arena, input_dim, hidden_dims, output_dim):
        return cls._factory_hidden(
            _core.lib.nc_zoo_create_drn, arena, input_dim, hidden_dims, output_dim
        )

    @classmethod
    def ae(cls, arena, input_dim, hidden_dim):
        return cls._factory(_core.lib.nc_zoo_create_ae, arena, input_dim, hidden_dim)

    @classmethod
    def dae(cls, arena, input_dim, hidden_dim, noise_level=0.2):
        arena._ensure_open()
        ptr = _core.lib.nc_zoo_create_dae(
            byref(arena._arena), c_size_t(input_dim), c_size_t(hidden_dim), c_float(noise_level)
        )
        return cls(ptr, arena)

    @classmethod
    def vae(cls, arena, input_dim, latent_dim):
        return cls._factory(_core.lib.nc_zoo_create_vae, arena, input_dim, latent_dim)

    # -- Energia / reservatório ---------------------------------------------
    @classmethod
    def hopfield(cls, arena, dim):
        return cls._factory(_core.lib.nc_zoo_create_hopfield, arena, dim)

    @classmethod
    def rbm(cls, arena, visible, hidden):
        return cls._factory(_core.lib.nc_zoo_create_rbm, arena, visible, hidden)

    @classmethod
    def boltzmann(cls, arena, visible, hidden):
        return cls._factory(_core.lib.nc_zoo_create_boltzmann, arena, visible, hidden)

    @classmethod
    def markov_chain(cls, arena, states, hidden):
        return cls._factory(_core.lib.nc_zoo_create_markov_chain, arena, states, hidden)

    @classmethod
    def dbn(cls, arena, visible, hidden_dims):
        arena._ensure_open()
        dims = [int(d) for d in hidden_dims]
        if not dims or any(d <= 0 for d in dims):
            raise ValueError("hidden_dims inválido")
        ptr = _core.lib.nc_zoo_create_dbn(
            byref(arena._arena), c_size_t(visible), _c_size_array(dims), c_size_t(len(dims))
        )
        return cls(ptr, arena)

    @classmethod
    def kohonen(cls, arena, input_dim, neurons):
        return cls._factory(_core.lib.nc_zoo_create_kohonen, arena, input_dim, neurons)

    @classmethod
    def elm(cls, arena, input_dim, neurons, output_dim):
        return cls._factory(_core.lib.nc_zoo_create_elm, arena, input_dim, neurons, output_dim)

    @classmethod
    def lsm(cls, arena, input_dim, neurons, output_dim):
        return cls._factory(_core.lib.nc_zoo_create_lsm, arena, input_dim, neurons, output_dim)

    @classmethod
    def esn(cls, arena, input_dim, neurons, output_dim):
        return cls._factory(_core.lib.nc_zoo_create_esn, arena, input_dim, neurons, output_dim)

    @classmethod
    def rbf(cls, arena, input_dim, centers, output_dim):
        return cls._factory(_core.lib.nc_zoo_create_rbf, arena, input_dim, centers, output_dim)

    @classmethod
    def svm(cls, arena, input_dim):
        return cls._factory(_core.lib.nc_zoo_create_svm, arena, input_dim)

    # -- Recorrentes ---------------------------------------------------------
    @classmethod
    def rnn(cls, arena, input_dim, hidden_dim, output_dim):
        return cls._factory(_core.lib.nc_zoo_create_rnn, arena, input_dim, hidden_dim, output_dim)

    @classmethod
    def lstm(cls, arena, input_dim, hidden_dim, output_dim):
        return cls._factory(_core.lib.nc_zoo_create_lstm, arena, input_dim, hidden_dim, output_dim)

    @classmethod
    def gru(cls, arena, input_dim, hidden_dim, output_dim):
        return cls._factory(_core.lib.nc_zoo_create_gru, arena, input_dim, hidden_dim, output_dim)

    @classmethod
    def ntm(cls, arena, input_dim, slots, mem_dim, output_dim):
        return cls._factory(_core.lib.nc_zoo_create_ntm, arena, input_dim, slots, mem_dim, output_dim)

    # -- Convolucionais ------------------------------------------------------
    @classmethod
    def dcn(cls, arena, channels, h, w, num_classes):
        return cls._factory(_core.lib.nc_zoo_create_dcn, arena, channels, h, w, num_classes)

    @classmethod
    def dn(cls, arena, channels, h, w):
        return cls._factory(_core.lib.nc_zoo_create_dn, arena, channels, h, w)

    @classmethod
    def dcign(cls, arena, channels, h, w, latent_channels):
        return cls._factory(
            _core.lib.nc_zoo_create_dcign, arena, channels, h, w, latent_channels
        )

    # -- Operações -----------------------------------------------------------
    def forward(self, x: Tensor, scratch: Arena) -> Tensor:
        self._ensure_alive()
        scratch._ensure_open()
        ptr = _core.nc_graph_forward(self._ptr, x._ptr, byref(scratch._arena))
        if not ptr:
            raise LibNNError("falha no forward do grafo")
        return _tensor_from_ptr(ptr, scratch)

    def train_backprop(
        self,
        pred: Tensor,
        target: Tensor,
        optimizer: Adam,
        scratch: Arena,
    ) -> float:
        """Executa perda MSE, backward e passo Adam. Retorna a perda."""
        self._ensure_alive()
        pred._ensure_valid()
        target._ensure_valid()
        loss = float(_core.nc_compute_mse_loss(pred._ptr, target._ptr))
        _core.nc_autograd_record_mse(c_void_p(self._ptr.contents.tape), pred._ptr, target._ptr)
        _core.nc_graph_backward(self._ptr)
        opt = optimizer._as_struct()
        _core.nc_train_backprop_step(self._ptr, byref(opt))
        return loss

    def contrastive_divergence(self, x: Tensor, k: int = 1, lr: float = 0.05) -> None:
        self._ensure_alive()
        _core.nc_train_contrastive_divergence(self._ptr, x._ptr, c_size_t(k), c_float(lr))

    def hebbian(self, lr: float = 0.5) -> None:
        self._ensure_alive()
        _core.nc_train_hebbian_step(self._ptr, c_float(lr))

    def stdp(self, lr: float = 0.1, tau: float = 10.0) -> None:
        self._ensure_alive()
        _core.nc_train_stdp_step(self._ptr, c_float(lr), c_float(tau))

    def ridge(self, states: Tensor, targets: Tensor, alpha: float = 1e-3) -> None:
        self._ensure_alive()
        ok = _core.nc_train_reservoir_ridge_regression(
            self._ptr, states._ptr, targets._ptr, c_float(alpha)
        )
        if not ok:
            raise LibNNError("falha na regressão ridge")

    def kohonen_step(self, x: Tensor, lr: float = 0.3, radius: float = 2.0) -> None:
        self._ensure_alive()
        _core.nc_train_kohonen_bmu_step(self._ptr, x._ptr, c_float(lr), c_float(radius))

    def hinge(self, x: Tensor, y: Tensor, C: float = 1.0) -> None:
        self._ensure_alive()
        _core.nc_train_kernel_hinge_loss(self._ptr, x._ptr, y._ptr, c_float(C))

    def sequence_forward(self, x_seq: Tensor, steps: int, scratch: Arena) -> list[Tensor]:
        """Desdobra `steps` passos (BPTT) e retorna as saídas de cada passo.

        `x_seq` deve ter shape `[T, N, F]` com `T == steps`.
        """
        self._ensure_alive()
        scratch._ensure_open()
        if x_seq.shape[0] != steps:
            raise ValueError(f"x_seq tem {x_seq.shape[0]} passos, esperado {steps}")
        ptr = _core.nc_sequence_forward(
            self._ptr, x_seq._ptr, c_size_t(steps), byref(scratch._arena)
        )
        if not ptr:
            raise LibNNError("falha no forward sequencial")
        return [_tensor_from_ptr(ptr[i], scratch) for i in range(steps)]

    def sequence_backward(self) -> None:
        self._ensure_alive()
        _core.nc_sequence_backward(self._ptr)

    def reset_state(self) -> None:
        """Limpa estados recorrentes antes de nova sequência."""
        self._ensure_alive()
        _core.nc_sequence_reset_state(self._ptr)

    def render_svg(self, filename: str | Path) -> None:
        """Exporta a topologia para um arquivo SVG."""
        self._ensure_alive()
        path = Path(filename)
        encoded = str(path).encode("utf-8")
        rc = _core.nc_render_svg(self._ptr, encoded)
        if rc != 0:
            raise LibNNError(f"falha ao escrever SVG em {path}")

    def to_svg(self) -> str:
        """Retorna o SVG como string."""
        self._ensure_alive()
        size = _core.nc_render_svg_string(self._ptr, None, c_size_t(0))
        if size < 0:
            raise LibNNError("falha ao calcular tamanho do SVG")
        buf = ctypes.create_string_buffer(size + 1)
        written = _core.nc_render_svg_string(self._ptr, buf, c_size_t(size + 1))
        if written < 0:
            raise LibNNError("falha ao renderizar SVG")
        return buf.value.decode("utf-8")

    def destroy(self) -> None:
        if not self._alive:
            return
        _core.nc_graph_destroy(self._ptr)
        self._arena._live_graphs.discard(id(self))
        self._alive = False

    def __enter__(self):
        self._ensure_alive()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.destroy()
        return False

    def __del__(self):
        try:
            if self._alive:
                self.destroy()
        except Exception:
            pass


class GAN:
    """Par gerador/discriminador da libnn (Marco 3).

    Use como context manager para destruir ambos os grafos ao sair.
    """

    def __init__(self, arena: Arena, latent_dim: int, data_dim: int):
        arena._ensure_open()
        ptr = _core.nc_zoo_create_gan(
            byref(arena._arena), c_size_t(latent_dim), c_size_t(data_dim)
        )
        if not ptr:
            raise LibNNError("falha ao criar GAN")
        self._ptr = ptr
        self._arena = arena
        self.generator = Graph(ptr.contents.generator, arena)
        self.discriminator = Graph(ptr.contents.discriminator, arena)
        self._alive = True

    def train_step(self, real: Tensor, optimizer: Adam, scratch: Arena) -> None:
        """Passo minimax: treina D com reais/falsos e G para enganar D."""
        if not self._alive:
            raise LibNNError("GAN já destruído")
        self._arena._ensure_open()
        scratch._ensure_open()
        opt = optimizer._as_struct()
        _core.nc_train_gan_step(
            self.generator._ptr,
            self.discriminator._ptr,
            real._ptr,
            byref(scratch._arena),
            byref(opt),
        )

    def destroy(self) -> None:
        if not self._alive:
            return
        self.generator.destroy()
        self.discriminator.destroy()
        self._alive = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.destroy()
        return False

    def __del__(self):
        try:
            if self._alive:
                self.destroy()
        except Exception:
            pass


# Permite construir um Tensor a partir de um ponteiro C sem alocar de novo.
def _tensor_from_ptr(ptr, arena: Arena) -> Tensor:
    obj = Tensor.__new__(Tensor)
    obj._arena = arena
    obj._ptr = ptr
    obj._alive = True
    arena._live_tensors.add(obj)
    return obj



# ---------------------------------------------------------------------------
# RNG / versão
# ---------------------------------------------------------------------------


def seed(value: int) -> None:
    """Fixa a semente do gerador determinístico da libnn."""
    _core.nc_rng_seed(c_void_p(value))


set_seed = seed


def version() -> str:
    """Versão do pacote Python de bindings."""
    return __version__
