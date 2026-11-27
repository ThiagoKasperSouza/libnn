"""Camada ctypes sobre libnn.so.

Este módulo é interno: ele carrega a biblioteca compartilhada, declara as
estruturas C necessárias e as assinaturas das funções. A API pública fica em
``libnn.__init__``.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import POINTER, c_bool, c_char_p, c_float, c_int, c_size_t, c_void_p
from pathlib import Path


class LibNNError(Exception):
    """Falha ao carregar ou chamar a biblioteca libnn."""


def _find_library() -> Path:
    """Localiza libnn.so.

    Ordem: variável LIBNN_SO, pasta do pacote, raiz do repositório (duas
    pastas acima de python/libnn), diretório de trabalho atual.
    """
    candidates = []
    env = os.environ.get("LIBNN_SO")
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve()
    candidates.extend(
        [
            here.parent / "libnn_shared.so",
            here.parents[2] / "libnn_shared.so",
            Path.cwd() / "libnn_shared.so",
        ]
    )
    for path in candidates:
        if path.is_file():
            return path
    checked = ", ".join(str(p) for p in candidates)
    raise LibNNError(
        "libnn_shared.so não encontrada. Rode `make shared` na raiz do repositório "
        f"ou defina LIBNN_SO. Procurado em: {checked}"
    )


LIB_PATH = _find_library()
try:
    lib = ctypes.CDLL(str(LIB_PATH))
except OSError as exc:  # pragma: no cover - depende do ambiente
    raise LibNNError(f"falha ao carregar {LIB_PATH}: {exc}") from exc


# ---------------------------------------------------------------------------
# Estruturas C
# ---------------------------------------------------------------------------


class NC_Arena(ctypes.Structure):
    _fields_ = [
        ("buffer", POINTER(c_void_p)),
        ("capacity", c_size_t),
        ("offset", c_size_t),
    ]


class NC_Tensor(ctypes.Structure):
    _fields_ = [
        ("data", POINTER(c_float)),
        ("grad", POINTER(c_float)),
        ("shape", c_size_t * 4),
        ("strides", c_size_t * 4),
        ("ndim", c_size_t),
        ("size", c_size_t),
        ("requires_grad", c_bool),
    ]


class NC_Cell(ctypes.Structure):
    # A união `state` e os ponteiros de gate são opacos para o wrapper.
    _fields_ = [("_opaque", c_void_p)]


class NC_Graph(ctypes.Structure):
    # Campos públicos até `tape`: o wrapper precisa do tape para gravar a perda.
    # O restante da estrutura permanece opaco (não é acessado).
    _fields_ = [
        ("model_arena", c_void_p),
        ("cells", c_void_p),
        ("cell_count", c_size_t),
        ("cell_capacity", c_size_t),
        ("edges", c_void_p),
        ("edge_count", c_size_t),
        ("edge_capacity", c_size_t),
        ("tape", c_void_p),
        ("_rest", c_void_p * 16),
    ]


ArenaPtr = POINTER(NC_Arena)
TensorPtr = POINTER(NC_Tensor)
GraphPtr = POINTER(NC_Graph)


# ---------------------------------------------------------------------------
# Assinaturas
# ---------------------------------------------------------------------------


def _bind(name: str, restype, argtypes) -> None:
    func = getattr(lib, name)
    func.restype = restype
    func.argtypes = argtypes
    globals()[name] = func  # expõe como atributo do módulo


# Arena
_bind("nc_arena_init", NC_Arena, [c_size_t])
_bind("nc_arena_reset", None, [ArenaPtr])
_bind("nc_arena_free", None, [ArenaPtr])
_bind("nc_arena_bytes_used", c_size_t, [ArenaPtr])
_bind("nc_arena_bytes_available", c_size_t, [ArenaPtr])

# Tensor
_bind("nc_tensor_create", TensorPtr, [ArenaPtr, POINTER(c_size_t), c_size_t, c_bool])
_bind("nc_tensor_fill_zeros", None, [TensorPtr])
_bind("nc_tensor_fill_random_normal", None, [TensorPtr, c_float, c_float])
_bind("nc_rng_seed", None, [c_void_p])  # uint64_t

# Grafo / topologia
_bind("nc_graph_destroy", None, [GraphPtr])
_bind(
    "nc_graph_forward",
    TensorPtr,
    [GraphPtr, TensorPtr, ArenaPtr],
)
_bind("nc_graph_backward", None, [GraphPtr])
_bind("nc_autograd_record_mse", None, [c_void_p, TensorPtr, TensorPtr])
_bind("nc_compute_mse_loss", c_float, [TensorPtr, TensorPtr])

# Treinadores
_bind("nc_train_backprop_step", None, [GraphPtr, c_void_p])
_bind("nc_train_contrastive_divergence", None, [GraphPtr, TensorPtr, c_size_t, c_float])
_bind("nc_train_hebbian_step", None, [GraphPtr, c_float])
_bind("nc_train_stdp_step", None, [GraphPtr, c_float, c_float])
_bind(
    "nc_train_reservoir_ridge_regression",
    c_bool,
    [GraphPtr, TensorPtr, TensorPtr, c_float],
)
_bind("nc_train_kohonen_bmu_step", None, [GraphPtr, TensorPtr, c_float, c_float])
_bind("nc_train_kernel_hinge_loss", None, [GraphPtr, TensorPtr, TensorPtr, c_float])

# Sequências (Marco 4)
_bind("nc_sequence_forward", TensorPtr, [GraphPtr, TensorPtr, c_size_t, ArenaPtr])
_bind("nc_sequence_backward", None, [GraphPtr])
_bind("nc_sequence_reset_state", None, [GraphPtr])

# Render (Marco 6)
_bind("nc_render_svg", c_int, [GraphPtr, c_char_p])
_bind("nc_render_svg_string", c_int, [GraphPtr, c_char_p, c_size_t])

# Fábricas do zoo -----------------------------------------------------------
_zoo_2i = [
    "nc_zoo_create_perceptron",
    "nc_zoo_create_ae",
    "nc_zoo_create_rbm",
    "nc_zoo_create_boltzmann",
    "nc_zoo_create_markov_chain",
]
for _name in _zoo_2i:
    _bind(_name, GraphPtr, [ArenaPtr, c_size_t, c_size_t])

_zoo_3i = [
    "nc_zoo_create_rnn",
    "nc_zoo_create_lstm",
    "nc_zoo_create_gru",
    "nc_zoo_create_elm",
    "nc_zoo_create_esn",
    "nc_zoo_create_lsm",
    "nc_zoo_create_rbf",
]
for _name in _zoo_3i:
    _bind(_name, GraphPtr, [ArenaPtr, c_size_t, c_size_t, c_float if _name == "nc_zoo_create_dae" else c_size_t])

_bind("nc_zoo_create_ff", GraphPtr, [ArenaPtr, c_size_t, c_size_t, c_size_t])
_bind("nc_zoo_create_dae", GraphPtr, [ArenaPtr, c_size_t, c_size_t, c_float])
_bind("nc_zoo_create_sae", GraphPtr, [ArenaPtr, c_size_t, c_size_t, c_float, c_float])
_bind("nc_zoo_create_svm", GraphPtr, [ArenaPtr, c_size_t])

_zoo_hopfield = "nc_zoo_create_hopfield"
_bind(_zoo_hopfield, GraphPtr, [ArenaPtr, c_size_t])
_bind("nc_zoo_create_kohonen", GraphPtr, [ArenaPtr, c_size_t, c_size_t])
_bind("nc_zoo_create_vae", GraphPtr, [ArenaPtr, c_size_t, c_size_t])
_bind("nc_zoo_create_ntm", GraphPtr, [ArenaPtr, c_size_t, c_size_t, c_size_t, c_size_t])
_bind("nc_zoo_create_dcn", GraphPtr, [ArenaPtr, c_size_t, c_size_t, c_size_t, c_size_t])
_bind("nc_zoo_create_dn", GraphPtr, [ArenaPtr, c_size_t, c_size_t, c_size_t])
_bind("nc_zoo_create_dcign", GraphPtr, [ArenaPtr, c_size_t, c_size_t, c_size_t, c_size_t])

# dff / drn / dbn recebem array de dimensões ocultas.
_bind("nc_zoo_create_dff", GraphPtr, [ArenaPtr, c_size_t, POINTER(c_size_t), c_size_t, c_size_t])
_bind("nc_zoo_create_drn", GraphPtr, [ArenaPtr, c_size_t, POINTER(c_size_t), c_size_t, c_size_t])
_bind("nc_zoo_create_dbn", GraphPtr, [ArenaPtr, c_size_t, POINTER(c_size_t), c_size_t])

__all__ = [
    "lib",
    "lib_path",
    "LibNNError",
    "NC_Arena",
    "NC_Tensor",
    "NC_Graph",
    "ArenaPtr",
    "TensorPtr",
    "GraphPtr",
]

# Alias público para o caminho da biblioteca carregada.
lib_path = LIB_PATH
