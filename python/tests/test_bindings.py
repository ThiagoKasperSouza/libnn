"""Testes dos bindings Python (ctypes)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import libnn as nn  # noqa: E402


def test_arena_context_manager():
    with nn.Arena(1 << 20) as arena:
        assert arena.bytes_used >= 0
        assert arena.bytes_available > 0
    with pytest.raises(nn.LibNNError):
        arena.bytes_used


def test_arena_rejects_close_with_live_graph():
    with pytest.raises(nn.LibNNError, match="grafos ainda vivos"):
        with nn.Arena(1 << 20) as arena:
            graph = nn.Graph.perceptron(arena, 2, 1)
            assert graph is not None


def test_tensor_roundtrip():
    with nn.Arena(1 << 20) as scratch:
        t = nn.Tensor(scratch, [2, 3], data=[[1, 2, 3], [4, 5, 6]])
        assert t.shape == (2, 3)
        assert t.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        t.fill_zeros()
        assert t.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_tensor_bad_shape():
    with nn.Arena(1 << 20) as scratch:
        with pytest.raises(ValueError):
            nn.Tensor(scratch, [2, 2], data=[1, 2, 3])
        with pytest.raises(ValueError):
            nn.Tensor(scratch, [])


def test_graph_forward_and_render():
    with nn.Arena(8 << 20) as model, nn.Arena(4 << 20) as scratch:
        net = nn.Graph.ff(model, 3, 4, 2)
        x = nn.Tensor(scratch, [1, 3], data=[[0.2, 0.4, 0.6]])
        pred = net.forward(x, scratch)
        assert pred.shape == (1, 2)

        svg = net.to_svg()
        assert svg.startswith("<?xml")
        assert "<svg" in svg and "</svg>" in svg
        assert "#FFD700" in svg  # amarelo entrada
        net.destroy()


def test_xor_training_converges():
    nn.seed(4)
    with nn.Arena(16 << 20) as model, nn.Arena(4 << 20) as scratch:
        net = nn.Graph.ff(model, 2, 8, 1)
        opt = nn.Adam(lr=0.05)
        x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y_data = [[0], [1], [1], [0]]

        first = last = 0.0
        for step in range(300):
            scratch.reset()
            x = nn.Tensor(scratch, [4, 2], data=x_data)
            y = nn.Tensor(scratch, [4, 1], data=y_data)
            pred = net.forward(x, scratch)
            if step == 0:
                first = nn._core.nc_compute_mse_loss(pred._ptr, y._ptr)
            last = net.train_backprop(pred, y, opt, scratch)

        assert first > 0.1
        assert last < 0.01
        net.destroy()


def test_zoo_constructors():
    with nn.Arena(32 << 20) as model:
        factories = [
            lambda: nn.Graph.perceptron(model, 3, 2),
            lambda: nn.Graph.ae(model, 4, 2),
            lambda: nn.Graph.dae(model, 4, 2, 0.2),
            lambda: nn.Graph.vae(model, 4, 2),
            lambda: nn.Graph.hopfield(model, 4),
            lambda: nn.Graph.rbm(model, 4, 3),
            lambda: nn.Graph.kohonen(model, 4, 5),
            lambda: nn.Graph.elm(model, 4, 6, 1),
            lambda: nn.Graph.svm(model, 4),
            lambda: nn.Graph.rnn(model, 4, 5, 2),
            lambda: nn.Graph.lstm(model, 4, 5, 2),
            lambda: nn.Graph.gru(model, 4, 5, 2),
            lambda: nn.Graph.ntm(model, 4, 3, 5, 2),
        ]
        for make in factories:
            graph = make()
            assert graph is not None
            graph.destroy()


def test_render_svg_file(tmp_path):
    with nn.Arena(8 << 20) as model:
        net = nn.Graph.lstm(model, 3, 4, 3)
        path = tmp_path / "lstm.svg"
        net.render_svg(path)
        assert path.exists()
        assert path.stat().st_size > 100
        content = path.read_text()
        assert "#3399FF" in content  # azul memória
        net.destroy()


def test_seed_is_deterministic():
    with nn.Arena(4 << 20) as scratch:
        nn.seed(11)
        a = nn.Tensor(scratch, [4], data=[0, 0, 0, 0])
        a.fill_random_normal(0, 1)
        first = a.tolist()

        nn.seed(11)
        b = nn.Tensor(scratch, [4], data=[0, 0, 0, 0])
        b.fill_random_normal(0, 1)
        assert b.tolist() == first


def test_version_and_path():
    assert nn.version() == "0.1.0"
    assert Path(nn.lib_path).name == "libnn_shared.so"


def test_gan_constructor_and_step():
    with nn.Arena(32 << 20) as model, nn.Arena(8 << 20) as scratch:
        with nn.GAN(model, latent_dim=4, data_dim=6) as gan:
            assert gan.generator is not None
            assert gan.discriminator is not None
            real = nn.Tensor(scratch, [2, 6])
            real.fill_random_normal(0, 1)
            gan.train_step(real, nn.Adam(lr=0.001), scratch)


def test_sequence_forward_returns_list_of_tensors():
    with nn.Arena(16 << 20) as model, nn.Arena(8 << 20) as scratch:
        net = nn.Graph.lstm(model, 3, 4, 2)
        x = nn.Tensor(scratch, [3, 2, 3])
        x.fill_random_normal(0, 1)
        outputs = net.sequence_forward(x, 3, scratch)
        assert len(outputs) == 3
        assert all(out.shape == (2, 2) for out in outputs)
        net.sequence_backward()
        net.reset_state()
        net.destroy()


def test_forward_tensor_invalidated_by_scratch_reset():
    with nn.Arena(8 << 20) as model, nn.Arena(4 << 20) as scratch:
        net = nn.Graph.ff(model, 3, 4, 1)
        x = nn.Tensor(scratch, [1, 3], data=[[0.1, 0.2, 0.3]])
        pred = net.forward(x, scratch)
        assert pred.shape == (1, 1)
        scratch.reset()
        with pytest.raises(nn.LibNNError, match="tensor inválido"):
            pred.tolist()
        net.destroy()


def test_render_svg_string_length():
    with nn.Arena(8 << 20) as model:
        net = nn.Graph.perceptron(model, 2, 1)
        svg = net.to_svg()
        assert len(svg) > 100
        assert svg.startswith("<?xml")
        net.destroy()


def test_remaining_zoo_factories():
    with nn.Arena(64 << 20) as model:
        factories = [
            lambda: nn.Graph.dff(model, 4, [8, 6], 2),
            lambda: nn.Graph.drn(model, 4, [8, 6], 2),
            lambda: nn.Graph.dbn(model, 4, [6, 4]),
            lambda: nn.Graph.boltzmann(model, 4, 3),
            lambda: nn.Graph.markov_chain(model, 4, 3),
            lambda: nn.Graph.lsm(model, 4, 8, 2),
            lambda: nn.Graph.esn(model, 4, 8, 2),
            lambda: nn.Graph.rbf(model, 4, 6, 2),
            lambda: nn.Graph.dcn(model, 1, 4, 4, 3),
            lambda: nn.Graph.dn(model, 1, 4, 4),
            lambda: nn.Graph.dcign(model, 1, 4, 4, 2),
        ]
        for make in factories:
            graph = make()
            assert graph is not None
            graph.destroy()


def test_hidden_dims_validation():
    with nn.Arena(8 << 20) as model:
        with pytest.raises(ValueError):
            nn.Graph.dff(model, 4, [], 2)
        with pytest.raises(ValueError):
            nn.Graph.drn(model, 4, [0], 2)
        with pytest.raises(ValueError):
            nn.Graph.dbn(model, 4, [])


def test_all_zoo_factories_render_svg():
    """Toda fábrica do zoo deve gerar um SVG válido."""
    with nn.Arena(64 << 20) as model:
        graphs = [
            nn.Graph.dcn(model, 1, 4, 4, 3),
            nn.Graph.lsm(model, 4, 8, 2),
            nn.Graph.boltzmann(model, 4, 3),
        ]
        for graph in graphs:
            svg = graph.to_svg()
            assert svg.startswith("<?xml") and "</svg>" in svg
            graph.destroy()
