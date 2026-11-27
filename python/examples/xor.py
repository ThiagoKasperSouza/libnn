"""Exemplo Python: treinar XOR com libnn e exportar o SVG da arquitetura.

Rode:  python3 python/examples/xor.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import libnn as nn


def main() -> None:
    nn.seed(4)
    out = Path("xor_python.svg")

    with nn.Arena(16 << 20) as model, nn.Arena(4 << 20) as scratch:
        net = nn.Graph.ff(model, input_dim=2, hidden_dim=8, output_dim=1)
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

        print(f"XOR: loss {first:.5f} -> {last:.5f}")
        net.render_svg(out)
        print(f"SVG: {out}")
        net.destroy()

    print("Concluído sem vazamentos.")


if __name__ == "__main__":
    main()
