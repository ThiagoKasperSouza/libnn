"""Gera a galeria interativa das 27 arquiteturas do infográfico.

Uso (na raiz do repositório)::

    make lib
    python3 python/examples/generate_gallery.py

Saídas:
    docs/gallery/svgs/<sigla>.svg   — 27 diagramas vetoriais
    docs/gallery/data.js            — metadados consumidos pelo index.html
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import libnn as nn  # noqa: E402

OUT = ROOT / "docs" / "gallery"
SVGS = OUT / "svgs"


# sigla, nome, família, fábrica, células, treinador, descrição, snippet
ARCHITECTURES = [
    ("P", "Perceptron", "feedforward",
     lambda a: nn.Graph.perceptron(a, 3, 1),
     "INPUT → OUTPUT", "Backprop (SGD)",
     "Classificador linear simples; uma camada de entrada ligada diretamente à saída.",
     "nn.Graph.perceptron(model, input_dim=3, output_dim=1)"),
    ("FF", "Feed Forward", "feedforward",
     lambda a: nn.Graph.ff(a, 3, 4, 2),
     "INPUT → HIDDEN → OUTPUT", "Backprop (Adam)",
     "Rede em camadas totalmente conectadas, com uma camada oculta.",
     "nn.Graph.ff(model, input_dim=3, hidden_dim=4, output_dim=2)"),
    ("RBF", "Radial Basis Network", "competitivo",
     lambda a: nn.Graph.rbf(a, 3, 5, 2),
     "INPUT → KERNEL(RBF) → OUTPUT", "Ridge / Hinge",
     "Usa funções de base radial como features antes da projeção final.",
     "nn.Graph.rbf(model, input_dim=3, centers=5, output_dim=2)"),
    ("DFF", "Deep Feed Forward", "feedforward",
     lambda a: nn.Graph.dff(a, 4, [8, 6], 3),
     "INPUT → HIDDEN×2 → OUTPUT", "Backprop (Adam)",
     "Perceptron multicamadas com duas ou mais camadas ocultas.",
     "nn.Graph.dff(model, 4, [8, 6], 3)"),
    ("RNN", "Recurrent Neural Network", "recorrente",
     lambda a: nn.Graph.rnn(a, 3, 5, 2),
     "INPUT → RECURRENT → OUTPUT", "BPTT",
     "Mantém um estado oculto que é realimentado a cada passo de tempo.",
     "nn.Graph.rnn(model, input_dim=3, hidden_dim=5, output_dim=2)"),
    ("LSTM", "Long / Short Term Memory", "recorrente",
     lambda a: nn.Graph.lstm(a, 3, 4, 3),
     "INPUT → MEMORY(4 portas) → OUTPUT", "BPTT (Adam)",
     "Célula de memória com portas de entrada, esquecimento e saída.",
     "nn.Graph.lstm(model, input_dim=3, hidden_dim=4, output_dim=3)"),
    ("GRU", "Gated Recurrent Unit", "recorrente",
     lambda a: nn.Graph.gru(a, 3, 4, 3),
     "INPUT → DIFFERENT_MEMORY → OUTPUT", "BPTT (Adam)",
     "Versão simplificada da LSTM, com menos portas e estado unificado.",
     "nn.Graph.gru(model, input_dim=3, hidden_dim=4, output_dim=3)"),
    ("AE", "Auto Encoder", "autoencoder",
     lambda a: nn.Graph.ae(a, 5, 3),
     "INPUT → HIDDEN → MATCH_IO", "Backprop (reconstrução)",
     "Aprende a codificar a entrada num vetor menor e reconstruí-la.",
     "nn.Graph.ae(model, input_dim=5, hidden_dim=3)"),
    ("VAE", "Variational Auto Encoder", "autoencoder",
     lambda a: nn.Graph.vae(a, 5, 3),
     "INPUT → PROBABILISTIC → MATCH_IO", "Backprop + KL",
     "Latente amostrado por reparametrização: z = mu + eps·exp(log_var/2).",
     "nn.Graph.vae(model, input_dim=5, latent_dim=3)"),
    ("DAE", "Denoising Auto Encoder", "autoencoder",
     lambda a: nn.Graph.dae(a, 5, 3, 0.2),
     "NOISY_INPUT → HIDDEN → MATCH_IO", "Backprop (reconstrução)",
     "Autoencoder treinado para remover ruído injetado na entrada.",
     "nn.Graph.dae(model, input_dim=5, hidden_dim=3, noise_level=0.2)"),
    ("SAE", "Sparse Auto Encoder", "autoencoder",
     lambda a: nn.Graph.sae(a, 5, 3, 0.05, 1.0),
     "INPUT → HIDDEN(sparsity) → MATCH_IO", "Backprop + penalidade L1",
     "Autoencoder com penalidade de esparsidade na camada oculta.",
     "nn.Graph.sae(model, 5, 3, sparsity_target=0.05, beta=1.0)"),
    ("MC", "Markov Chain", "energia",
     lambda a: nn.Graph.markov_chain(a, 4, 3),
     "INPUT → PROBABILISTIC_HIDDEN", "Energia (transição)",
     "Grafo estocástico de transições entre estados.",
     "nn.Graph.markov_chain(model, states=4, hidden=3)"),
    ("HN", "Hopfield Network", "energia",
     lambda a: nn.Graph.hopfield(a, 5),
     "BACKFED_INPUT → MATCH", "Hebbiano",
     "Memória associativa totalmente recorrente; recupera padrões ruidosos.",
     "nn.Graph.hopfield(model, dim=5)"),
    ("BM", "Boltzmann Machine", "energia",
     lambda a: nn.Graph.boltzmann(a, 4, 3),
     "BACKFED + PROBABILISTIC", "Divergência contrastiva",
     "Modelo de energia não-direcionado com unidades visíveis e ocultas.",
     "nn.Graph.boltzmann(model, visible=4, hidden=3)"),
    ("RBM", "Restricted Boltzmann Machine", "energia",
     lambda a: nn.Graph.rbm(a, 4, 3),
     "INPUT → PROBABILISTIC_HIDDEN", "CD-k",
     "Versão bipartite do BM, treinada por amostragem de Gibbs.",
     "nn.Graph.rbm(model, visible=4, hidden=3)"),
    ("DBN", "Deep Belief Network", "energia",
     lambda a: nn.Graph.dbn(a, 5, [4, 3]),
     "INPUT → PROBABILISTIC×2 → OUTPUT", "CD-k empilhado",
     "Várias RBMs empilhadas, treinadas camada por camada.",
     "nn.Graph.dbn(model, visible=5, hidden_dims=[4, 3])"),
    ("DCN", "Deep Convolutional Network", "convolucional",
     lambda a: nn.Graph.dcn(a, 1, 6, 6, 3),
     "INPUT → KERNEL → CONV/POOL → OUTPUT", "Backprop (Adam)",
     "Rede convolucional com compartilhamento de pesos e pooling.",
     "nn.Graph.dcn(model, channels=1, h=6, w=6, num_classes=3)"),
    ("DN", "Deconvolutional Network", "convolucional",
     lambda a: nn.Graph.dn(a, 1, 6, 6),
     "INPUT → KERNEL → MATCH(upsample)", "Backprop (Adam)",
     "Convolução transposta para reconstruir/projetar para dimensão maior.",
     "nn.Graph.dn(model, channels=1, h=6, w=6)"),
    ("DCIGN", "Deep Conv. Inverse Graphics Network", "autoencoder",
     lambda a: nn.Graph.dcign(a, 1, 6, 6, 2),
     "INPUT → CONV → LATENTE → CONV_T → MATCH", "Backprop (reconstrução)",
     "Autoencoder convolucional com latente probabilístico.",
     "nn.Graph.dcign(model, 1, 6, 6, latent_channels=2)"),
    ("GAN", "Generative Adversarial Network", "generativo",
     None,
     "GERADOR ⇄ DISCRIMINADOR", "Minimax adversarial",
     "Dois grafos acoplados: o gerador produz amostras e o discriminador as avalia.",
     "gan = nn.GAN(model, latent_dim=4, data_dim=6)\ngan.train_step(real, opt, scratch)"),
    ("LSM", "Liquid State Machine", "reservatorio",
     lambda a: nn.Graph.lsm(a, 3, 6, 2),
     "NOISY_INPUT → SPIKING_LIF → OUTPUT", "STDP + Ridge",
     "Reservatório de neurônios LIF com potenciais de membrana e spikes.",
     "nn.Graph.lsm(model, input_dim=3, neurons=6, output_dim=2)"),
    ("ELM", "Extreme Learning Machine", "reservatorio",
     lambda a: nn.Graph.elm(a, 3, 7, 2),
     "INPUT → HIDDEN(congelada) → OUTPUT", "Pseudoinversa ridge",
     "Camada oculta aleatória e fixa; apenas a saída é resolvida analiticamente.",
     "nn.Graph.elm(model, input_dim=3, neurons=7, output_dim=2)"),
    ("ESN", "Echo State Network", "reservatorio",
     lambda a: nn.Graph.esn(a, 3, 6, 2),
     "INPUT → RECURRENT(congelada) → OUTPUT", "Ridge",
     "Reservatório recorrente fixo com leitura linear treinada por ridge.",
     "nn.Graph.esn(model, input_dim=3, neurons=6, output_dim=2)"),
    ("DRN", "Deep Residual Network", "feedforward",
     lambda a: nn.Graph.drn(a, 4, [8, 6], 3),
     "INPUT → HIDDEN×2 (skip) → OUTPUT", "Backprop (Adam)",
     "Rede profunda com conexões residuais que pulam camadas.",
     "nn.Graph.drn(model, 4, [8, 6], 3)"),
    ("KN", "Kohonen Network (SOM)", "competitivo",
     lambda a: nn.Graph.kohonen(a, 3, 6),
     "INPUT → HIDDEN(competitiva)", "BMU competitivo",
     "Mapa auto-organizável: encontra o BMU e move a vizinhança.",
     "nn.Graph.kohonen(model, input_dim=3, neurons=6)"),
    ("NTM", "Neural Turing Machine", "recorrente",
     lambda a: nn.Graph.ntm(a, 4, 3, 5, 2),
     "INPUT → DIFFERENT_MEMORY → OUTPUT", "BPTT (conteúdo)",
     "Rede com memória externa endereçável por conteúdo.",
     "nn.Graph.ntm(model, 4, slots=3, mem_dim=5, output_dim=2)"),
    ("SVM", "Support Vector Machine", "competitivo",
     lambda a: nn.Graph.svm(a, 3),
     "INPUT → OUTPUT(margem)", "Hinge loss",
     "Classificador de margem máxima com perda hinge.",
     "nn.Graph.svm(model, input_dim=3)"),
]

FAMILIES = {
    "feedforward": "Feedforward",
    "autoencoder": "Autoencoders",
    "energia": "Modelos de energia",
    "convolucional": "Convolucionais",
    "recorrente": "Recorrentes e memória",
    "reservatorio": "Reservatórios",
    "competitivo": "Competitivos e kernel",
    "generativo": "Generativos",
}


def main() -> None:
    SVGS.mkdir(parents=True, exist_ok=True)
    nn.seed(17)

    records = []
    with nn.Arena(64 << 20) as model:
        for sigla, name, family, factory, cells, trainer, description, snippet in ARCHITECTURES:
            if factory is None:
                graph = nn.GAN(model, latent_dim=4, data_dim=6)
                renderable = graph.generator
            else:
                graph = factory(model)
                renderable = graph

            svg = renderable.to_svg()
            (SVGS / f"{sigla}.svg").write_text(svg, encoding="utf-8")
            records.append(
                {
                    "sigla": sigla,
                    "name": name,
                    "family": family,
                    "familyLabel": FAMILIES[family],
                    "cells": cells,
                    "trainer": trainer,
                    "description": description,
                    "snippet": snippet,
                    "svg": f"svgs/{sigla}.svg",
                }
            )
            print(f"  {sigla:6s} → {SVGS / (sigla + '.svg')}")

            if factory is None:
                graph.destroy()
            else:
                graph.destroy()

    data_js = (
        "// Gerado por python/examples/generate_gallery.py — não editar à mão.\n"
        "window.GALLERY_DATA = "
        + json.dumps(records, ensure_ascii=False, indent=2)
        + ";\n"
    )
    (OUT / "data.js").write_text(data_js, encoding="utf-8")
    print(f"\n{len(records)} SVGs e data.js gerados em {OUT}")


if __name__ == "__main__":
    main()
