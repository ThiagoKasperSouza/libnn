# GAN toy

## O que esta aplicação faz

Treina uma **Generative Adversarial Network** mínima: o gerador aprende a produzir vetores 3D parecidos com três centros de dados (`[1,0,0]`, `[0,1,0]`, `[0,0,1]`), enquanto o discriminador tenta distinguir amostras reais das geradas.

## Arquitetura

```
Gerador:      NOISE(4) → HIDDEN(32, ReLU) → HIDDEN(32, ReLU) → OUTPUT(3)
Discriminador: DATA(3) → HIDDEN(32, ReLU) → HIDDEN(32, ReLU) → OUTPUT(1)
```

Construtor: `nc_zoo_create_gan(&model, 4, 3)` → `NC_GAN { generator, discriminator }`.

## Dados

Oito amostras reais, uma em torno de cada centro, com ruído gaussiano de 0.02. O ruído de entrada do gerador é amostrado de `N(0,1)` a cada passo.

## Como treinar

```sh
make run_apps APP=gan-toy
```

`nc_train_gan_step(g, d, real, &scratch, &opt)` executa um passo minimax completo:

1. discriminador com dados reais (alvo `1`) e atualização;
2. discriminador com dados falsos (alvo `0`) e atualização;
3. gerador com o erro do discriminador (alvo `1`), usando `nc_graph_backward_no_zero()` para preservar a semente de gradiente e atualizando **somente** o gerador.

### Hiperparâmetros

| Parâmetro | Valor |
| :-- | :-- |
| Passos | 600 |
| Taxa de aprendizado | 0.002 |
| β₁ / β₂ / ε | 0.9 / 0.999 / 1e-8 |
| Dimensão latente | 4 |
| Semente RNG | 7 |

Use uma taxa menor que 0.01: com valores altos o objective de MSE faz as amplitudes explodirem.

## Saída esperada

Três amostras geradas próximas de um dos centros (por exemplo `[1.22, 0.03, 0.89]`). É comum haver **mode collapse** — as três amostras ficam parecidas — porque esta é uma GAN toy sem normalização nem ruído de rótulos.

## Como adaptar

- Aumente o número de passos ou a dimensionalidade latente.
- Troque o objective adversarial por BCE (seria necessário adicionar a perda ao autograd).
- Use `nc_zoo_create_dff` diretamente se quiser arquiteturas diferentes para gerador e discriminador.
