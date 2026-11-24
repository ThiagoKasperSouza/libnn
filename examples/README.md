# Exemplos de aplicação da libnn

Cada aplicação vive na sua própria pasta, com `main.c` e um `README.md` que explica o problema, a arquitetura, os dados e como treinar.

| Aplicação | Marco | O que demonstra |
| :-- | :-- | :-- |
| [`tensor-basics`](./apps/tensor-basics/README.md) | 1 | Arenas, tensores, RNG, matmul, convolução e pooling (sem treino) |
| [`xor-classifier`](./apps/xor-classifier/README.md) | 2 | Classificação XOR com feedforward e Adam |
| [`denoising-autoencoder`](./apps/denoising-autoencoder/README.md) | 2 | Autoencoder que remove ruído gaussiano |
| [`image-classifier`](./apps/image-classifier/README.md) | 3 | CNN (im2col + pooling) em imagens sintéticas 8×8 |
| [`gan-toy`](./apps/gan-toy/README.md) | 3 | GAN mínima com treino adversarial alternado |
| [`sequence-copy`](./apps/sequence-copy/README.md) | 4 | LSTM com BPTT numa tarefa de cópia com atraso |
| [`ntm-memory`](./apps/ntm-memory/README.md) | 4 | Memória externa endereçável por conteúdo |

## Como compilar e executar

```sh
make apps                # compila todas
make run_apps            # executa todas em sequência
make run_apps APP=xor-classifier   # executa apenas uma
```

Os executáveis ficam em `examples/apps/<aplicacao>/<aplicacao>`. Cada `main.c` também pode ser compilado diretamente:

```sh
cc -std=c11 -O2 -Wall -Wextra -Iinclude examples/apps/xor-classifier/main.c libnn.a -lm -o /tmp/xor
/tmp/xor
```

Todos os exemplos usam dados **sintéticos gerados em memória**: a libnn não tem dependências de leitura de arquivos ou datasets externos.
