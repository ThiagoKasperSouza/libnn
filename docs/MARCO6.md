# Marco 6 — Gerador Visual SVG e Certificação do SDK

## 1. Escopo entregue

| Item do plano | Entrega |
| :-- | :-- |
| Exportador `nc_render.c` | `nc_render_svg()`, `nc_render_svg_string()` |
| Header público | `include/nc_render.h` |
| Paleta de cores do infográfico | 5 cores + marcadores de borda/triângulo |
| Testes | `tests/test_render.c` (43 verificações) |
| Aplicação de exemplo | `examples/apps/render-graph/` |
| Documentação | `docs/MARCO6.md`, README atualizado |
| Certificação | `make test` (8 binários) + `make asan` (zero erros/vazamentos) |

## 2. API de renderização

```c
int nc_render_svg(const NC_Graph *graph, const char *filename);
long nc_render_svg_string(const NC_Graph *graph, char *buffer, size_t buffer_size);
```

- `nc_render_svg`: escreve o SVG em arquivo; retorna 0 em sucesso, -1 em erro.
- `nc_render_svg_string`: renderiza em buffer de memória; se `buffer == NULL`, retorna o tamanho necessário.

## 3. Paleta de cores

A paleta reproduz as 5 cores do infográfico *"A mostly complete chart of Neural Networks"* (Fjodor van Veen, Asimov Institute):

| Tipo de célula | Cor | Hex |
| :-- | :-- | :-- |
| Entrada (Input, Backfed, Noisy) | Amarelo | `#FFD700` |
| Oculta (Hidden, Probabilistic, Spiking) | Verde | `#90EE90` |
| Saída (Output, Match I/O) | Vermelho | `#FF6347` |
| Recorrente (Recurrent, Memory, Different Memory) | Azul | `#87CEEB` |
| Convolucional (Kernel, Conv/Pool) | Rosa | `#FFB6C1` |

Variantes da legenda recebem marcadores adicionais:
- **Com borda** (Backfed, Noisy, Probabilistic, Spiking, Match, Memory, Different Memory, Conv/Pool): anel tracejado.
- **Com triângulo** (Noisy, Spiking, Different Memory): triângulo preto.

## 4. Layout e estética (infográfico)

O SVG reproduce a estética do infográfico de Fjodor van Veen:

- **Neurônios**: cada camada do `NC_Graph` é expandida em múltiplos círculos (um por neurônio), com no máximo 8 por camada para manter o diagrama legível. A quantidade é inferida das dimensões dos pesos da célula.
- **Conexões**: completamente conectadas entre camadas adjacentes (`fully-connected`), desenhadas como linhas finas com setas.
- **Marcadores das variantes**: células "com borda" recebem um círculo concêntrico interno; células "com triângulo" recebem um triângulo preenchido dentro do círculo — exatamente como na legenda do guia.
- **Sem rótulos de texto** dentro dos neurônios (o guia usa apenas cor + marcador).
- **Legenda completa** com as 13 variantes do infográfico, em duas colunas.
- Arestas recorrentes são desenhadas tracejadas e não participam do cálculo de camadas.

## 5. Certificação final

### Testes
```
make test → arena 357/357 · tensor 234124/234124 · autograd 62/62 · zoo 65/65
            spatial 789/789 · sequence 615/615 · energy 40/40 · render 43/43
```

### Sanitizers
```
make asan → 8 binários sem erros de memória nem vazamentos (ASan/UBSan)
```

### Aplicações
```
make run_apps → 11 aplicações sem vazamentos (incluindo render-graph)
```

### Cobertura de marcos
- Marco 1: Arena, tensores, matmul com tiling, RNG, testes de memória ✅
- Marco 2: Tape de autograd, Adam, P/FF/DFF/AE/DAE/SAE/DRN ✅
- Marco 3: im2col + GEMM, conv transposta, pooling, DCN/DN/DCIGN/GAN ✅
- Marco 4: Unrolling temporal, BPTT, RNN/LSTM/GRU/NTM ✅
- Marco 5: Energia, spiking e reservatórios (CD-k, Hebb, STDP, ridge, BMU, hinge) ✅
- Marco 6: Renderizador SVG e certificação ✅

## 6. Limitações conhecidas

- O layout é linear (camadas horizontais); redes muito profundas ou com muitas conexões cruzadas podem ter sobreposição visual.
- O SVG não inclui interatividade (hover, zoom); é um diagrama estático.
- A renderização de grafos com mais de 4096 células ou 256 camadas não é suportada.
- Arestas recorrentes (RNN/LSTM/GRU) são desenhadas mas não participam do cálculo de camadas.
- O exportador não renderiza pesos/valores de ativação, apenas a topologia.
- Grafos com ciclos (sem arestas recorrentes marcadas) podem ter layout incorreto.

## 7. Conclusão

O SDK `libnn` está completo com todos os 6 marcos implementados. A biblioteca oferece:
- 27 arquiteturas de redes neurais do infográfico de Fjodor van Veen
- 6 motores de treinamento (Backprop/Adam, GAN, CD-k, Ridge, Kohonen BMU, Hinge)
- Motor de renderização SVG com paleta de cores do infográfico
- Zero dependências externas além da biblioteca padrão C
- Certificação: zero erros de memória, zero vazamentos (ASan/UBSan)
