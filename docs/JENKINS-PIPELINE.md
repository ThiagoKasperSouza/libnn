# Pipeline Jenkins — SonarQube + Snyk

## 1. Visão geral

O [`Jenkinsfile`](../Jenkinsfile) define uma pipeline declarativa com `agent none`:
cada estágio roda em seu próprio container Docker, com o workspace sincronizado
pelo plugin Docker Pipeline. Isso evita instalar ferramentas no host do Jenkins
e garante reprodutibilidade.

| Estágio | Imagem | Comando principal | Artefatos |
| :-- | :-- | :-- | :-- |
| Checkout | `gcc:14-bookworm` | `checkout scm` | `build.commit`, `build.info` |
| Build & Test | `gcc:14-bookworm` | `make clean && make lib && make test` | `libnn.a`, `libnn_shared.so` |
| Sanitizers | `gcc:14-bookworm` | `make asan` | — |
| Coverage | `gcc:14-bookworm` | `make coverage` (gcov + lcov) | `coverage.xml`, `coverage.info` |
| Python bindings | `python:3.12-slim-bookworm` | `pytest python/tests -q` | — |
| SonarQube | `libnn-ci-sonar:latest` | build-wrapper + sonar-scanner | quality gate |
| Snyk | `snyk/snyk:alpine` | `snyk monitor` + `snyk code test` | `snyk-reports/*.json` |

## 2. Decisões tomadas

| Decisão | Escolha | Motivo |
| :-- | :-- | :-- |
| Execução | Docker por estágio | Ambientes isolados e reprodutíveis |
| SonarQube C | build-wrapper + sonar-scanner | Caminho oficial do CFamily; detecta bugs reais no código C |
| Snyk | Code (SAST) + Open Source | SAST é útil em C puro; Open Source documenta a ausência de dependências |
| Credenciais | Jenkins Credentials por ID | Tokens nunca aparecem no Jenkinsfile |
| Quality gate | Obrigatório | `waitForQualityGate` falha o build se o status não for `OK` |

## 3. Configuração do Jenkins

### 3.1 Plugins
- Pipeline
- Docker Pipeline
- SonarQube Scanner for Jenkins

### 3.2 Credenciais (Secret text)

| ID | Uso |
| :-- | :-- |
| `sonar-token` | Autenticação com o servidor SonarQube e `waitForQualityGate` |
| `snyk-token` | Autenticação com a API do Snyk |
| `sonarqube-url` | URL base do servidor SonarQube |

### 3.3 SonarQube server
Em **Manage Jenkins → Configure System → SonarQube servers**, registre um
servidor chamado `sonarqube`. O `waitForQualityGate` usa essa configuração.

### 3.4 Cobertura (gcov/lcov → SonarQube)

O estágio **Coverage** instrumenta o código com `--coverage -O0 -g`, executa
toda a suíte, captura os dados com `lcov` e converte para o formato
**SonarQube Generic Coverage** usando `ci/lcov_to_sonar.py`. O resultado
`coverage.xml` é:

1. Arquivado como artifact do build;
2. Enviado ao estágio SonarQube via `stash`/`unstash` (cada estágio tem seu
   próprio container/workspace);
3. Passado ao scanner com `-Dsonar.coverageReportPaths=coverage.xml`.

O conversor lê os registros `SF:` (arquivo), `DA:` (linha/execuções) e `BRDA:`
(branch/taken) do `.info`, remove o prefixo do workspace e gera
`<coverage version="1">` com `<file path="src/...">` e `<lineToCover>`.

Para rodar localmente (requer `lcov` e `python3`):

```sh
make coverage
head -20 coverage.xml
```

### 3.5 Variáveis
No topo do Jenkinsfile:

```groovy
SONAR_PROJECT_KEY   = 'libnn'
SONAR_ORGANIZATION  = ''     // preencher apenas para SonarCloud
SNYK_ORG            = ''     // ID da organização no Snyk
```

## 4. Imagem Sonar personalizada

A imagem parte do `sonarsource/sonar-scanner-cli:latest` (Amazon Linux 2023) e adiciona gcc/make e o build-wrapper. Construa-a
imagem no mesmo Docker daemon usado pelo Jenkins:

```sh
docker build -f ci/Dockerfile.sonar -t libnn-ci-sonar:latest .
```

Se o seu SonarQube for on-prem, baixe o `build-wrapper-linux-x86.zip` da
própria instância (**Administration → Tools & Settings → C/C++**) e coloque
em `ci/build-wrapper-linux-x86.zip` antes do build. Isso garante que a versão
do wrapper seja compatível com o servidor.

## 5. Análise SonarQube

O estágio executa:

1. `build-wrapper-linux-x86-64 --out-dir bw-output make lib` — captura os
   comandos de compilação de todos os `.c`.
2. `make test` — executa a suíte (os resultados não são enviados como cobertura,
   pois o projeto não gera relatório de cobertura ainda).
3. `sonar-scanner` com:
   - `sonar.sources=src,include`
   - `sonar.tests=tests`
   - `sonar.exclusions=docs/gallery/**,python/**,examples/**`
   - `sonar.cfamily.build-wrapper-output=bw-output`
4. `waitForQualityGate` — falha o build se o gate não estiver `OK`.

O arquivo [`sonar-project.properties`](../sonar-project.properties) permite
reproduzir a análise localmente.

## 6. Snyk

- **Open Source**: `snyk monitor --project-name=libnn`. Como o projeto é C
  puro, não há manifestos de dependência — o relatório sai vazio, mas o
  monitoramento fica registrado.
- **Code (SAST)**: `snyk code test --severity-threshold=high`. Qualquer
  vulnerabilidade de severidade alta ou crítica falha o estágio.

Os relatórios JSON são arquivados como artifacts do build.

## 7. Pipeline local (sem Jenkins)

```sh
# Build, testes e sanitizers
make clean && make lib && make test && make asan

# SonarQube
build-wrapper-linux-x86-64 --out-dir bw-output make lib
sonar-scanner -Dsonar.token="$SONAR_TOKEN"

# Snyk
snyk code test --severity-threshold=high
snyk monitor --project-name=libnn
```

## 8. Limitações

- O quality gate depende das regras configuradas no servidor SonarQube; este
  repositório não define thresholds.
- A cobertura depende de `lcov` estar disponível no estágio Coverage
  (instalado via `apt-get` dentro do container `gcc:14-bookworm`).
- `snyk monitor` para um projeto sem manifesto pode exigir `--file` apontando
  para um arquivo de dependências quando houver um.
- A imagem `libnn-ci-sonar:latest` precisa existir no daemon Docker do Jenkins
  antes da primeira execução.

## 9. Próximos passos

1. Integrar `gcov`/`lcov` e publicar cobertura no SonarQube.
2. Adicionar `snyk iac test` se um Dockerfile/CMake for introduzido.
3. Publicar a galeria (`docs/gallery/`) como artifact do build.
