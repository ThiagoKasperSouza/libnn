# CI — Jenkins + SonarQube + Snyk

Pipeline declarativa para o `libnn`, com estágios isolados em containers Docker.

## Pré-requisitos no Jenkins

### Plugins
- **Pipeline** (já embutido no Jenkins moderno)
- **Docker Pipeline** (`docker` agent)
- **SonarQube Scanner for Jenkins** (`waitForQualityGate`)

### Credenciais (Manage Jenkins → Credentials → System → Global credentials)

| ID | Kind | Descrição |
| :-- | :-- | :-- |
| `sonar-token` | Secret text | Token de usuário do SonarQube (My Account → Security → Generate Tokens) |
| `snyk-token` | Secret text | Token da API do Snyk (Settings → API Token) |
| `sonarqube-url` | Secret text | URL do servidor SonarQube (ex.: `https://sonar.exemplo.com`) |

### Configuração global
- **Manage Jenkins → Configure System → SonarQube servers**: adicione um servidor chamado `sonarqube`.
- Docker disponível para o usuário do Jenkins (geralmente membership no grupo `docker`).

### Variáveis de ambiente (opcionais)
Ajuste no topo do `Jenkinsfile` ou via `withEnv`:
- `SONAR_PROJECT_KEY` (padrão: `libnn`)
- `SONAR_ORGANIZATION` (deixe vazio para SonarQube on-prem; preencha para SonarCloud)
- `SNYK_ORG` (ID da organização no Snyk)

## Estágios

| Estágio | Imagem | O que faz |
| :-- | :-- | :-- |
| Checkout | `gcc:14-bookworm` | Faz checkout e registra commit/info |
| Build & Test | `gcc:14-bookworm` | `make clean && make lib && make test`; arquiva `.a`/`.so` |
| Sanitizers | `gcc:14-bookworm` | `make asan` (ASan/UBSan + leak detection) |
| Python bindings | `python:3.12-slim-bookworm` | Instala pytest, `make lib`, `pip install -e python`, roda os testes |
| SonarQube | `libnn-ci-sonar:latest` | `build-wrapper` + `sonar-scanner`; **quality gate obrigatório** |
| Snyk | `snyk/snyk:alpine` | `snyk monitor` (Open Source) + `snyk code test` (SAST, threshold high) |

## Quality gate

O estágio SonarQube chama `waitForQualityGate` e falha o build se o status
não for `OK`. Configure o quality gate no servidor SonarQube (por exemplo:
"Novo código: 0 bugs, 0 vulnerabilidades, cobertura ≥ 80%").

## Snyk

- **Open Source**: o projeto é C puro, sem `package.json`/`requirements.txt` de
  produção; o `snyk monitor` registra o projeto e gera `snyk-oss.json` (vazio é esperado).
- **Code (SAST)**: analisa `src/` e `include/` com `--severity-threshold=high`.
  O relatório JSON é arquivado como artifact.

## Construir a imagem Sonar

```sh
docker build -f ci/Dockerfile.sonar -t libnn-ci-sonar:latest .
```

A imagem herda de `sonarsource/sonar-scanner-cli:latest` (Amazon Linux 2023)
e instala gcc/make/glibc-devel via yum. O `build-wrapper` do CFamily é
específico da versão do servidor: por padrão o Dockerfile baixa o pacote
público do SonarCloud (SHA-256 fixado em `BUILD_WRAPPER_SHA256`). O download
usa `curl --proto '=https' --proto-redir '=https' --tlsv1.2` e rejeita URLs
que não comecem com `https://`, portanto redirecionamentos para HTTP são
bloqueados. Para SonarQube on-prem, substitua **ambos** `BUILD_WRAPPER_URL` e
`BUILD_WRAPPER_SHA256` pelos valores do zip disponível em Administration →
Tools & Settings → C/C++.

Verificação local rápida:

```sh
docker run --rm libnn-ci-sonar:latest /bin/sh -c \
  'sonar-scanner --version; build-wrapper-linux-x86-64 --version'
```

## Conversor de cobertura

`ci/lcov_to_sonar.py` converte o `.info` do lcov para o XML genérico que o
SonarQube consome via `sonar.coverageReportPaths`:

```sh
python3 ci/lcov_to_sonar.py coverage.info coverage.xml .
```

O terceiro argumento (opcional) é o diretório-raiz removido dos caminhos `SF:`.

## Rodar localmente (sem Jenkins)

```sh
# Build e testes
make clean && make lib && make test && make asan

# Análise Sonar (requer servidor e token)
build-wrapper-linux-x86-64 --out-dir bw-output make lib
sonar-scanner -Dsonar.token="$SONAR_TOKEN"

# Snyk (requer token)
snyk code test --severity-threshold=high
snyk monitor --project-name=libnn
```

## Solução de problemas

- **`waitForQualityGate` falha com timeout**: aumente o timeout do estágio ou
  verifique se o plugin SonarQube está configurado com o mesmo `sonar-token`.
- **`build-wrapper` não encontrado**: a imagem `libnn-ci-sonar` precisa ser
  construída no Docker daemon que o Jenkins usa.
- **`snyk code test` falha com 401**: token inválido ou `SNYK_ORG` incorreto.
- **Sanitizers falham em container**: o `gcc:14-bookworm` já traz libasan/ubsan.
