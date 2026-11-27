# Smoke build local do Jenkins

Este documento registra como subir um Jenkins local em Docker, criar o job
`libnn-smoke` e executar a pipeline sem depender de tokens reais de SonarQube
ou Snyk.

## 1. Subir o Jenkins

```sh
cd jenkins/local
docker compose up -d --build
```

A imagem (`libnn-jenkins-local:latest`) inclui:
- Jenkins LTS (JDK 17) com setup wizard desabilitado
- Plugins: Pipeline, Docker Workflow, Sonar, Configuration as Code, Job DSL,
  Credentials Binding e Git
- Docker CLI (para os agentes `agent { docker { ... } }`)
- JCasC (`casc.yaml`) que cria:
  - usuário `admin` / senha `admin`
  - credenciais dummy: `sonar-token`, `snyk-token`, `sonarqube-url`
  - job `libnn-smoke` apontando para `/workspace` (o repo montado), branch `main`
  - parâmetro `ENABLE_EXTERNAL_SCANS` (padrão `false`)
- System property `hudson.plugins.git.GitSCM.ALLOW_LOCAL_CHECKOUT=true` para
  permitir o checkout de `/workspace`

Acesso: http://localhost:8080 (admin/admin).

Aguarde até ver `Jenkins is fully up and running` em `docker logs libnn-jenkins`.

## 2. Disparar o smoke build

O Jenkins exige CSRF crumb para POSTs. Use um cookie jar:

```sh
CRUMB=$(curl -s -u admin:admin -c /tmp/jcookies \
  http://localhost:8080/crumbIssuer/api/json | \
  python3 -c 'import sys,json;print(json.load(sys.stdin)["crumb"])')

curl -X POST -u admin:admin -b /tmp/jcookies \
  -H "Jenkins-Crumb: $CRUMB" \
  'http://localhost:8080/job/libnn-smoke/buildWithParameters?ENABLE_EXTERNAL_SCANS=false'
```

HTTP 201 + `Location: /queue/item/N/` = build enfileirado.

## 3. Resultado esperado

| Estágio | Resultado |
| :-- | :-- |
| Checkout | clona `/workspace` @ `main` |
| Build & Test | 8 suítes C (357+234124+62+65+789+615+40+52) |
| Sanitizers | `make asan` — zero erros/vazamentos |
| Coverage | `make coverage` — 88,7% linhas, 65,1% branches |
| Python bindings | 17 testes pytest |
| SonarQube | **pulado** (`when: !ENABLE_EXTERNAL_SCANS`) |
| Snyk | **pulado** (`when: !ENABLE_EXTERNAL_SCANS`) |

Artefatos arquivados: `build.commit`, `build.info`, `libnn.a`,
`libnn_shared.so`, `coverage.info`, `coverage.xml`.

## 4. Resultado obtido

Build `#7`: **SUCCESS**

```
Pipeline finalizada: SUCCESS (#7)
Smoke build passou: build, testes, sanitizers, cobertura e bindings.
SonarQube/Snyk pulados.
Finished: SUCCESS
```

## 5. Correções feitas no Jenkinsfile durante o smoke

O smoke revelou e corrigiu estes problemas:

| Problema | Correção |
| :-- | :-- |
| `timestamps()` sem plugin timestamper | Removido |
| Concatenação de strings em bloco `post` | Uso de `script { def msg = ...; echo msg }` |
| `set -o pipefail` em `sh` (dash) | Trocado por `set -eu` |
| Imagem `python:3.12-slim-bookworm` indisponível no daemon local | Estágio Python usa a mesma imagem gcc + venv |
| `deleteDir()` e checkout falhavam com arquivos root-owned | Removido `deleteDir()`; `chown -R 1000:1000 .` ao fim dos estágios root |
| Checkout local bloqueado por segurança | `ALLOW_LOCAL_CHECKOUT=true` no JAVA_OPTS |

## 6. Habilitar SonarQube/Snyk depois

Substitua as credenciais dummy por valores reais em
`jenkins/local/casc.yaml` (ou via UI) e dispare com:

```
ENABLE_EXTERNAL_SCANS=true
```

O job então executa os dois estágios e o quality gate do SonarQube passa a
ser obrigatório.

## 7. Limpeza

```sh
cd jenkins/local
docker compose down -v
```

Isso remove o container, a rede e o volume `jenkins_home`.
