/*
 * Jenkinsfile — pipeline de CI para libnn.
 *
 * Estágios:
 *   1. Checkout (scm)
 *   2. Build & Test (gcc, make test)
 *   3. Sanitizers (ASan/UBSan, make asan)
 *   4. SonarQube (build-wrapper + sonar-scanner, quality gate obrigatório)
 *   5. Snyk (SAST via Snyk Code + auditoria de dependências)
 *
 * Credenciais necessárias no Jenkins (Kind: Secret text):
 *   - sonar-token : token de usuário do SonarQube (ou sonar-scanner)
 *   - snyk-token  : token da API do Snyk
 *
 * Configuração global esperada:
 *   - SonarQube server "sonarqube" (Manage Jenkins → Configure System)
 *   - Docker disponível no agente (socket /var/run/docker.sock)
 *
 * Variáveis de ambiente configuráveis (withEnv no nível do pipeline):
 *   SONAR_PROJECT_KEY, SONAR_ORGANIZATION, SNYK_ORG
 */

pipeline {
    agent none

    parameters {
        booleanParam(
            name: 'ENABLE_EXTERNAL_SCANS',
            defaultValue: false,
            description: 'Executa estágios SonarQube e Snyk (requer tokens reais). ' +
                         'Desligado no smoke build local.'
        )
    }

    options {
        buildDiscarder(logRotator(numToKeepStr: '20', artifactNumToKeepStr: '10'))
        timeout(time: 45, unit: 'MINUTES')
        disableConcurrentBuilds()
        skipDefaultCheckout(true)
    }

    environment {
        // Imagens pinadas por tag para reprodutibilidade.
        IMAGE_BUILD    = 'gcc:14-bookworm'
        IMAGE_SONAR    = 'libnn-ci-sonar:latest'
        IMAGE_SNYK     = 'snyk/snyk:alpine'

        SONAR_PROJECT_KEY = 'libnn'
        SONAR_ORGANIZATION = ''
        SNYK_ORG = ''
    }

    stages {
        stage('Checkout') {
            agent { docker { image "${env.IMAGE_BUILD}"; args '-u root:root' } }
            steps {
                deleteDir()
                checkout scm
                sh 'git rev-parse HEAD | tee build.commit'
                sh 'git log -1 --format="%h %ad %s" --date=short | tee build.info'
            }
        }

        stage('Build & Test') {
            agent { docker { image "${env.IMAGE_BUILD}"; args '-u root:root' } }
            steps {
                checkout scm
                sh '''
                    set -euo pipefail
                    make clean
                    make lib
                    make test
                '''
            }
            post {
                always {
                    archiveArtifacts artifacts: 'libnn.a, libnn_shared.so, build.commit, build.info',
                                     allowEmptyArchive: true
                }
            }
        }

        stage('Sanitizers (ASan/UBSan)') {
            agent { docker { image "${env.IMAGE_BUILD}"; args '-u root:root' } }
            steps {
                checkout scm
                sh '''
                    set -euo pipefail
                    make clean
                    make lib
                    make asan
                '''
            }
        }

        stage('Coverage (gcov/lcov)') {
            agent { docker { image "${env.IMAGE_BUILD}"; args '-u root:root' } }
            steps {
                checkout scm
                sh '''
                    set -euo pipefail
                    apt-get update -qq
                    apt-get install -y -qq --no-install-recommends lcov python3
                    make coverage
                '''
                // O estágio Sonar usa outro container/workspace: passa o XML por stash.
                stash includes: 'coverage.xml', name: 'sonar-coverage'
            }
            post {
                always {
                    archiveArtifacts artifacts: 'coverage.xml, coverage.info',
                                     allowEmptyArchive: true
                }
            }
        }

        stage('Python bindings') {
            agent { docker { image 'python:3.12-slim-bookworm'; args '-u root:root' } }
            steps {
                checkout scm
                sh '''
                    set -euo pipefail
                    apt-get update -qq && apt-get install -y -qq --no-install-recommends build-essential
                    make lib
                    python -m pip install --no-cache-dir -q pytest
                    python -m pip install --no-cache-dir -q -e python
                    python -m pytest python/tests -q
                '''
            }
        }

        stage('SonarQube') {
            when {
                expression { params.ENABLE_EXTERNAL_SCANS }
            }
            agent {
                docker {
                    image "${env.IMAGE_SONAR}"
                    args '''
                        -u root:root
                        -e SONAR_HOST_URL
                        -v ${WORKSPACE}:/workspace
                        -w /workspace
                    '''
                }
            }
            environment {
                SONAR_HOST_URL = credentials('sonarqube-url')
                SONAR_TOKEN = credentials('sonar-token')
            }
            steps {
                // checkout novamente: cada estágio tem workspace e container próprios.
                checkout scm
                // Recupera o XML de cobertura gerado no estágio Coverage.
                unstash 'sonar-coverage'
                withEnv([
                    "SONAR_PROJECT_KEY=${env.SONAR_PROJECT_KEY}",
                    "SONAR_ORGANIZATION=${env.SONAR_ORGANIZATION}"
                ]) {
                    sh '''
                        set -euo pipefail

                        # 1) Compila instrumentando os comandos do compilador.
                        make clean
                        build-wrapper-linux-x86-64 \
                            --out-dir bw-output \
                            make lib

                        # 2) Executa a suíte para gerar relatórios (se houver).
                        make test

                        # 3) Envia a análise ao SonarQube.
                        args=""
                        if [ -n "${SONAR_ORGANIZATION}" ]; then
                            args="$args -Dsonar.organization=${SONAR_ORGANIZATION}"
                        fi

                        sonar-scanner \
                            -Dsonar.host.url="${SONAR_HOST_URL}" \
                            -Dsonar.token="${SONAR_TOKEN}" \
                            -Dsonar.projectKey="${SONAR_PROJECT_KEY}" \
                            -Dsonar.cfamily.build-wrapper-output=bw-output \
                            -Dsonar.sources=src,include \
                            -Dsonar.tests=tests \
                            -Dsonar.exclusions=docs/gallery/**,python/**,examples/** \
                            -Dsonar.cfamily.threads=2 \
                            -Dsonar.cfamily.cache.enabled=true \
                            -Dsonar.coverageReportPaths=coverage.xml \
                            $args
                    '''
                }
                // Quality gate obrigatório: falha o estágio se não passar.
                script {
                    def qg = waitForQualityGate(abortPipeline: false, credentialsId: 'sonar-token')
                    if (qg.status != 'OK') {
                        error "SonarQube quality gate falhou: ${qg.status}"
                    }
                    echo "SonarQube quality gate: ${qg.status}"
                }
            }
        }

        stage('Snyk') {
            when {
                expression { params.ENABLE_EXTERNAL_SCANS }
            }
            agent {
                docker {
                    image "${env.IMAGE_SNYK}"
                    args '-u root:root -e SNYK_TOKEN -v ${WORKSPACE}:/workspace -w /workspace'
                }
            }
            environment {
                SNYK_TOKEN = credentials('snyk-token')
            }
            steps {
                checkout scm
                sh '''
                    set -euo pipefail
                    make lib

                    mkdir -p snyk-reports

                    # Snyk Open Source: dependências (projeto C puro → resultado vazio).
                    snyk monitor \
                        --org="${SNYK_ORG:-}" \
                        --project-name=libnn \
                        --json-file-output=snyk-reports/snyk-oss.json || true

                    # Snyk Code (SAST): varredura estática do código C/C++.
                    snyk code test \
                        --org="${SNYK_ORG:-}" \
                        --severity-threshold=high \
                        --json-file-output=snyk-reports/snyk-code.json
                '''
            }
            post {
                always {
                    archiveArtifacts artifacts: 'snyk-reports/*.json',
                                     allowEmptyArchive: true
                    sh 'rm -f snyk-reports/*.json || true'
                }
            }
        }
    }

    post {
        always {
            echo "Pipeline finalizada: ${currentBuild.currentResult} (#${currentBuild.number})"
        }
        failure {
            echo 'Pipeline falhou. Verifique os relatórios arquivados e o console.'
        }
        success {
            script {
                def msg = params.ENABLE_EXTERNAL_SCANS
                    ? 'Todos os estágios passaram: build, testes, sanitizers, cobertura, bindings, SonarQube e Snyk.'
                    : 'Smoke build passou: build, testes, sanitizers, cobertura e bindings. SonarQube/Snyk pulados.'
                echo msg
            }
        }
    }
}
