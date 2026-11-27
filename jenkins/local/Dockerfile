# Jenkins local para smoke build do libnn.
#
# Inclui:
#   - Jenkins LTS (JDK 17)
#   - plugins: pipeline, docker-workflow, sonar, configuration-as-code, job-dsl
#   - Docker CLI (para os agentes `agent { docker { ... } }`)
#   - setup wizard desabilitado + JCasC para credenciais/job automáticos
#
# Uso: docker compose -f jenkins/local/docker-compose.yml up -d --build

FROM jenkins/jenkins:lts-jdk17

USER root

# Docker CLI dentro do controller (o socket é montado pelo compose).
RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        ca-certificates curl gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg \
       | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/debian bookworm stable" \
       > /etc/apt/sources.list.d/docker.list \
    && apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        docker-ce-cli \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER jenkins

# Plugins necessários para o Jenkinsfile + JCasC.
COPY jenkins/local/plugins.txt /usr/share/jenkins/ref/plugins.txt
RUN jenkins-plugin-cli --plugin-file /usr/share/jenkins/ref/plugins.txt

ENV JAVA_OPTS="-Djenkins.install.runSetupWizard=false -Dhudson.plugins.git.GitSCM.ALLOW_LOCAL_CHECKOUT=true"
ENV CASC_JENKINS_CONFIG="/usr/share/jenkins/ref/casc.yaml"
COPY jenkins/local/casc.yaml /usr/share/jenkins/ref/casc.yaml
