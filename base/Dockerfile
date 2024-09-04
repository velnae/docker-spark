FROM alpine:3.10

LABEL maintainer="Gezim Sejdiu <g.sejdiu@gmail.com>, Giannis Mouchakis <gmouchakis@gmail.com>"

ENV ENABLE_INIT_DAEMON false
ENV INIT_DAEMON_BASE_URI http://identifier/init-daemon
ENV INIT_DAEMON_STEP spark_master_init

ENV BASE_URL=https://archive.apache.org/dist/spark/
ENV SPARK_VERSION=3.3.0
ENV HADOOP_VERSION=3

COPY wait-for-step.sh /
COPY execute-step.sh /
COPY finish-step.sh /

# Instalar dependencias y preparar el entorno
RUN apk add --no-cache \
        curl \
        bash \
        openjdk8-jre \
        python3 \
        py3-pip \
        python3-dev \
        gfortran \
        build-base \
        musl-dev \
        linux-headers \
        libexecinfo-dev \
        lapack-dev \
        libgfortran \
        nss \
        libc6-compat \
        coreutils \
        procps

RUN ln -s /lib64/ld-linux-x86-64.so.2 /lib/ld-linux-x86-64.so.2 \
    && pip3 install --upgrade pip \
    && pip3 install numpy \
    && chmod +x *.sh \
    && wget ${BASE_URL}/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} spark \
    && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && cd /

# Dar permisos para ejecutar los scripts
RUN chmod +x /wait-for-step.sh && chmod +x /execute-step.sh && chmod +x /finish-step.sh

# Fijar el valor de PYTHONHASHSEED
ENV PYTHONHASHSEED 1
