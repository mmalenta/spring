# We should be able to use a lighter base
FROM nvidia/cuda:9.2-devel-ubuntu18.04

# This is required, otherwise pydedm fails on install
ENV LC_ALL=C.UTF-8

RUN echo 'deb http://us.archive.ubuntu.com/ubuntu trusty main multiverse' >> /etc/apt/sources.list && \
    apt-get -y check && \
    apt-get -y update && \
    apt-get -y install apt-utils && \
    apt-get -y upgrade

RUN apt-get update && \
    apt-get --no-install-recommends --allow-unauthenticated -y install \
    software-properties-common \
    git \
    python3-dev \
    python3-pip && \
    python3 -m pip install --upgrade pip setuptools wheel

ENV SOFTHOME="/software/"

WORKDIR $SOFTHOME

ARG POST_BRANCH=dev
RUN git clone https://github.com/mmalenta/spring.git && \
    cd spring && \
    git checkout $POST_BRANCH && \
    git log -2 && \
    pip install .

RUN git clone https://bitbucket.org/vmorello/mtcutils.git && \
    cd mtcutils && \
    make install