FROM alpine:edge
LABEL org.opencontainers.image.description "A collection of minimal and clean implementations of machine learning algorithms, designed to be lightweight, modular and extensible, making it easy to integrate with other frameworks and tools."
ENV PYTHONUNBUFFERED=1
RUN apk add --update --no-cache python3 g++ musl-dev && ln -sf python3 /usr/bin/python
RUN python3 -m ensurepip
RUN pip3 install --no-cache --upgrade pip setuptools
RUN pip3 install --break-system-packages --no-cache numpy autograd