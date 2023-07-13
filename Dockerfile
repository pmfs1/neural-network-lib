FROM alpine:edge
COPY ./REQUIREMENT[S].txt /REQUIREMENTS.txt
ENV PYTHONUNBUFFERED=1
RUN apk --no-cache --update add git
RUN apk add --update --no-cache python3 g++ musl-dev && ln -sf python3 /usr/bin/python
RUN python3 -m ensurepip
RUN pip3 install --no-cache --upgrade pip setuptools
RUN pip3 install pylint build
RUN if [ -f /REQUIREMENTS.txt ]; then pip3 install -r /REQUIREMENTS.txt; else pip3 install --break-system-packages --no-cache numpy autograd; fi