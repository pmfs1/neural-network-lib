FROM ubuntu:devel
COPY ./REQUIREMENTS.txt /REQUIREMENTS.txt
RUN apt-get update -y && apt-get install -y locales python3.11-minimal python3-pip && python3 -m pip install pylint --break-system-packages --no-cache-dir && python3 -m pip install -r REQUIREMENTS.txt --break-system-packages --no-cache-dir && rm -rf /var/lib/apt/lists/* \ && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8