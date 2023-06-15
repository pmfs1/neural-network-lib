FROM ubuntu:devel
COPY ./REQUIREMENTS.txt /REQUIREMENTS.txt
RUN apt-get update -y && apt-get install -y apt-utils locales python3.11-full python3-pip curl nano lsb-release apt-transport-https sudo init-system-helpers software-properties-common build-essential && apt-get full-upgrade -y && python3 -m pip install -r REQUIREMENTS.txt --break-system-packages --no-cache-dir && rm -rf /var/lib/apt/lists/* \ && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8 && apt-get full-upgrade -y && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
ENV LANG en_US.utf8
