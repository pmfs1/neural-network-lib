FROM ubuntu:devel
# LABEL org.opencontainers.image.maintainer "pmfs1"
# LABEL org.opencontainers.image.description "𝒜 𝒸𝑜𝓁𝓁𝑒𝒸𝓉𝒾𝑜𝓃 𝑜𝒻 𝓂𝒾𝓃𝒾𝓂𝒶𝓁 𝒶𝓃𝒹 𝒸𝓁𝑒𝒶𝓃 𝒾𝓂𝓅𝓁𝑒𝓂𝑒𝓃𝓉𝒶𝓉𝒾𝑜𝓃𝓈 𝑜𝒻 𝓂𝒶𝒸𝒽𝒾𝓃𝑒 𝓁𝑒𝒶𝓇𝓃𝒾𝓃𝑔, 𝒶𝓃𝒹 𝓈𝓉𝒶𝓉𝒾𝓈𝓉𝒾𝒸𝒶𝓁 𝒶𝓃𝒶𝓁𝓎𝓉𝒾𝒸𝒶𝓁 𝒶𝓁𝑔𝑜𝓇𝒾𝓉𝒽𝓂𝓈; 𝒹𝑒𝓈𝒾𝑔𝓃𝑒𝒹 𝓉𝑜 𝒷𝑒 𝓁𝒾𝑔𝒽𝓉𝓌𝑒𝒾𝑔𝒽𝓉, 𝓂𝑜𝒹𝓊𝓁𝒶𝓇 𝒶𝓃𝒹 𝑒𝓍𝓉𝑒𝓃𝓈𝒾𝒷𝓁𝑒, 𝓂𝒶𝓀𝒾𝓃𝑔 𝒾𝓉 𝑒𝒶𝓈𝓎 𝓉𝑜 𝒾𝓃𝓉𝑒𝑔𝓇𝒶𝓉𝑒 𝓌𝒾𝓉𝒽 𝑜𝓉𝒽𝑒𝓇 𝒻𝓇𝒶𝓂𝑒𝓌𝑜𝓇𝓀𝓈 𝒶𝓃𝒹 𝓉𝑜𝑜𝓁𝓈."
COPY ./REQUIREMENTS.txt /REQUIREMENTS.txt
RUN apt-get update -y && apt-get install -y apt-utils locales python3.11-full python3-pip wget nano lsb-release apt-transport-https sudo && apt-get full-upgrade -y && python3 -m pip install pylint --break-system-packages && python3 -m pip install -r REQUIREMENTS.txt --break-system-packages --no-cache-dir && rm -rf /var/lib/apt/lists/* \ && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8 && apt-get full-upgrade -y && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
ENV LANG en_US.utf8