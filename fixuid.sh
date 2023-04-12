#!/bin/bash
set -e

addgroup --gid 1000 docker
adduser --uid 1000 \
        --ingroup docker \
        --home /home/docker \
        --shell /bin/sh \
        --disabled-password \
        --gecos "" \
        docker

apt update
apt install --no-install-recommends -y curl
USER=docker
GROUP=docker
curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.5.1/fixuid-0.5.1-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf -
chown root:root /usr/local/bin/fixuid 
chmod 4755 /usr/local/bin/fixuid
mkdir -p /etc/fixuid
printf "user: $USER\ngroup: $GROUP\n" > /etc/fixuid/config.yml
