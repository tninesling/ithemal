#!/usr/bin/env bash

BASH_HISTORY_FILE="$(dirname $0)/.docker_bash_history"
if [ ! -f "${BASH_HISTORY_FILE}" ]; then
    touch "${BASH_HISTORY_FILE}"
fi

DEFAULT_USER="ithemal"
USER="${DEFAULT_USER}"

if [ "$#" -eq 1 ]; then
    USER="${1}"
elif [ "$#" -gt 1 ]; then
    echo "Usage: ./docker_connect.sh [user] to log in as 'user' (default: '${DEFAULT_USER}')"
    exit 1
fi

function container_id() {
    docker ps -q --filter 'name=ithemal$'
}

CONTAINER="$(container_id)"

if [[ -z "${CONTAINER}" ]]; then
    read -p "Container is not currently running. Would you like to start it? (y/n) " -r

    if [[ !($REPLY =~ ^[Yy]) ]]; then
	echo "Not starting."
	exit 1
    fi

    # try to start the local X server if possible
    if [ ! -d "/tmp/.X11-unix" ]; then
	open -a XQuartz 2>/dev/null | :
    fi

    # allow local connections to X server (i.e. from Docker)
    xhost + "${HOSTNAME}" 2>/dev/null || :

    FAKE_X_SERVER=
    if [ ! -d "/tmp/.X11-unix" ]; then
	FAKE_X_SERVER="yep"
	mkdir /tmp/.X11-unix
    fi

    FAKE_AWS_DIR=
    AWS_DIR="$(bash -c 'echo ${HOME}/.aws')"
    if [ ! -d "${AWS_DIR}" ]; then
        FAKE_AWS_DIR="yep"
        mkdir -p "${AWS_DIR}"
    fi

    docker compose up -d --force-recreate

    CONTAINER="$(container_id)"

    if [[ "${FAKE_AWS_DIR}" == "yep" ]]; then
	# hide the evidence
	rmdir "${AWS_DIR}"
    fi

    if [[ "${FAKE_X_SERVER}" == "yep" ]]; then
	# hide the evidence
	rmdir /tmp/.X11-unix
    fi

    docker exec -u ithemal "${CONTAINER}" bash -lc 'ithemal/build_all.sh'
fi

docker exec -u "${USER}" -it "${CONTAINER}" /bin/bash # /home/ithemal/ithemal/aws/aws_utils/tmux_attach.sh
