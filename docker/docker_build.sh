#!/usr/bin/env bash

docker build --build-arg HOST_UID=$(id -u) -t ithemal:latest "$(dirname $0)"
