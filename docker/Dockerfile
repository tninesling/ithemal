FROM debian:bookworm-slim

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        build-essential             \
        cmake                       \
        coreutils                   \
        curl                        \
        git                         \
        gnupg                       \
        libboost-all-dev            \
        libsqlite3-dev              \
        libtorch-dev                \
        locales                     \
        man                         \
        nasm                        \
        pkg-config                  \
        pv                          \
        python3-dev                 \
        qtbase5-dev qt5-qmake       \
        sqlite3                     \
        sudo                        \
        tmux                        \
        unzip                       \
        vim                         \
        wget                        \
        zip

# install mysql
RUN apt-get update && apt-get install -y gnupg curl ca-certificates && \
        mkdir -p /etc/apt/keyrings && \
        curl -fsSL https://repo.mysql.com/RPM-GPG-KEY-mysql-2023 | gpg --dearmor -o /etc/apt/keyrings/mysql.gpg && \
        chmod 644 /etc/apt/keyrings/mysql.gpg && \
        echo "deb [signed-by=/etc/apt/keyrings/mysql.gpg] http://repo.mysql.com/apt/debian/ bookworm mysql-8.0" > /etc/apt/sources.list.d/mysql.list && \
        apt-get update && \
        DEBIAN_FRONTEND=noninteractive apt-get install -y libmysqlclient-dev mysql-server

COPY docker_my.cnf /etc/my.cnf

RUN curl -sL https://github.com/DynamoRIO/dynamorio/releases/download/release_7_0_0_rc1/DynamoRIO-Linux-7.0.0-RC1.tar.gz | tar xz -C /opt
ENV DYNAMORIO_HOME "/opt/DynamoRIO-Linux-7.0.0-RC1"
# tar will not give you `755 & ~umask` because tar is evil
RUN chown -R root:root "${DYNAMORIO_HOME}" && \
        find "${DYNAMORIO_HOME}" -type d -exec chmod 755 {} \; && \
        find "${DYNAMORIO_HOME}" -type f -exec chmod 644 {} \;

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

ARG HOST_UID=1000
ENV HOST_UID $HOST_UID

RUN groupadd -g 1000 ithemal && useradd -m -s /bin/bash -r -u $HOST_UID -g ithemal ithemal
USER ithemal
WORKDIR /home/ithemal

RUN wget --quiet -O /tmp/install_conda.sh https://repo.anaconda.com/miniconda/Miniconda3-py312_25.1.1-2-Linux-x86_64.sh && \
        sh /tmp/install_conda.sh -b -p /home/ithemal/conda && \
        rm /tmp/install_conda.sh
COPY env /tmp/docker_env
RUN /home/ithemal/conda/bin/conda create --name ithemal --file /tmp/docker_env


# non-login shell
RUN /bin/echo 'export PATH=/home/ithemal/.local/bin:/home/ithemal/conda/bin:$PATH' >> /home/ithemal/.bash_profile && \
        /bin/echo 'source activate ithemal' >> /home/ithemal/.bash_profile && \
        /bin/echo 'export PYTHONPATH="/home/ithemal/ithemal/learning/pytorch"' >> /home/ithemal/.bash_profile

# login shell
RUN /bin/echo 'export PATH=/home/ithemal/.local/bin:/home/ithemal/conda/bin:$PATH' >> /home/ithemal/.bashrc && \
        /bin/echo 'source activate ithemal' >> /home/ithemal/.bashrc && \
        /bin/echo 'export PYTHONPATH="/home/ithemal/ithemal/learning/pytorch"' >> /home/ithemal/.bashrc

RUN bash -lc 'pip install --upgrade --user jupyter pyhamcrest pip; jupyter notebook --generate-config'

COPY notebook_config.patch /tmp/_docker_notebook_conf.patch
RUN sed -i "s/#c.NotebookApp.ip = 'localhost'/c.NotebookApp.ip = '0.0.0.0'/" /home/ithemal/.jupyter/jupyter_notebook_config.py