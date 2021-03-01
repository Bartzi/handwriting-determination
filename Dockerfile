FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
	git \
	software-properties-common \
	pkg-config \
	unzip \
    zsh \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libgl1

RUN pip3 install cython

ARG UNAME=user
ARG UID=1000
ARG GID=100

RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/zsh $UNAME

RUN mkdir /data
ARG BASE=/app
RUN mkdir -p ${BASE}

RUN pip3 install cupy-cuda111
COPY requirements_docker.txt ${BASE}/requirements.txt

WORKDIR ${BASE}
RUN pip3 install -r requirements.txt

USER $UNAME
RUN git clone https://github.com/robbyrussell/oh-my-zsh.git ~/.oh-my-zsh
RUN cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc

CMD ["/bin/zsh"]
