_DOCKERFILE_BASE_AGNOSTIC = r"""
FROM --platform={platform} ubuntu:{ubuntu_version}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt update && apt install -y \
    wget \
    git \
    build-essential \
    libtool \
    automake \
    autoconf \
    tcl \
    bison \
    flex \
    cmake \
    python3 \
    python3-pip \
    python3-venv \
    python-is-python3 \
    jq \
    curl \
    locales \
    locales-all \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install pytest

# Install NVM
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
ENV NVM_DIR=/root/.nvm
RUN bash -c "source $NVM_DIR/nvm.sh && nvm install node"

# Install Claude Code
RUN bash -c "source $NVM_DIR/nvm.sh && npm install -g @anthropic-ai/claude-code"
ENV ANTHROPIC_API_KEY={anthropic_api_key}

RUN adduser --disabled-password --gecos 'dog' nonroot


ENTRYPOINT ["bash", "-c", "source $NVM_DIR/nvm.sh && exec \"$@\"", "--"]
CMD ["/bin/bash"]
"""

_DOCKERFILE_INSTANCE_AGNOSTIC = r"""FROM --platform={platform} {env_image_name}

COPY ./setup_repo.sh /root/
RUN /bin/bash /root/setup_repo.sh

WORKDIR /testbed/
"""
