_DOCKERFILE_ENV_AGNOSTIC = r"""FROM --platform={platform} {base_image_key}

WORKDIR /testbed/
"""

_DOCKERFILE_INSTANCE_AGNOSTIC = r"""FROM --platform={platform} {env_image_name}

COPY ./setup_repo.sh /root/
RUN /bin/bash /root/setup_repo.sh

WORKDIR /testbed/
"""
