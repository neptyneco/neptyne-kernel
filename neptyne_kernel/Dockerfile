FROM python:3.11.2 as build-wheels

ENV VIRTUAL_ENV=/opt/venv
ADD https://github.com/astral-sh/uv/releases/download/0.2.6/uv-installer.sh /install.sh
RUN chmod -R 655 /install.sh && /install.sh && rm /install.sh

COPY requirements.txt .
RUN /root/.cargo/bin/uv venv --system-site-packages ${VIRTUAL_ENV}
RUN /root/.cargo/bin/uv pip install --no-cache -r requirements.txt

FROM python:3.11.2-slim-bullseye

# We include git here to enable users to pip-install from git repos
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
    && apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /secrets

COPY --from=build-wheels /opt/venv /opt/venv

RUN mkdir /neptyne_kernel && touch /neptyne_kernel/codepanel.py
RUN mkdir /.streamlit && touch /.streamlit/config.toml

COPY requirements.txt ./

COPY formulas kernel/formulas
COPY widgets kernel/widgets
COPY tyne_model kernel/tyne_model
COPY neptyne_api kernel/neptyne_api
COPY kernel_globals kernel/kernel_globals
COPY kernel_entrypoint.sh kernel/
COPY *.py kernel/

ENV KERNEL_LANGUAGE python
ENV PATH="/opt/venv/bin:$PATH"

CMD /kernel/kernel_entrypoint.sh
