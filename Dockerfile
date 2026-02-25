ARG FROM_IMAGE=cmsml/cmsml:3.12
FROM ${FROM_IMAGE}

WORKDIR /app

ADD . .

RUN echo "=======================================" && \
    echo "Installing hgcal-offline-reco" && \
    echo "Current time:" $(date) && \
    echo "=======================================" && \
    python -m pip install --upgrade pip && \
    python -m pip install -e . --verbose && \
    echo "Installation complete!"

