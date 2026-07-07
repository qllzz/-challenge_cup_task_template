FROM kuavo_challenge_cup_2026:latest

RUN add-apt-repository -y ppa:kisak/turtle && \
    apt-get install -y \
        libgl1-mesa-dri \
        libglx-mesa0 \
        libegl-mesa0 \
        libglapi-mesa \
        mesa-libgallium \
        mesa-va-drivers \
        mesa-vulkan-drivers \
        mesa-vdpau-drivers \
        libosmesa6 \
        mesa-utils