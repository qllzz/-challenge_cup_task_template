#!/bin/bash

DOCKER="${DOCKER:-podman}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DELIVERY_DIR="$SCRIPT_DIR"
LEJU_DIR="$SCRIPT_DIR/leju-kuavo-challenge-cup-2026"

if [ ! -d "$LEJU_DIR" ]; then
    echo -e "\033[31mError: leju-kuavo-challenge-cup-2026/ not found under $SCRIPT_DIR\033[0m"
    echo "Make sure the leju repo is cloned alongside this script."
    exit 1
fi

FAKEHOME="$DELIVERY_DIR/fakehome"
CCACHE_DIR="$FAKEHOME/.ccache"
ROS_HOME="$FAKEHOME/.ros"
LEJUCONFIG_DIR="$FAKEHOME/.config/lejuconfig"
mkdir -p "$CCACHE_DIR" "$ROS_HOME" "$LEJUCONFIG_DIR"

ROBOT_VERSION=52
DIR_HASH=$(echo "$DELIVERY_DIR" | md5sum | cut -c1-8)
echo "Delivery dir  : $DELIVERY_DIR"
echo "Leju workspace: $LEJU_DIR"
echo "Directory hash: $DIR_HASH"

CONTAINER_NAME="kuavo_challenge_container_${DIR_HASH}"
IMAGE_NAME="kuavo_challenge_cup_2026:with-mesa"

if ! ${DOCKER} image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo -e "\033[31mError: Docker image '${IMAGE_NAME}' not found.\033[0m"
    echo "Load it first:"
    echo "  gunzip -c kuavo_challenge_cup_2026_latest.tar.gz | ${DOCKER} load"
    exit 1
fi

show_container_info() {
    local div_line="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "\n$div_line"
    echo -e "📌 \033[34mContainer Info\033[0m: $CONTAINER_NAME"
    echo -e "📂 \033[32mMounted Volumes\033[0m:"
    ${DOCKER} inspect -f '{{range .Mounts}}   {{.Source}} → {{.Destination}}{{println}}{{end}}' $CONTAINER_NAME
    echo -e "$div_line\n"
}

_setup_robon_version() {
    ${DOCKER} exec $CONTAINER_NAME bash -c "
        if grep -q '^export ROBOT_VERSION=' /root/.zshrc; then
            sed -i 's/^export ROBOT_VERSION=.*/export ROBOT_VERSION=52/' /root/.zshrc
        elif grep -q 'export ROBOT_VERSION=' /root/.zshrc; then
            sed -i 's/export ROBOT_VERSION=.*/export ROBOT_VERSION=52/' /root/.zshrc
        else
            echo '' >> /root/.zshrc
            echo '# Auto-configured by docker script' >> /root/.zshrc
            echo 'export ROBOT_VERSION=52' >> /root/.zshrc
        fi
    "
}

_fix_hosts() {
    ${DOCKER} exec $CONTAINER_NAME bash -c "
        grep -q \"\$(hostname)\" /etc/hosts || echo '127.0.0.1  '\$(hostname) >> /etc/hosts
    "
}

# check for existing container
EXISTING_CONTAINER=$(${DOCKER} ps -aq -f name=^/${CONTAINER_NAME}$)

if [[ -n "$EXISTING_CONTAINER" ]]; then
    echo "Container '${CONTAINER_NAME}' already exists."

    CONTAINER_STATUS=$(${DOCKER} inspect -f '{{.State.Status}}' $CONTAINER_NAME 2>/dev/null)

    if [[ "$CONTAINER_STATUS" == "exited" ]] || [[ "$CONTAINER_STATUS" == "created" ]]; then
        echo "Starting container '$CONTAINER_NAME' ..."
        ${DOCKER} start $CONTAINER_NAME
        sleep 1
    elif [[ "$CONTAINER_STATUS" == "running" ]]; then
        echo "Container '$CONTAINER_NAME' is already running."
    else
        echo -e "\033[31mError: Container status is '$CONTAINER_STATUS'. Please check container manually.\033[0m"
        exit 1
    fi

    echo "Updating ROBOT_VERSION to 52 in container..."
    _setup_robon_version

    echo "Fixing /etc/hosts..."
    _fix_hosts

    show_container_info
    echo "Exec into container '$CONTAINER_NAME' ..."
    ${DOCKER} exec -it -e CHALLENGE_TIMER_DISABLE=1 $CONTAINER_NAME zsh
    exit 0
fi

# create new container
if [[ -z "$EXISTING_CONTAINER" ]]; then
    echo "Creating a new container '${CONTAINER_NAME}' based on image '${IMAGE_NAME}' ..."

    # prefer Wayland, fall back to X11
    DISPLAY_ARGS=()
    if [[ -n "$WAYLAND_DISPLAY" ]] && [[ -S "$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY" ]]; then
        echo "Using Wayland display: $WAYLAND_DISPLAY"
        DISPLAY_ARGS+=(-e WAYLAND_DISPLAY="$WAYLAND_DISPLAY")
        DISPLAY_ARGS+=(-e XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR")
        DISPLAY_ARGS+=(-e GDK_BACKEND=wayland)
        DISPLAY_ARGS+=(-e QT_QPA_PLATFORM=wayland)
        DISPLAY_ARGS+=(--volume="$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY:$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY:Z")
        if [[ -n "$DISPLAY" ]]; then
            DISPLAY_ARGS+=(-e DISPLAY="$DISPLAY")
        fi
    elif [[ -n "$DISPLAY" ]]; then
        echo "Using X11 display: $DISPLAY"
        xhost +
        DISPLAY_ARGS+=(-e DISPLAY="$DISPLAY")
        DISPLAY_ARGS+=(--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw")
    fi

    ${DOCKER} run -it --net host \
        --name $CONTAINER_NAME \
        --privileged \
        -v /dev/dri:/dev/dri \
        -v "$ROS_HOME:/root/.ros" \
        -v "$CCACHE_DIR:/root/.ccache" \
        -v "$LEJUCONFIG_DIR:/root/.config/lejuconfig" \
        -v "$LEJU_DIR:/root/kuavo_ws" \
        -v "$DELIVERY_DIR/scripts:/root/kuavo_ws/src/challenge_cup_task_template/scripts" \
        -v "$DELIVERY_DIR/src:/root/kuavo_ws/src/challenge_cup_task_template/src" \
        -e HOME=/root \
        -e ROBOT_VERSION=52 \
        -e CHALLENGE_TIMER_DISABLE=1 \
        --group-add=dialout \
        --cap-add=sys_nice \
        --ipc=host \
        "${DISPLAY_ARGS[@]}" \
        ${IMAGE_NAME} \
        bash -c "
            grep -q \"\$(hostname)\" /etc/hosts || echo '127.0.0.1  '\$(hostname) >> /etc/hosts

            if grep -q '^export ROBOT_VERSION=' /root/.zshrc; then
                sed -i 's/^export ROBOT_VERSION=.*/export ROBOT_VERSION=52/' /root/.zshrc
            elif grep -q 'export ROBOT_VERSION=' /root/.zshrc; then
                sed -i 's/export ROBOT_VERSION=.*/export ROBOT_VERSION=52/' /root/.zshrc
            else
                echo '' >> /root/.zshrc
                echo '# Auto-configured by docker script' >> /root/.zshrc
                echo 'export ROBOT_VERSION=52' >> /root/.zshrc
            fi
            exec zsh
        "
fi
