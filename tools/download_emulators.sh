#!/bin/bash

# Get current directory
MDIR=$(pwd)

# ============================ #
#   Require user-specified path
# ============================ #

# Expect one argument: the install path
if [ $# -ne 1 ]; then
    echo "Usage: $0 /path/where/emulators/will/be/installed"
    exit 1
fi

USER_PATH="$1"

# Expand ~ if used
USER_PATH=$(eval echo "$USER_PATH")

# Validate
if [ -z "$USER_PATH" ]; then
    echo "ERROR: Invalid path. Aborting."
    exit 1
fi

# Create hmfast_data directory inside the user path
TARGET_DIR="${USER_PATH}/hmfast_data"
mkdir -p "$TARGET_DIR"

echo "Installing emulator repositories into: $TARGET_DIR"

# Clone emulators
cd "$TARGET_DIR" || exit 1
git clone https://github.com/cosmopower-organization/lcdm.git
git clone https://github.com/cosmopower-organization/mnu.git
git clone https://github.com/cosmopower-organization/mnu-3states.git
git clone https://github.com/cosmopower-organization/ede.git
git clone https://github.com/cosmopower-organization/neff.git
git clone https://github.com/cosmopower-organization/wcdm.git

# Get absolute path
PATH_TO_HMFAST_DATA=$(realpath "$TARGET_DIR")

echo "Emulators installed at: $PATH_TO_HMFAST_DATA"

