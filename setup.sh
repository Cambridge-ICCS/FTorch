#!/usr/bin/bash
# ============================================================================ #
# Activate the Python virtual environment used by FTorch.                      #
# ============================================================================ #

if [ -z "${VIRTUAL_ENV}" ]
then
        source /path/to/ftorch/bin/activate
fi
export Torch_DIR=/path/to/torch/
