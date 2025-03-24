#!/bin/bash
# pip install -U huggingface_hub

LOCAL_DIR="./c4_data"

huggingface-cli download \
    --repo-type dataset \
    --cache-dir $LOCAL_DIR \
    --resume-download allenai/c4 \
    --include "en/c4-train*" \
    --local-dir-use-symlinks False

huggingface-cli download \
    --repo-type dataset \
    --cache-dir $LOCAL_DIR \
    --resume-download allenai/c4 \
    --include "en/c4-validation*" \
    --local-dir-use-symlinks False