# Checkpoints live on the mounted lustre disk (compute-node local disks are
# ephemeral). Override with the VFA_CHECKPOINT_ROOT env var if needed.
import os
pkl_root = os.environ.get("VFA_CHECKPOINT_ROOT",
                          "/mnt/lustre/ifm-lustre-prod-us-west1-a/av-data/checkpoints/VFA/")
EMBED_SIZE = 512
PAD = 0
SIL = 1
BOS = 53  # phoneme
EOS = 54
TOKEN_MAX_LEN = 80

