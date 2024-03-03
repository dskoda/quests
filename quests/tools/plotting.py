import os

import matplotlib as mpl
import matplotlib.pyplot as plt

FIGS_DIR = "../../figs"


COLORS = {"fcc": "#3C9645", "bcc": "#6064AE", "hcp": "#F16666", "other": "#808080"}


def savefig(fig, name, prefix=FIGS_DIR, **kwargs):
    dst = os.path.join(prefix, name)
    fig.savefig(dst, bbox_inches="tight", transparent=True, **kwargs)
    return dst
