"""
Common utilities for admin report generation.
"""

import warnings

import matplotlib.pyplot as plt

# Suppress matplotlib warnings about non-GUI backend
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def save_chart(png_filename, fig=None, dpi=300, debug=False):
    """
    Save a matplotlib figure as PNG.

    Args:
        png_filename: Path to save the PNG file
        fig: matplotlib figure (if None, uses current figure)
        dpi: Resolution for the saved image
        debug: If True, print debug information

    Returns:
        str: The filename of the saved PNG chart
    """
    if fig is None:
        fig = plt.gcf()

    fig.tight_layout()
    fig.savefig(png_filename, dpi=dpi, bbox_inches="tight")

    if debug:
        print(f"Chart saved as: {png_filename}")

    plt.close(fig)
    return png_filename
