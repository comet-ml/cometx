"""
Common utilities for admin report generation.
"""

import warnings

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

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


def get_distinct_colors(n):
    """
    Generate n distinct colors for charts.

    Uses a combination of colormaps to ensure distinct colors even for large n.
    For n <= 20, uses tab20. For larger n, uses a continuous colormap.

    Args:
        n: Number of colors needed

    Returns:
        List of RGBA color tuples
    """
    if n <= 0:
        return []
    elif n <= 20:
        # Use tab20 colormap which has 20 distinct colors
        return [cm.tab20(i) for i in range(n)]
    else:
        # For more than 20, use a continuous colormap (hsv) and sample evenly
        # hsv provides good color separation
        colormap = cm.get_cmap("hsv")
        # Sample evenly across the colormap, avoiding the very end which wraps
        indices = np.linspace(0, 0.9, n)
        return [colormap(i) for i in indices]
