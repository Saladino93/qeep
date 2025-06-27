import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import LogLocator, LogFormatter, AutoMinorLocator
from matplotlib.gridspec import GridSpec
from matplotlib import patheffects
import matplotlib as mpl

# ---- CONFIGURATION ----
# Use golden ratio for figure dimensions
GOLDEN_RATIO = (5**0.5 - 1) / 2
FIG_WIDTH = 5  # inches
FIG_HEIGHT = FIG_WIDTH * GOLDEN_RATIO
DPI = 300

# Check for and configure LaTeX if available (optional but professional)
# Uncomment this if you have LaTeX installed
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"],
# })

# If not using LaTeX, use a clean serif font
# Try to use TeX fonts that are included with matplotlib
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm"  # Use Computer Modern math font
})

# Define a modern, colorblind-friendly palette with higher contrast
# Based on colorblindness-friendly scientific palettes like viridis
# and ones recommended by Nature and Science publications
COLORBLIND_PALETTE = [
    '#0072B2',  # Blue
    '#D55E00',  # Orange
    '#009E73',  # Green
    '#CC79A7',  # Pink
    '#56B4E9',  # Light blue
    '#E69F00',  # Yellow
    '#F0E442'   # Light yellow
    '#000000',  # Black
]

# More descriptive and professionally formatted names
names = {
    "n": r"$\mathcal{D}$", 
    "s": r"$\mathrm{\mathcal{S}}_{\mathrm{symm}}$",
    "t": r"$\mathrm{\mathcal{T}}_{\mathrm{symm}}$",
    "g": r"$\mathrm{\mathcal{G}}_{\mathrm{symm}}$",
    "ga": r"$\mathrm{\mathcal{G}}_{\mathrm{asymm}}$",
    "sa": r"$\mathrm{\mathcal{S}}_{\mathrm{asymm}}$",
    "ta": r"$\mathrm{\mathcal{T}}_{\mathrm{asymm}}$",
}

