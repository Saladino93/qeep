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

pt = 1./72.27 # Hundreds of years of history... 72.27 points to an inch.

jour_sizes = {"PRD": {"onecol": 246.*pt, "twocol": 510.*pt},
              "CQG": {"onecol": 374.*pt}, # CQG is only one column
              # Add more journals below. Can add more properties to each journal
             }

my_width = jour_sizes["PRD"]["onecol"]
# Our figure's aspect ratio
golden = (1 + 5 ** 0.5) / 2

size = (my_width, my_width/golden)

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
    '#F0E442',   # Light yellow
    '#000000',  # Black
]


# Alternative colorblind-friendly palette with warm/cool contrast
# Based on Paul Tol's scientific color schemes and IBM Design accessibility guidelines
COLORBLIND_PALETTE_2 = [
    '#1f77b4',  # Strong blue
    '#ff7f0e',  # Vivid orange
    '#2ca02c',  # Forest green
    '#d62728',  # Crimson red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Magenta
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf'   # Cyan
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

