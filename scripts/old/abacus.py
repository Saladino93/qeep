import sys
# Add your lenscarf site-packages to the path
sys.path.append('/users/odarwish/lenscarf/lib/python3.12/site-packages')

import numpy as np

sys.path.append('/users/odarwish/nbodykit/')
import nbodykit
from nbodykit.lab import ArrayCatalog

import matplotlib.pyplot as plt

import time
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

path_to_sim = "/capstor/scratch/cscs/odarwish/ABACUS/AbacusSummit_base_c000_ph000/halos/z0.100/" #halo_info/halo_info_000.asdf"
s = time.time()
cat = CompaSOHaloCatalog(path_to_sim, subsamples=dict(A=True,pos=True))

print("Time", time.time()-s)
print("Done!")

