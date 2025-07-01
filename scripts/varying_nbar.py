#!/usr/bin/env python3
"""
MPI script to run thy.py with config files distributed across ranks.
Each rank can handle multiple files sequentially.
"""

import os
import subprocess
import glob

class MPIComm(object):
    def __init__(self, start, Ntot, do_mpi):
        if do_mpi:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.size = 1
            self.rank = 0
        self.Ntot = Ntot
        
        delta = int(Ntot/self.size)
        self.iMin = self.rank*delta+start
        self.iMax = (self.rank+1)*delta+start
        if self.rank == self.size-1:
            self.iMax = Ntot+start
        self.tasks = range(self.iMin, self.iMax)

def run_config(config_file, rank):
    """Run thy.py with a single config file."""
    filename = os.path.basename(config_file)
    
    print(f"Rank {rank}: Starting {filename}", flush=True)
    
    cmd = [
        "python", "thy.py",
        "--config", config_file,
        "--config_dir", "." #"../configs/abacus/desi_nbars/"
    ]
    
    result = subprocess.run(cmd) #, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Rank {rank}: SUCCESS - {filename}")
        return True
    else:
        print(f"Rank {rank}: FAILED - {filename}")
        print(f"Rank {rank}: Error: {result.stderr}")
        return False

def main():
    # Find all config files
    config_files = sorted(glob.glob("../configs/abacus/desi_nbars/config_desi_base_*.yaml"))
    
    if not config_files:
        print("No config files found!")
        return
    
    # Initialize MPI communication
    mpi_comm = MPIComm(start=0, Ntot=len(config_files), do_mpi=True)

    #set up which GPU to use among the four GPUs available for each rank, modulo 4
    os.environ["CUDA_VISIBLE_DEVICES"] = str(mpi_comm.rank % 4)
    
    if mpi_comm.rank == 0:
        print(f"Found {len(config_files)} config files")
        print(f"Running on {mpi_comm.size} MPI ranks")

    
    # Each rank processes its assigned files
    success_count = 0
    total_files = len(mpi_comm.tasks)
    
    print(f"Rank {mpi_comm.rank}: Processing {total_files} files (indices {mpi_comm.iMin}-{mpi_comm.iMax-1})")
    
    for i, file_index in enumerate(mpi_comm.tasks):
        config_file = config_files[file_index]
        print(f"Rank {mpi_comm.rank}: File {i+1}/{total_files}")
        
        if run_config(config_file, mpi_comm.rank):
            success_count += 1
    
    print(f"Rank {mpi_comm.rank}: Completed {success_count}/{total_files} files successfully")

if __name__ == "__main__":
    main()