#!/usr/bin/env python3
"""
MPI script to run thy.py with config files distributed across ranks.
Each rank can handle multiple files sequentially.
"""

import os
import subprocess
import glob
import argparse
import pathlib

def parse_args():
    parser = argparse.ArgumentParser(description='Run thy.py with multiple configurations')
    parser.add_argument('--force', action='store_true', 
                       help='Force regeneration of results even if they exist')
    return parser.parse_args()

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

def check_existing_results(config_file):
    """Check if results already exist for this configuration."""
    # Extract the base name from config file (e.g., 'desi_base_11_2' from 'config_desi_base_11_2.yaml')
    base_name = os.path.basename(config_file).replace('config_', '').replace('.yaml', '')
    
    # Construct the results directory path
    results_dir = os.path.join('/users/odarwish/qeep/results', base_name)
    
    # Check if directory exists and has more than 5 files
    if os.path.exists(results_dir):
        files = list(pathlib.Path(results_dir).glob('*'))
        return len(files) > 5
    return False

def run_config(config_file, rank, gpu_num, force=False):
    """Run thy.py with a single config file."""
    filename = os.path.basename(config_file)
    
    # Check if results already exist
    if not force and check_existing_results(config_file):
        print(f"Rank {rank}: Skipping {filename} - results already exist")
        return True
    
    print(f"Rank {rank}: Starting {filename}", flush=True)
    
    cmd = [
        "python", "thy.py",
        "--config", config_file,
        "--config_dir", ".", #"../configs/abacus/desi_nbars/"
        "--gpu", gpu_num
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
    # Parse command line arguments
    args = parse_args()
    
    # Find all config files
    config_files = sorted(glob.glob("../configs/abacus/desi_nbars/config_desi_base_*.yaml"))
    
    if not config_files:
        print("No config files found!")
        return
    
    # Initialize MPI communication
    mpi_comm = MPIComm(start=0, Ntot=len(config_files), do_mpi=True)

    #set up which GPU to use among the four GPUs available for each rank, modulo 4
    gpu_num = str(mpi_comm.rank % 4)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    
    if mpi_comm.rank == 0:
        print(f"Found {len(config_files)} config files")
        print(f"Running on {mpi_comm.size} MPI ranks")
        if args.force:
            print("Force flag enabled - will regenerate existing results")

    # Each rank processes its assigned files
    success_count = 0
    total_files = len(mpi_comm.tasks)
    
    print(f"Rank {mpi_comm.rank}: Processing {total_files} files (indices {mpi_comm.iMin}-{mpi_comm.iMax-1})")
    
    for i, file_index in enumerate(mpi_comm.tasks):
        config_file = config_files[file_index]
        print(f"Rank {mpi_comm.rank}: File {i+1}/{total_files}")
        
        if run_config(config_file, mpi_comm.rank, gpu_num, force=args.force):
            success_count += 1
    
    print(f"Rank {mpi_comm.rank}: Completed {success_count}/{total_files} files successfully")

if __name__ == "__main__":
    main()