#!/usr/bin/env python3
"""
Script to generate multiple YAML configuration files with varying nbar values.
Reads a base config file and creates multiple versions with different names and nbar values.
"""

import yaml
import numpy as np
import os
from pathlib import Path

def load_base_config(filename):
    """Load the base YAML configuration file."""
    with open(filename, 'r') as file:
        return yaml.safe_load(file)

def generate_nbar_values(start=1e-8, end=100, num_points=20):
    """Generate logarithmically spaced nbar values."""
    return np.logspace(np.log10(start), np.log10(end), num_points)

def create_config_variant(base_config, name_suffix, nbar_value):
    """Create a variant of the base config with modified name and nbar values."""
    # Create a deep copy of the base config
    config = yaml.safe_load(yaml.dump(base_config))
    
    # Update the name
    config['name'] = f"desi_base_{name_suffix}"
    
    # Update nbar_A and nbar_B to be equal
    config['number_density']['nbar_A'] = float(nbar_value)
    config['number_density']['nbar_B'] = float(nbar_value)
    
    return config

def save_config(config, filename):
    """Save configuration to YAML file."""
    with open(filename, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)

def main():
    # Configuration
    base_filename = "config_desi_nbars.yaml"
    output_dir = "."
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Check if base file exists
    if not os.path.exists(base_filename):
        print(f"Error: Base file '{base_filename}' not found!")
        print("Please make sure the file exists in the current directory.")
        return
    
    try:
        # Load base configuration
        print(f"Loading base configuration from '{base_filename}'...")
        base_config = load_base_config(base_filename)
        
        # Generate nbar values
        nbar_values = generate_nbar_values(start=1e-8, end=100, num_points=20)
        
        print(f"Generating {len(nbar_values)} configuration files...")
        print(f"nbar values range from {nbar_values[0]:.2e} to {nbar_values[-1]:.2e}")
        
        # Generate and save each configuration
        for i, nbar_value in enumerate(nbar_values, 1):
            # Create modified config
            modified_config = create_config_variant(base_config, i, nbar_value)
            
            # Generate filename
            output_filename = os.path.join(output_dir, f"config_desi_base_{i}.yaml")
            
            # Save to file
            save_config(modified_config, output_filename)
            
            print(f"Created: {output_filename} (nbar = {nbar_value:.2e})")
        
        print(f"\nSuccessfully generated {len(nbar_values)} configuration files in '{output_dir}' directory.")
        
        # Print summary of nbar values
        print("\nSummary of nbar values:")
        for i, nbar_value in enumerate(nbar_values, 1):
            print(f"  desi_base_{i}: {nbar_value:.2e}")
            
    except FileNotFoundError:
        print(f"Error: Could not find the base configuration file '{base_filename}'")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()