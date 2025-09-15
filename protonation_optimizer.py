#!/usr/bin/env python3
"""
QM-based Histidine Protonation State Optimizer (HID vs HIE only)
Uses Open Babel for initial hydrogen placement and xTB for energy calculations.
Includes special handling for truncated residues and charge correction for ASP, GLU, LYS and ARG.
"""

import os
import sys
import tempfile
import subprocess
import pandas as pd
import numpy as np
import argparse
from Bio.PDB import PDBParser
import mdtraj as md
from tabulate import tabulate
import re
import shutil
import datetime

# Ensure we are running on Python 3.7+ (f-strings and features used below)
if sys.version_info < (3, 7):
    raise SystemExit(
        "This script requires Python 3.7+; run with 'python3' or upgrade your environment. "
        f"Detected Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}."
    )


def normalize(vector):
    """Return the normalized vector"""
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    return vector
def get_orthogonal_vector(vector):
    """Return a vector orthogonal to the input vector"""
    # Create an arbitrary vector not parallel to the input
    v = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normalize(vector), v)) > 0.9:
        v = np.array([0.0, 1.0, 0.0])
    
    # Create orthogonal vector using cross product
    orthogonal = np.cross(vector, v)
    return normalize(orthogonal)

def get_tetrahedral_vectors(bond_vector):
    """Generate tetrahedral arrangement of vectors"""
    bond_norm = normalize(bond_vector)
    v1 = get_orthogonal_vector(bond_norm)
    v2 = np.cross(bond_norm, v1)
    v2 = normalize(v2)
    
    # Standard tetrahedral angle is 109.5 degrees
    cos_angle = -1/3  # cosine of 109.5 degrees
    sin_angle = np.sqrt(1 - cos_angle**2)
    
    # Create vectors at tetrahedral positions
    tetrahedral = []
    tetrahedral.append(-bond_norm)  # First position along the bond but inverted
    
    # Calculate other three positions
    for i in range(3):
        angle = i * 2 * np.pi / 3  # 120 degree spacing
        vec = cos_angle * (-bond_norm) + sin_angle * (np.cos(angle) * v1 + np.sin(angle) * v2)
        tetrahedral.append(normalize(vec))
    
    return tetrahedral

def find_histidines(pdb_file):
    """Find histidine residues in any chain with proper residue numbering"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('input', pdb_file)
    
    histidines = []
    for model in structure:
        for chain in model:
            chain_id = chain.id  # Get actual chain ID
            for residue in chain:
                res_id = residue.id[1]  # Tuple 
                res_name = residue.get_resname().strip()
                
                if res_name in ['HIS', 'HID', 'HIE']:
                    histidines.append({
                        'chain': chain_id,
                        'resid': res_id,
                        'resname': res_name,
                        'residue_obj': residue  # Store the actual residue object
                    })
    return histidines

def run_propka_predictions(pdb_file):
    """Run PropKa3 and return dictionary of ALL titratable residue pKa values with proper parsing"""
    print("Running PropKa3 for pKa predictions...")
    pwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(pdb_file)))
    try:
        # Try the standard entry point first
        result = subprocess.run(
            ["propka3", pdb_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except FileNotFoundError:
        print("PropKa3 not found in PATH; skipping pKa predictions.")
        os.chdir(pwd)
        return {}
    
    # Check for errors
    if result.returncode != 0:
        print(f"PropKa3 Error: {result.stderr}")
        os.chdir(pwd)
        return {}

    pka_values = {}
    try:
        propka_out_file = pdb_file.split('/')[-1].split('.')[0]+'.pka'
        with open(propka_out_file, 'r') as propkaout:
            # Flag to track when we find the summary section
            in_summary = False
            header_skipped = False
            
            for line in propkaout:
                line = line.strip()
                
                # Check if we've found the summary section
                if "SUMMARY OF THIS PREDICTION" in line:
                    in_summary = True
                    continue
                    
                # Skip the header line that comes after "SUMMARY OF THIS PREDICTION"
                if in_summary and not header_skipped:
                    header_skipped = True
                    continue
                
                # Process data lines in the summary section
                if in_summary:
                    # Check if we've reached the end of the summary section
                    if not line or "------" in line or "Free energy" in line:
                        break
                        
                    parts = line.split()
                    
                    # Ensure the line has the expected format and contains a titratable residue
                    if len(parts) >= 4:
                        resname = parts[0]
                        
                        # Only process known titratable residue types
                        if resname in ["ASP", "GLU", "HIS", "CYS", "TYR", "LYS", "ARG"]:
                            try:
                                resid = int(parts[1])
                                chain = parts[2]
                                pka = float(parts[3].replace('*',''))  # Handle any asterisks in pKa values
                                
                                # Store in dictionary
                                pka_values[(chain, resid)] = {
                                    'residue': resname,
                                    'pKa': pka
                                }
                            except (ValueError, IndexError) as e:
                                print(f"Warning: Skipped malformed line: {line} - {e}")
                                continue
    except Exception as e:
        print(f"Error parsing PropKa output: {e}")
    
    os.chdir(pwd)
    return pka_values


def determine_protonation_state(residue, pka, ph):
    """Determine appropriate residue name based on pKa and pH"""
    delta = pka - ph

    if residue == "HIS":
        if delta > 1.0:  # Strongly protonated
            return "HIP"  # Doubly protonated
        elif delta < -1.0:  # Strongly deprotonated
            # Will be determined by QM in the original pipeline
            return "HIS"
        else:
            # In the buffer zone, require QM verification
            return "HIS"

    elif residue == "ASP":
        if delta > 1.0:  # Strongly protonated
            return "ASH"  # Protonated form
        else:
            return "ASP"  # Deprotonated form

    elif residue == "GLU":
        if delta > 1.0:  # Strongly protonated
            return "GLH"  # Protonated form
        else:
            return "GLU"  # Deprotonated form

    elif residue == "LYS":
        if delta < -1.0:  # Strongly deprotonated
            return "LYN"  # Neutral form
        else:
            return "LYS"  # Protonated form

    elif residue == "ARG":
        if delta < -3.0:  # Very strongly deprotonated (rare)
            return "ARN"  # Neutral form (rarely used)
        else:
            return "ARG"  # Protonated form

    elif residue == "CYS":
        if delta < -1.0:  # Strongly deprotonated
            return "CYM"  # Negative form
        else:
            return "CYS"  # Neutral form

    elif residue == "TYR":
        if delta < -1.0:  # Strongly deprotonated
            return "TYD"  # Negative form (or TYM in some force fields)
        else:
            return "TYR"  # Neutral form

    # Default: return original name
    return residue

def update_pdb_with_protonations(input_pdb, output_pdb, protonation_map, hydrogen_map=None):
    """Update PDB with optimized residue names and histidine hydrogen positions"""
    print(f"Updating PDB file with optimized protonation states: {output_pdb}")

    if hydrogen_map is None:
        hydrogen_map = {}

    # Define standard residues and their protonation variants
    standard_to_variant = {
        "ASP": ["ASP", "ASH"],
        "GLU": ["GLU", "GLH"],
        "HIS": ["HIS", "HID", "HIE", "HIP"],
        "LYS": ["LYS", "LYN"],
        "ARG": ["ARG", "ARN"],
        "CYS": ["CYS", "CYM"],
        "TYR": ["TYR", "TYD", "TYM"]
    }

    # Create reverse mapping to identify titratable residues
    titratable_variants = {}
    for std, variants in standard_to_variant.items():
        for var in variants:
            titratable_variants[var] = std

    with open(input_pdb, 'r') as f_in, open(output_pdb, 'w') as f_out:
        current_key = None
        atom_serial = 1

        def write_hydrogens(key):
            nonlocal atom_serial
            for h_line in hydrogen_map.get(key, []):
                new_resname = protonation_map.get(key, h_line[17:20].strip())
                line = h_line[:6] + f"{atom_serial:5d}" + h_line[11:17] + new_resname.ljust(3) + h_line[20:]
                f_out.write(line)
                atom_serial += 1

        for line in f_in:
            if line.startswith(('ATOM', 'HETATM')):
                chain_id = line[21]
                residue_num = int(line[22:26].strip())
                key = (chain_id, residue_num)

                if key != current_key:
                    if current_key is not None:
                        write_hydrogens(current_key)
                    current_key = key

                resname = line[17:20].strip()
                new_resname = protonation_map.get(key, resname)

                if key in hydrogen_map:
                    element = line[76:78].strip()
                    atom_name = line[12:16].strip()
                    if element == 'H' or atom_name.startswith('H'):
                        continue  # Skip existing hydrogens for this residue

                line = line[:6] + f"{atom_serial:5d}" + line[11:17] + new_resname.ljust(3) + line[20:]
                f_out.write(line)
                atom_serial += 1
            else:
                if current_key is not None and line.startswith('TER'):
                    write_hydrogens(current_key)
                    current_key = None
                f_out.write(line)

        if current_key is not None:
            write_hydrogens(current_key)


def extract_environment(input_pdb, histidine, cutoff):
    """Extract COMPLETE RESIDUES within cutoff of target histidine"""
    try:
        # Load trajectory
        traj = md.load(input_pdb)
        
        print(f"  Loaded structure with {traj.n_residues} residues, {traj.n_atoms} atoms")
        
        # Find chain index for the target chain
        chain_id = histidine['chain']
        chain_index = None
        for chain in traj.topology.chains:
            if chain.chain_id == chain_id:
                chain_index = chain.index
                break
        
        if chain_index is None:
            print(f"  ERROR: Chain {chain_id} not found in topology")
            return None, None
        
        # Find target histidine residue
        target_resid = histidine['resid']
        target_residue = None
        for res in traj.topology.residues:
            if res.chain.index == chain_index and res.resSeq == target_resid:
                target_residue = res
                break
        
        if target_residue is None:
            print(f"  ERROR: Residue {target_resid} not found in chain {chain_id}")
            return None, None
            
        # Get atoms from target residue
        target_atoms = [atom.index for atom in target_residue.atoms]
        
        # Convert cutoff from Angstroms to nanometers
        cutoff_nm = cutoff / 10.0  # 1 Å = 0.1 nm
        
        # Find all atoms within cutoff of ANY atom in the target residue
        neighbor_atoms = set()
        for atom_i in target_atoms:
            pairs = md.compute_neighbors(traj, cutoff_nm, np.array([atom_i]))
            
            # Handle the case where compute_neighbors returns a single integer
            for pair in pairs:
                if isinstance(pair, np.integer):
                    # Single neighbor as numpy.int64
                    neighbor_atoms.add(int(pair))
                else:
                    # Array of neighbors
                    neighbor_atoms.update(pair)
        
        # Find which residues these atoms belong to
        neighbor_residues = set()
        for atom_idx in neighbor_atoms:
            atom = traj.topology.atom(atom_idx)
            neighbor_residues.add(atom.residue.index)
        
        # Get ALL atoms from these residues for complete residues
        env_indices = []
        for res_idx in neighbor_residues:
            residue = traj.topology.residue(res_idx)
            env_indices.extend([atom.index for atom in residue.atoms])
        
        # Add target residue atoms
        env_indices.extend(target_atoms)
        env_indices = sorted(set(env_indices))  # Remove duplicates
        
        if not env_indices:
            print(f"  ERROR: No atoms found within {cutoff} Å of histidine")
            return None, None
        
        # Create trajectory with full residues in the environment
        env_traj = traj.atom_slice(env_indices)
        print(f"  Extracted environment: {env_traj.n_residues} residues, {env_traj.n_atoms} atoms")
        
        return env_traj, histidine
        
    except Exception as e:
        import traceback
        print(f"  ERROR extracting environment: {str(e)}")
        traceback.print_exc()
        return None, None

def identify_charged_residues(pdb_file):
    charged_residues = {}
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                resname = line[17:20].strip()
                if resname in ['ASP', 'GLU', 'LYS', 'ARG']:
                    chain = line[21]
                    resid = int(''.join(filter(str.isdigit, line[22:26].strip())))  # Numeric resid only
                    charged_residues[(chain, resid)] = {
                        'type': resname,
                        'charge': -1 if resname in ['ASP', 'GLU'] else 1
                    }
    return charged_residues

def process_with_obabel(input_pdb, output_pdb):
    """
    Process PDB file with Open Babel to:
    1. Remove all hydrogens
    2. Add them back consistently based on valence
    
    This ensures all residues, including truncated ones, have proper hydrogens.
    """
    print("  Processing with Open Babel to fix hydrogen atoms...")
    
    # First remove all hydrogens
    no_h_pdb = os.path.join(os.path.dirname(output_pdb), "noh_temp.pdb")
    cmd1 = ["obabel", input_pdb, "-O", no_h_pdb, "-d"]
    try:
        r1 = subprocess.run(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        r1 = None
    if not os.path.exists(no_h_pdb):
        print("  Warning: Open Babel hydrogen removal failed; proceeding with original file.")
        shutil.copy(input_pdb, output_pdb)
        return os.path.exists(output_pdb)
    
    # Then add them back with proper valence
    cmd2 = ["obabel", no_h_pdb, "-O", output_pdb, "-h"]
    try:
        r2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        r2 = None
    if not os.path.exists(output_pdb):
        print("  Warning: Open Babel hydrogen addition failed; proceeding with previous file.")
        shutil.copy(no_h_pdb, output_pdb) if os.path.exists(no_h_pdb) else shutil.copy(input_pdb, output_pdb)
    
    # Clean up temporary file
    if os.path.exists(no_h_pdb):
        os.remove(no_h_pdb)
    
    return os.path.exists(output_pdb)

def fix_charged_residues(input_pdb, output_pdb, charged_residues):
    """Correct protonation states of specified charged residues"""
    print("  Fixing charged residues with precise hydrogen placement...")
    
    # Read structure
    structure = []
    max_atom_id = 0
    with open(input_pdb, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                atom_name = line[12:16].strip()
                # Extract element from columns 77-78 if available, otherwise infer from atom name
                element = line[76:78].strip() if len(line) >= 78 else atom_name[0]
                
                atom = {
                    'line': line,
                    'id': int(line[6:11]),
                    'name': atom_name,
                    'resname': line[17:20].strip(),
                    'chain': line[21],
                    'resid': int(''.join(filter(str.isdigit, line[22:26].strip()))),
                    'coords': np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                    'element': element  # Add the element information
                }
                structure.append(atom)
                max_atom_id = max(max_atom_id, atom['id'])

    # Process charged residues
    new_hydrogens = []
    atoms_to_remove = set()
    
    for (chain, resid), info in charged_residues.items():
        res_atoms = [a for a in structure if a['chain'] == chain and a['resid'] == resid]
        
        if info['type'] == 'ASP':
            # Process ASP - remove hydrogen from carboxylate oxygens (OD1/OD2)
            for oxygen_name in ['OD1', 'OD2']:
                oxygen = next((a for a in res_atoms if a['name'] == oxygen_name), None)
                if oxygen:
                    for a in res_atoms:
                        if a['element'] == 'H' and np.linalg.norm(a['coords'] - oxygen['coords']) < 1.2:
                            atoms_to_remove.add(a['id'])
            
        elif info['type'] == 'GLU':
            # Process GLU - remove hydrogen from carboxylate oxygens (OE1/OE2)
            for oxygen_name in ['OE1', 'OE2']:
                oxygen = next((a for a in res_atoms if a['name'] == oxygen_name), None)
                if oxygen:
                    for a in res_atoms:
                        if a['element'] == 'H' and np.linalg.norm(a['coords'] - oxygen['coords']) < 1.2:
                            atoms_to_remove.add(a['id'])
        
        elif info['type'] == 'LYS':
            # Process lysine NZ
            nz = next((a for a in res_atoms if a['name'] == 'NZ'), None)
            if nz:
                # Remove existing hydrogens
                for a in res_atoms:
                    if a['element'] == 'H' and np.linalg.norm(a['coords'] - nz['coords']) < 1.2:
                        atoms_to_remove.add(a['id'])
        
                # Add 3 hydrogens with improved tetrahedral geometry
                ce = next((a for a in res_atoms if a['name'] == 'CE'), None)
                if ce:
                    # Standard N-H bond length
                    bond_length = 1.01  # Angstroms
            
                    # Vector FROM NZ TO CE (the existing bond)
                    nz_to_ce = normalize(ce['coords'] - nz['coords'])
            
                    # Create an initial orthogonal vector
                    v1 = get_orthogonal_vector(nz_to_ce)
                    # Create a second orthogonal vector completing the basis
                    v2 = normalize(np.cross(nz_to_ce, v1))
            
                    # Calculate the 3 vectors for hydrogen positions
                    h_vectors = []
            
                    # The dot product of any two vectors in a regular tetrahedron is -1/3
                    # Calculate the component parallel to nz_to_ce (same for all 3 hydrogens)
                    parallel_component = -1/3  # Ensures proper tetrahedral angle (~109.5°)
            
                    # Calculate the magnitude of the perpendicular component
                    perp_magnitude = np.sqrt(1 - parallel_component**2)
            
                    # Create the 3 hydrogen vectors
                    for i in range(3):
                        theta = i * 2 * np.pi / 3  # 120° spacing around nz_to_ce
                        # Combine the parallel component with the rotated perpendicular component
                        h_dir = parallel_component * nz_to_ce + perp_magnitude * (np.cos(theta) * v1 + np.sin(theta) * v2)
                        h_vectors.append(normalize(h_dir))
            
                    # Place hydrogens at correct positions
                    for i, h_vec in enumerate(h_vectors, 1):
                        max_atom_id += 1
                        h_coords = nz['coords'] + h_vec * bond_length
                        new_hydrogens.append(
                            f"ATOM  {max_atom_id:5d} H{i:<3} LYS {chain}{resid:4d}    "
                            f"{h_coords[0]:8.3f}{h_coords[1]:8.3f}{h_coords[2]:8.3f}  1.00  0.00           H  \n"
                        )

        elif info['type'] == 'ARG':
            # Process arginine
            for nh, prefix in [('H1', 'H1'), ('H2', 'H2')]:
                nitrogen = next((a for a in res_atoms if a['name'] == nh), None)
                if nitrogen:
                    # Remove existing hydrogens
                    for a in res_atoms:
                        if a['element'] == 'H' and np.linalg.norm(a['coords'] - nitrogen['coords']) < 1.2:
                            atoms_to_remove.add(a['id'])
                    
                    # Add 2 hydrogens
                    cz = next((a for a in res_atoms if a['name'] == 'CZ'), None)
                    if cz:
                        bond_vec = cz['coords'] - nitrogen['coords']
                        ortho = get_orthogonal_vector(bond_vec)
                        for i in [1, 2]:
                            max_atom_id += 1
                            h_coords = nitrogen['coords'] + ortho * (1.01 if i == 1 else -1.01)
                            new_hydrogens.append(
                                f"ATOM  {max_atom_id:5d} {prefix}{i:<2} ARG {chain}{resid:4d}    "
                                f"{h_coords[0]:8.3f}{h_coords[1]:8.3f}{h_coords[2]:8.3f}  1.00  0.00           H  \n"
                            )
            # Add HE hydrogen for NE with improved positioning
            ne = next((a for a in res_atoms if a['name'] == 'NE'), None)
            cd = next((a for a in res_atoms if a['name'] == 'CD'), None)
            cz = next((a for a in res_atoms if a['name'] == 'CZ'), None)
    
            if ne and cd and cz:
                # Calculate vectors FROM the NE atom TO connected atoms
                ne_to_cd = normalize(cd['coords'] - ne['coords'])
                ne_to_cz = normalize(cz['coords'] - ne['coords'])
        
                # Calculate the plane normal to ensure we stay in the guanidinium plane
                normal = np.cross(ne_to_cd, ne_to_cz)
                normal = normalize(normal)
        
                # The hydrogen should be in the opposite direction of the sum of these vectors
                # This places it at approximately 120° from both existing bonds
                h_dir = -(ne_to_cd + ne_to_cz)
                h_dir = normalize(h_dir)
        
                # Standard N-H bond length for sp² hybridized nitrogen
                nh_bond_length = 1.01  # Angstroms
        
                # Position hydrogen at correct location
                h_coords = ne['coords'] + h_dir * nh_bond_length
        
                max_atom_id += 1
                new_hydrogens.append(
                    f"ATOM  {max_atom_id:5d} HE   ARG {chain}{resid:4d}    "
                    f"{h_coords[0]:8.3f}{h_coords[1]:8.3f}{h_coords[2]:8.3f}  1.00  0.00           H  \n"
                ) 

    # Write corrected structure
    with open(output_pdb, 'w') as f:
        for atom in structure:
            if atom['id'] not in atoms_to_remove:
                f.write(atom['line'])
        for h_line in new_hydrogens:
            f.write(h_line)
        f.write("END\n")
    
    print(f"  Corrected {len(charged_residues)} charged residues")
    return True

def validate_histidine_cb_hydrogens(pdb_file, output_pdb):
    """Ensure histidine CB atoms have exactly 2 hydrogen atoms"""
    print("  Validating histidine CB hydrogen count...")

    # Read structure
    atoms = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                atoms.append(line)

    # Process atoms to build connectivity
    parsed_atoms = []
    for line in atoms:
        atom_name = line[12:16].strip()
        resname = line[17:20].strip()
        chain_id = line[21]
        if resname in ["HIS", "HID", "HIE"]:
            element = line[76:78].strip() if len(line) >= 78 else atom_name[0]
            parsed_atoms.append({
                'line': line,
                'id': int(line[6:11]),
                'name': atom_name,
                'element': element,
                'coords': np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                'resid': int(''.join(filter(str.isdigit, line[22:26].strip()))),
                'chain': chain_id,
                'resname': resname
            })

    # Group atoms by residue
    residues = {}
    for atom in parsed_atoms:
        key = (atom['chain'], atom['resid'])
        if key not in residues:
            residues[key] = []
        residues[key].append(atom)

    # Check each histidine residue's CB atoms
    atoms_to_remove = set()
    for (chain, resid), residue_atoms in residues.items():
        # Find CB atom
        cb = next((a for a in residue_atoms if a['name'] == 'CB'), None)
        if not cb:
            print(f"    Warning: Could not find CB in histidine {chain}:{resid}")
            continue

        # Find hydrogens attached to CB
        cb_hydrogens = []
        for atom in residue_atoms:
            if atom['element'] == 'H' and np.linalg.norm(atom['coords'] - cb['coords']) < 1.2:
                cb_hydrogens.append(atom)

        # Check count and remove excess hydrogens
        if len(cb_hydrogens) > 2:
            print(f"    Found {len(cb_hydrogens)} hydrogens on CB of histidine {chain}:{resid}, removing {len(cb_hydrogens)-2}")
            # Sort by atom ID and keep first 2
            cb_hydrogens.sort(key=lambda a: a['id'])
            for hydrogen in cb_hydrogens[2:]:
                atoms_to_remove.add(hydrogen['id'])

    # Write corrected structure
    with open(output_pdb, 'w') as f_out:
        for line in atoms:
            atom_id = int(line[6:11])
            if atom_id not in atoms_to_remove:
                f_out.write(line)
        f_out.write("END\n")

    print(f"  Removed {len(atoms_to_remove)} excess hydrogens from histidine CB atoms")
    return True

def validate_histidine_tautomers(pdb_file, output_pdb):
    """
    Ensure histidine imidazole nitrogens have correct protonation:
    - HID should have H on ND1 only
    - HIE should have H on NE2 only
    - No nitrogen should have more than one hydrogen
    """
    print("  Validating histidine nitrogen protonation states...")
    
    # Read structure
    atoms = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                atoms.append(line)
    
    # Process atoms to build connectivity
    parsed_atoms = []
    for line in atoms:
        atom_name = line[12:16].strip()
        resname = line[17:20].strip()
        chain_id = line[21]
        if resname in ["HIS", "HID", "HIE"]:
            element = line[76:78].strip() if len(line) >= 78 else atom_name[0]
            parsed_atoms.append({
                'line': line,
                'id': int(line[6:11]),
                'name': atom_name,
                'element': element,
                'coords': np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                'resid': int(''.join(filter(str.isdigit, line[22:26].strip()))),
                'chain': chain_id,
                'resname': resname
            })
    
    # Group atoms by residue
    residues = {}
    for atom in parsed_atoms:
        key = (atom['chain'], atom['resid'])
        if key not in residues:
            residues[key] = []
        residues[key].append(atom)
    
    # Check each histidine residue
    atoms_to_remove = set()
    for (chain, resid), residue_atoms in residues.items():
        # Identify nitrogen atoms in the imidazole ring
        nd1 = next((a for a in residue_atoms if a['name'] == 'ND1'), None)
        ne2 = next((a for a in residue_atoms if a['name'] == 'NE2'), None)
        
        if not nd1 or not ne2:
            print(f"    Warning: Could not find ND1 or NE2 in histidine {chain}:{resid}")
            continue
        
        # Count and identify hydrogens connected to each nitrogen
        nd1_hydrogens = []
        ne2_hydrogens = []
        
        for atom in residue_atoms:
            if atom['element'] == 'H':
                # Check if this hydrogen is bonded to ND1
                dist_to_nd1 = np.linalg.norm(atom['coords'] - nd1['coords'])
                if dist_to_nd1 < 1.2:
                    nd1_hydrogens.append(atom)
                
                # Check if this hydrogen is bonded to NE2
                dist_to_ne2 = np.linalg.norm(atom['coords'] - ne2['coords'])
                if dist_to_ne2 < 1.2:
                    ne2_hydrogens.append(atom)
        
        resname = residue_atoms[0]['resname']
        
        # For HID: Keep one H on ND1, remove any on NE2
        if resname == "HID":
            # Remove excess hydrogens on ND1
            if len(nd1_hydrogens) > 1:
                # Keep only the closest hydrogen
                nd1_hydrogens.sort(key=lambda h: np.linalg.norm(h['coords'] - nd1['coords']))
                for h in nd1_hydrogens[1:]:
                    atoms_to_remove.add(h['id'])
                    print(f"    Removed excess ND1 hydrogen from HID {chain}:{resid}")
            
            # Remove any hydrogens on NE2
            for h in ne2_hydrogens:
                atoms_to_remove.add(h['id'])
                print(f"    Removed unexpected NE2 hydrogen from HID {chain}:{resid}")
        
        # For HIE: Keep one H on NE2, remove any on ND1
        elif resname == "HIE":
            # Remove excess hydrogens on NE2
            if len(ne2_hydrogens) > 1:
                # Keep only the closest hydrogen
                ne2_hydrogens.sort(key=lambda h: np.linalg.norm(h['coords'] - ne2['coords']))
                for h in ne2_hydrogens[1:]:
                    atoms_to_remove.add(h['id'])
                    print(f"    Removed excess NE2 hydrogen from HIE {chain}:{resid}")
            
            # Remove any hydrogens on ND1
            for h in nd1_hydrogens:
                atoms_to_remove.add(h['id'])
                print(f"    Removed unexpected ND1 hydrogen from HIE {chain}:{resid}")
        
        # For generic HIS, ensure it has at most one H on either ND1 or NE2
        else:
            # If both nitrogens have hydrogens, keep only one based on preference
            # (This is arbitrary - you could modify this based on your preference)
            if nd1_hydrogens and ne2_hydrogens:
                # Default to HID (keep ND1 hydrogen, remove NE2 hydrogen)
                for h in ne2_hydrogens:
                    atoms_to_remove.add(h['id'])
                    print(f"    Converting doubly protonated HIS {chain}:{resid} to HID")
                
                # Remove excess hydrogens on ND1
                if len(nd1_hydrogens) > 1:
                    nd1_hydrogens.sort(key=lambda h: np.linalg.norm(h['coords'] - nd1['coords']))
                    for h in nd1_hydrogens[1:]:
                        atoms_to_remove.add(h['id'])
                        print(f"    Removed excess ND1 hydrogen from HIS {chain}:{resid}")
            
            # If only one nitrogen has hydrogens but there are excess
            elif len(nd1_hydrogens) > 1:
                nd1_hydrogens.sort(key=lambda h: np.linalg.norm(h['coords'] - nd1['coords']))
                for h in nd1_hydrogens[1:]:
                    atoms_to_remove.add(h['id'])
                    print(f"    Removed excess ND1 hydrogen from HIS {chain}:{resid}")
            
            elif len(ne2_hydrogens) > 1:
                ne2_hydrogens.sort(key=lambda h: np.linalg.norm(h['coords'] - ne2['coords']))
                for h in ne2_hydrogens[1:]:
                    atoms_to_remove.add(h['id'])
                    print(f"    Removed excess NE2 hydrogen from HIS {chain}:{resid}")
    
    # Write corrected structure
    with open(output_pdb, 'w') as f_out:
        for line in atoms:
            atom_id = int(line[6:11])
            if atom_id not in atoms_to_remove:
                f_out.write(line)
        f_out.write("END\n")
    
    print(f"  Removed {len(atoms_to_remove)} excess hydrogens from histidine nitrogens")
    return True


def identify_histidine_hydrogens(pdb_file):
    """
    Identify the hydrogens on histidine imidazole nitrogens based on distance.
    Open Babel doesn't name hydrogens according to PDB standards, so we need
    to identify them by their positions relative to the nitrogen atoms.
    
    Returns:
    --------
    dict with information about ND1 and NE2 hydrogens
    """
    # Extract histidine ring atoms and their coordinates
    ring_atoms = {}  # Will hold coordinates of ND1, NE2, etc.
    hydrogens = {}   # Will hold all hydrogen atoms
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
                
            resname = line[17:20].strip()
            if resname not in ["HIS", "HID", "HIE"]:
                continue
                
            atom_name = line[12:16].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords = np.array([x, y, z])
            
            # Store ring atoms
            if atom_name in ['ND1', 'NE2', 'CE1', 'CD2', 'CG']:
                ring_atoms[atom_name] = {
                    'coords': coords,
                    'line': line
                }
            
            # Store all hydrogens regardless of name
            element = line[76:78].strip() if len(line) >= 78 else atom_name[0]
            if element == 'H' or atom_name.startswith('H'):
                hydrogens[atom_name] = {
                    'coords': coords,
                    'line': line
                }
    
    # Check if we have the required ring atoms
    required = ['ND1', 'NE2', 'CE1', 'CD2', 'CG']
    missing = [atom for atom in required if atom not in ring_atoms]
    if missing:
        raise ValueError(f"Missing histidine ring atoms: {', '.join(missing)}")
    
    # Find hydrogens closest to ND1 and NE2
    nd1_hydrogen = None
    ne2_hydrogen = None
    nd1_distance = float('inf')
    ne2_distance = float('inf')
    
    for h_name, h_data in hydrogens.items():
        h_coords = h_data['coords']
        
        # Calculate distances to both nitrogens
        dist_to_nd1 = np.linalg.norm(h_coords - ring_atoms['ND1']['coords'])
        dist_to_ne2 = np.linalg.norm(h_coords - ring_atoms['NE2']['coords'])
        
        # Check if this hydrogen is bonded to ND1
        if dist_to_nd1 < 1.2 and dist_to_nd1 < dist_to_ne2 and dist_to_nd1 < nd1_distance:
            nd1_hydrogen = h_name
            nd1_distance = dist_to_nd1
        
        # Check if this hydrogen is bonded to NE2
        if dist_to_ne2 < 1.2 and dist_to_ne2 < dist_to_nd1 and dist_to_ne2 < ne2_distance:
            ne2_hydrogen = h_name
            ne2_distance = dist_to_ne2
    
    # Determine current tautomer
    if nd1_hydrogen and not ne2_hydrogen:
        current_tautomer = "HID"
    elif ne2_hydrogen and not nd1_hydrogen:
        current_tautomer = "HIE"
    elif nd1_hydrogen and ne2_hydrogen:
        current_tautomer = "HIP"  # Doubly protonated
    else:
        current_tautomer = None  # No hydrogens found
    
    return {
        'ND1': {
            'hydrogen': nd1_hydrogen,
            'distance': nd1_distance if nd1_hydrogen else None
        },
        'NE2': {
            'hydrogen': ne2_hydrogen,
            'distance': ne2_distance if ne2_hydrogen else None
        },
        'current_tautomer': current_tautomer,
        'ring_atoms': ring_atoms,
        'hydrogens': hydrogens
    }


def create_hid_tautomer(pdb_file, output_pdb, his_data):
    """
    Create HID tautomer (hydrogen on ND1) by:
    1. Remove ANY existing hydrogens on BOTH nitrogens
    2. Add a properly positioned hydrogen to ND1
    """
    print("  Creating HID tautomer...")
    
    ring_atoms = his_data['ring_atoms']
    hydrogens = his_data['hydrogens']
    
    # Calculate proper position for HD1
    nd1_pos = ring_atoms['ND1']['coords']
    cg_pos = ring_atoms['CG']['coords'] 
    ce1_pos = ring_atoms['CE1']['coords']
    
    # Calculate vectors in the plane of the ring
    v1 = cg_pos - nd1_pos   # ND1->CG vector
    v2 = ce1_pos - nd1_pos  # ND1->CE1 vector
    
    # Average and invert these vectors for the hydrogen position
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    in_plane = -(v1_norm + v2_norm) / 2
    in_plane = in_plane / np.linalg.norm(in_plane)
    
    # Position hydrogen at N-H bond length
    nh_bond_length = 1.04  # Angstroms
    hd1_pos = nd1_pos + in_plane * nh_bond_length
    
    # Find max atom ID and collect all atoms except histidine nitrogen hydrogens
    max_atom_id = 0
    atom_lines = []
    hydrogen_on_nd1 = None
    hydrogen_on_ne2 = None
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                atom_id = int(line[6:11])
                max_atom_id = max(max_atom_id, atom_id)
                
                # Check if this is a hydrogen bound to ND1 or NE2
                atom_name = line[12:16].strip()
                element = line[76:78].strip() if len(line) >= 78 else atom_name[0]
                
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords = np.array([x, y, z])
                
                # Skip hydrogens bound to either nitrogen
                if element == 'H' or atom_name.startswith('H'):
                    dist_to_nd1 = np.linalg.norm(coords - nd1_pos)
                    dist_to_ne2 = np.linalg.norm(coords - ring_atoms['NE2']['coords'])
                    
                    if dist_to_nd1 < 1.2:
                        hydrogen_on_nd1 = line
                        continue  # Skip this hydrogen, we'll add our own
                    
                    if dist_to_ne2 < 1.2:
                        hydrogen_on_ne2 = line
                        continue  # Skip this hydrogen
                
                atom_lines.append(line)
            elif line.startswith('END'):
                break
    
    # Create new hydrogen line
    chain_id = ring_atoms['ND1']['line'][21]
    resid = ring_atoms['ND1']['line'][22:26]
    resname = "HIS"  # Keep as HIS for consistent nomenclature
    
    new_h_line = f"ATOM  {max_atom_id+1:5d} HD1  {resname} {chain_id}{resid}    {hd1_pos[0]:8.3f}{hd1_pos[1]:8.3f}{hd1_pos[2]:8.3f}  1.00  0.00           H  \n"
    
    # Write output file with new hydrogen
    with open(output_pdb, 'w') as f_out:
        for line in atom_lines:
            f_out.write(line)
        f_out.write(new_h_line)
        f_out.write("END\n")
    
    return True


def create_hie_tautomer(pdb_file, output_pdb, his_data):
    """
    Create HIE tautomer (hydrogen on NE2) by:
    1. Remove ANY existing hydrogens on BOTH nitrogens
    2. Add a properly positioned hydrogen to NE2
    """
    print("  Creating HIE tautomer...")
    
    ring_atoms = his_data['ring_atoms']
    hydrogens = his_data['hydrogens']
    
    # Calculate proper position for HE2
    ne2_pos = ring_atoms['NE2']['coords']
    ce1_pos = ring_atoms['CE1']['coords']
    cd2_pos = ring_atoms['CD2']['coords']
    
    # Calculate vectors in the plane of the ring
    v1 = ce1_pos - ne2_pos  # NE2->CE1 vector
    v2 = cd2_pos - ne2_pos  # NE2->CD2 vector
    
    # Average and invert these vectors for the hydrogen position
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    in_plane = -(v1_norm + v2_norm) / 2
    in_plane = in_plane / np.linalg.norm(in_plane)
    
    # Position hydrogen at N-H bond length
    nh_bond_length = 1.04  # Angstroms
    he2_pos = ne2_pos + in_plane * nh_bond_length
    
    # Find max atom ID and collect all atoms except histidine nitrogen hydrogens
    max_atom_id = 0
    atom_lines = []
    hydrogen_on_nd1 = None
    hydrogen_on_ne2 = None
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                atom_id = int(line[6:11])
                max_atom_id = max(max_atom_id, atom_id)
                
                # Check if this is a hydrogen bound to ND1 or NE2
                atom_name = line[12:16].strip()
                element = line[76:78].strip() if len(line) >= 78 else atom_name[0]
                
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords = np.array([x, y, z])
                
                # Skip hydrogens bound to either nitrogen
                if element == 'H' or atom_name.startswith('H'):
                    dist_to_nd1 = np.linalg.norm(coords - ring_atoms['ND1']['coords'])
                    dist_to_ne2 = np.linalg.norm(coords - ne2_pos)
                    
                    if dist_to_nd1 < 1.2:
                        hydrogen_on_nd1 = line
                        continue  # Skip this hydrogen
                    
                    if dist_to_ne2 < 1.2:
                        hydrogen_on_ne2 = line
                        continue  # Skip this hydrogen, we'll add our own
                
                atom_lines.append(line)
            elif line.startswith('END'):
                break
    
    # Create new hydrogen line
    chain_id = ring_atoms['NE2']['line'][21]
    resid = ring_atoms['NE2']['line'][22:26]
    resname = "HIS"  # Keep as HIS for consistent nomenclature
    
    new_h_line = f"ATOM  {max_atom_id+1:5d} HE2  {resname} {chain_id}{resid}    {he2_pos[0]:8.3f}{he2_pos[1]:8.3f}{he2_pos[2]:8.3f}  1.00  0.00           H  \n"
    
    # Write output file with new hydrogen
    with open(output_pdb, 'w') as f_out:
        for line in atom_lines:
            f_out.write(line)
        f_out.write(new_h_line)
        f_out.write("END\n")
    
    return True

def validate_histidine_hydrogens(pdb_file, output_pdb):
    """Remove excess hydrogens from histidine ring atoms to ensure proper valency"""
    print("  Validating histidine ring atom valency...")

    # Read structure
    atoms = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                atoms.append(line)

    # Process atoms to build connectivity
    parsed_atoms = []
    for line in atoms:
        atom_name = line[12:16].strip()
        resname = line[17:20].strip()
        if resname in ["HIS", "HID", "HIE"]:
            element = line[76:78].strip() if len(line) >= 78 else atom_name[0]
            parsed_atoms.append({
                'line': line,
                'id': int(line[6:11]),
                'name': atom_name,
                'element': element,
                'coords': np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                'resid': int(''.join(filter(str.isdigit, line[22:26].strip())))
            })

    # Group atoms by residue
    residues = {}
    for atom in parsed_atoms:
        if atom['resid'] not in residues:
            residues[atom['resid']] = []
        residues[atom['resid']].append(atom)

    # Check each histidine residue
    atoms_to_remove = set()
    for resid, residue_atoms in residues.items():
        # Identify ring atoms
        ring_atoms = {}
        for atom in residue_atoms:
            if atom['name'] in ['CG', 'ND1', 'CE1', 'NE2', 'CD2']:
                ring_atoms[atom['name']] = atom

        # Skip if not a complete histidine ring
        if len(ring_atoms) < 5:
            continue

        # Check carbon atoms (CG, CE1, CD2) for excess hydrogens
        for carbon_name in ['CG', 'CE1', 'CD2']:
            if carbon_name not in ring_atoms:
                continue

            carbon = ring_atoms[carbon_name]
            attached_h = []

            for atom in residue_atoms:
                if atom['element'] == 'H' and np.linalg.norm(atom['coords'] - carbon['coords']) < 1.2:
                    attached_h.append(atom)

            # CG should have 1 H, CE1 should have 1 H, CD2 should have 1 H
            max_h = 1

            # Remove excess hydrogens
            if len(attached_h) > max_h:
                for h in attached_h[max_h:]:
                    atoms_to_remove.add(h['id'])
                print(f"    Removed {len(attached_h) - max_h} excess hydrogens from {carbon_name}")

    # Write corrected structure
    with open(output_pdb, 'w') as f_out:
        for line in atoms:
            atom_id = int(line[6:11])
            if atom_id not in atoms_to_remove:
                f_out.write(line)
        f_out.write("END\n")

    print(f"  Removed {len(atoms_to_remove)} excess hydrogens from histidine rings")
    return True

def _format_pdb_atom_line(serial, name, resname, chain_id, resid_str, x, y, z, element):
    """Create a PDB ATOM line with standard alignment for coordinates and names."""
    # Ensure atom name alignment (right-justified in columns 13-16 unless starts with H)
    atom_name = name.ljust(4) if name.startswith('H') and len(name) < 4 else name.rjust(4)
    res_field = resname.ljust(3)
    # Element field in columns 77-78
    elem_field = element.rjust(2)
    return (
        f"ATOM  {serial:5d} {atom_name} {res_field} {chain_id}{resid_str}"
        f"   {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem_field}\n"
    )

def enforce_histidine_tautomer_hydrogens(pdb_in, pdb_out):
    """
    Enforce consistency of H on histidine nitrogens based on residue name:
    - HIE: exactly one HE2 near NE2; no H near ND1
    - HID: exactly one HD1 near ND1; no H near NE2
    If missing, add the correct H at a reasonable in-plane position using ring geometry.
    If extra/wrong, remove and/or rename appropriately.
    """
    # Load all lines
    with open(pdb_in, 'r') as f:
        lines = [ln for ln in f]

    # Collect ATOM/HETATM entries as dicts for easier processing
    atoms = []
    max_serial = 0
    for ln in lines:
        if not ln.startswith(('ATOM', 'HETATM')):
            continue
        serial = int(ln[6:11])
        max_serial = max(max_serial, serial)
        atoms.append({
            'serial': serial,
            'name': ln[12:16].strip(),
            'resname': ln[17:20].strip(),
            'chain': ln[21],
            'resid': int(ln[22:26]),
            'resid_str': ln[22:26],
            'x': float(ln[30:38]),
            'y': float(ln[38:46]),
            'z': float(ln[46:54]),
            'element': (ln[76:78].strip() if len(ln) >= 78 else ln[12:16].strip()[0]),
            'raw': ln,
            'keep': True,
        })

    # Group by (chain, resid)
    from collections import defaultdict
    by_res = defaultdict(list)
    for a in atoms:
        by_res[(a['chain'], a['resid'])].append(a)

    import numpy as _np

    def _norm(v):
        n = _np.linalg.norm(v)
        return v / n if n > 0 else v

    def _add_h(target_res, name, pos):
        nonlocal max_serial
        max_serial += 1
        resname = target_res[0]['resname']
        chain = target_res[0]['chain']
        resid_str = target_res[0]['resid_str']
        return _format_pdb_atom_line(max_serial, name, resname, chain, resid_str, pos[0], pos[1], pos[2], 'H')

    out_lines = []
    # We'll regenerate only ATOM/HETATM lines with possible edits
    atom_iter = iter(atoms)
    for ln in lines:
        if not ln.startswith(('ATOM', 'HETATM')):
            # Non-atom lines are copied later; skip for now
            continue
        else:
            break
    # Build edited atoms per residue
    edited = {}
    for key, res_atoms in by_res.items():
        # Only adjust HIE/HID residues
        resname = res_atoms[0]['resname']
        if resname not in {'HIE', 'HID'}:
            edited[key] = [a['raw'] for a in res_atoms]
            continue

        # Find ring atoms
        ring = {a['name']: a for a in res_atoms if a['name'] in {'ND1','NE2','CE1','CD2','CG'}}
        if not all(k in ring for k in ('ND1','NE2','CE1','CD2','CG')):
            # Cannot fix without ring; keep as-is
            edited[key] = [a['raw'] for a in res_atoms]
            continue

        nd1 = ring['ND1']; ne2 = ring['NE2']; ce1 = ring['CE1']; cd2 = ring['CD2']; cg = ring['CG']
        nd1_pos = _np.array([nd1['x'], nd1['y'], nd1['z']])
        ne2_pos = _np.array([ne2['x'], ne2['y'], ne2['z']])
        ce1_pos = _np.array([ce1['x'], ce1['y'], ce1['z']])
        cd2_pos = _np.array([cd2['x'], cd2['y'], cd2['z']])
        cg_pos  = _np.array([cg['x'],  cg['y'],  cg['z']])

        # Identify existing N-attached hydrogens
        nd1_h = [a for a in res_atoms if a['element']=='H' and _np.linalg.norm(_np.array([a['x'],a['y'],a['z']]) - nd1_pos) < 1.3]
        ne2_h = [a for a in res_atoms if a['element']=='H' and _np.linalg.norm(_np.array([a['x'],a['y'],a['z']]) - ne2_pos) < 1.3]

        keep_raw = []
        # Keep all non-hydrogen atoms and non-N-attached hydrogens
        for a in res_atoms:
            if a in nd1_h or a in ne2_h:
                continue
            keep_raw.append(a['raw'])

        # Compute target position and enforce
        nh = 1.04
        if resname == 'HID':
            # Desired: one HD1 at ND1, none at NE2
            # Position in-plane opposite of CG/CE1
            v1 = cg_pos - nd1_pos
            v2 = ce1_pos - nd1_pos
            in_plane = -_norm(v1) - _norm(v2)
            in_plane = _norm(in_plane)
            pos = nd1_pos + in_plane * nh
            # Add new HD1
            keep_raw.append(_add_h(res_atoms, 'HD1', pos))
        else:  # HIE
            # Desired: one HE2 at NE2, none at ND1
            v1 = ce1_pos - ne2_pos
            v2 = cd2_pos - ne2_pos
            in_plane = -_norm(v1) - _norm(v2)
            in_plane = _norm(in_plane)
            pos = ne2_pos + in_plane * nh
            keep_raw.append(_add_h(res_atoms, 'HE2', pos))

        edited[key] = keep_raw

    # Reassemble output by walking original lines and swapping per-residue blocks
    out = []
    current_key = None
    for ln in lines:
        if ln.startswith(('ATOM','HETATM')):
            chain = ln[21]
            resid = int(ln[22:26])
            key = (chain, resid)
            if key != current_key:
                # Flush previous residue atoms if any
                if current_key is not None:
                    # They are already appended when we see the first line of residue; so nothing here
                    pass
                current_key = key
                # Instead of writing incoming atom lines, write our edited block for this residue
                block = edited.get(key)
                if block is not None:
                    out.extend(block)
                else:
                    # Not edited (e.g., HETATM without residue parse), fall back to writing original lines for this residue
                    # We'll collect on the fly until residue changes
                    out.append(ln)
            else:
                # Skip subsequent original lines for this residue because we already wrote the edited block
                continue
        else:
            out.append(ln)

    with open(pdb_out, 'w') as f:
        f.writelines(out)
    return True

def create_tautomer_variants(input_pdb, temp_dir):
    """
    Create both HID and HIE tautomer variants for energy comparison
    using Open Babel to handle truncated residues first.
    """
    variants = {}
    
    # Step 1: Identify charged residues that should be preserved
    charged_residues = identify_charged_residues(input_pdb)
    print(f"  Identified {len(charged_residues)} charged residues to preserve")
    
    # Step 2: Process with Open Babel to remove and add back hydrogens
    # This ensures proper hydrogen placement on truncated residues
    obabel_pdb = os.path.join(temp_dir, "obabel_processed.pdb")
    success = process_with_obabel(input_pdb, obabel_pdb)
    
    if not success:
        print("  ERROR: Failed to process with Open Babel")
        return {}

    # Step 3.0: Fix charged residues that OpenBabel neutralized
    processed_pdb = os.path.join(temp_dir, "processed.pdb")
    success = fix_charged_residues(obabel_pdb, processed_pdb, charged_residues)
    
    if not success:
        print("  ERROR: Failed to fix charged residues")
        return {}
    
    # Step 3.5: Validate histidine hydrogens to remove excess ones
    validated_pdb = os.path.join(temp_dir, "validated.pdb")
    success = validate_histidine_hydrogens(processed_pdb, validated_pdb)
    
    if not success:
        print("  ERROR: Failed to validate histidine hydrogens")
        return {}
    
    # Continue with validated PDB
    processed_pdb = validated_pdb
    
    # Step 3.6: Validate histidine nitrogen protonation states
    tautomer_validated_pdb = os.path.join(temp_dir, "tautomer_validated.pdb")
    success = validate_histidine_tautomers(validated_pdb, tautomer_validated_pdb)
    
    if not success:
        print("  ERROR: Failed to validate histidine tautomers")
        return {}
        
    # Continue with validated PDB
    processed_pdb = tautomer_validated_pdb

    # Step 3.7: Histidine CB validation step
    cb_fixed_pdb = os.path.join(temp_dir, "cb_fixed.pdb")
    success = validate_histidine_cb_hydrogens(processed_pdb, cb_fixed_pdb)

    if not success:
        print("  ERROR: Failed to validate histidine CB hydrogens")
        return {}

    # Update processed_pdb to use the fixed version
    processed_pdb = cb_fixed_pdb

    # Step 4: Identify current histidine hydrogens
    try:
        his_data = identify_histidine_hydrogens(processed_pdb)
        print(f"  Initial tautomer from Open Babel: {his_data['current_tautomer']}")
        print(f"  ND1 hydrogen: {his_data['ND1']['hydrogen']}")
        print(f"  NE2 hydrogen: {his_data['NE2']['hydrogen']}")
        
        # Step 5: Create HID tautomer
        hid_pdb = os.path.join(temp_dir, "HID.pdb")
        create_hid_tautomer(processed_pdb, hid_pdb, his_data)
        variants["HID"] = hid_pdb
        
        # Step 6: Create HIE tautomer
        hie_pdb = os.path.join(temp_dir, "HIE.pdb")
        create_hie_tautomer(processed_pdb, hie_pdb, his_data)
        variants["HIE"] = hie_pdb
        
    except ValueError as e:
        print(f"  ERROR: {str(e)}")
        print(f"  Could not create tautomer variants. Check if histidine has all required atoms.")
        return {}
    # Final validation of created tautomers
    for tautomer, pdb_file in variants.items():
        tautomer_final = os.path.join(temp_dir, f"{tautomer}_final.pdb")
        validate_histidine_tautomers(pdb_file, tautomer_final)
        variants[tautomer] = tautomer_final
        
    return variants


def verify_tautomers(variants):
    """Verify that HID and HIE tautomers have correct hydrogen placement"""
    results = {}
    
    for tautomer, pdb_file in variants.items():
        # Identify hydrogens in the tautomer
        try:
            his_data = identify_histidine_hydrogens(pdb_file)
            current = his_data['current_tautomer']
            
            if tautomer == "HID":
                status = "CORRECT" if current == "HID" else "INCORRECT"
            elif tautomer == "HIE":
                status = "CORRECT" if current == "HIE" else "INCORRECT"
            else:
                status = "UNKNOWN"
            
            results[tautomer] = {
                "status": status,
                "ND1_hydrogen": his_data['ND1']['hydrogen'],
                "NE2_hydrogen": his_data['NE2']['hydrogen'],
                "detected_tautomer": current
            }
            
            print(f"  {tautomer} tautomer check: {status}")
            print(f"    ND1 hydrogen: {his_data['ND1']['hydrogen']}")
            print(f"    NE2 hydrogen: {his_data['NE2']['hydrogen']}")
            
        except ValueError as e:
            print(f"  ERROR verifying {tautomer}: {str(e)}")
            results[tautomer] = {
                "status": "ERROR",
                "message": str(e)
            }
    
    return results


def extract_histidine_hydrogen_lines(pdb_file, chain, resid):
    """Extract all hydrogen atom lines for a specific histidine residue"""
    hydrogen_lines = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            if line[21] != chain:
                continue
            resnum = int(line[22:26].strip())
            if resnum != resid:
                continue
            atom_name = line[12:16].strip()
            element = line[76:78].strip() if len(line) >= 78 else atom_name[0]
            if element == 'H' or atom_name.startswith('H'):
                hydrogen_lines.append(line)
    return hydrogen_lines


def calculate_formal_charge(pdb_file):
    """
    Calculate formal charge based on residue types with proper counting
    and charge capping to prevent extreme values
    """
    residue_charges = {
        'ARG': 1,   # Positively charged
        'LYS': 1,   # Positively charged
        'ASP': -1,  # Negatively charged
        'GLU': -1,  # Negatively charged
        'HIS': 0,   # Neutral histidine
        'HID': 0,   # Delta-protonated histidine (neutral)
        'HIE': 0,   # Epsilon-protonated histidine (neutral)
        'HIP': 1,   # Doubly protonated histidine (positive)
        'CYS': 0,   # Neutral cysteine
        'CYM': -1,  # Deprotonated cysteine
        'CYX': 0,   # Cysteine in disulfide bond
        'ASH': 0,   # Protonated aspartate
        'GLH': 0,   # Protonated glutamate
        'TYR': 0,   # Neutral tyrosine
        'TYD': -1,  # Deprotonated tyrosine
        'LYN': 0,   # Neutral lysine
        'SER': 0,   # Neutral serine
        'THR': 0,   # Neutral threonine
        'MET': 0,   # Neutral methionine
        'ALA': 0,   # Neutral alanine
        'VAL': 0,   # Neutral valine
        'ILE': 0,   # Neutral isoleucine
        'LEU': 0,   # Neutral leucine
        'PHE': 0,   # Neutral phenylalanine
        'TRP': 0,   # Neutral tryptophan
        'PRO': 0,   # Neutral proline
        'GLY': 0,   # Neutral glycine
        'GLN': 0,   # Neutral glutamine
        'ASN': 0,   # Neutral asparagine
        'ATP': -4,  # ATP
        'ADP': -3,  # ADP
        'NAD': -1,  # NAD+
        'NADH': -2, # NADH
        'WAT': 0,   # Water
        'HOH': 0,   # Water
        'TIP3': 0,  # Water
        'NA': 1,    # Sodium ion
        'CL': -1,   # Chloride ion
        'MG': 2,    # Magnesium ion
        'CA': 2,    # Calcium ion
        'K': 1,     # Potassium ion
        'ZN': 2     # Zinc ion
    }
    
    # Count residues and their charges
    residue_counts = {}
    total_charge = 0
    unknown_residues = set()
    
    with open(pdb_file, 'r') as f:
        current_resid = None
        current_chain = None
        
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                chain_id = line[21]
                resname = line[17:20].strip()
                resid = int(line[22:26])

                # Only count each residue once
                if (chain_id, resid) != (current_chain, current_resid):
                    current_chain = chain_id
                    current_resid = resid
                    
                    if resname in residue_charges:
                        charge = residue_charges[resname]
                        total_charge += charge
                        
                        # For debugging
                        if charge != 0:
                            key = f"{resname}({charge})"
                            residue_counts[key] = residue_counts.get(key, 0) + 1
                    else:
                        unknown_residues.add(resname)
    
    # Cap charge at reasonable values for small environments
    if total_charge > 5:
        total_charge = 5
    elif total_charge < -5:
        total_charge = -5
    
    # Print debugging information
    print(f"    Residue charge contributions: {residue_counts}")
    if unknown_residues:
        print(f"    Unknown residues (assumed neutral): {', '.join(unknown_residues)}")
    print(f"    Total charge (capped if extreme): {total_charge}")
    
    return total_charge


def create_constraints_file(pdb_file, constraints_file):
    """
    Create a constraints file for xTB that fixes only non-hydrogen atoms,
    allowing all hydrogen atoms to move freely during optimization.
    """
    with open(constraints_file, 'w') as f:
        f.write("$fix\n")
        f.write("    elements: C,N,S,O")
        f.write("\n$opt")
        f.write("\n    engine=lbfgs")
        f.write("\n$end\n")
    
    return


def run_xtb_calculation(pdb_file, temp_dir, protonation, system_charge=None, xtbopt='loose', solvent='ether', log_dir="xtb_logs", mode='opt', engine='xtb'):
    """
    Run xTB calculation with fixed heavy atoms, ether solvent, and proper logging
    mode: 'opt' for optimization (default), 'SP' for single point calculation
    """
    calc_dir = os.path.join(temp_dir, protonation)
    os.makedirs(calc_dir, exist_ok=True)
    
    # Create permanent log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Extract residue and chain info from temp_dir path
    chain_res = os.path.basename(os.path.dirname(calc_dir))
    
    # Create a unique log file name
    log_file_base = os.path.join(log_dir, f"{chain_res}_{protonation}")
    xtb_log = f"{log_file_base}.out"
    xtb_error = f"{log_file_base}.err"
    
    xyz_file = os.path.join(calc_dir, "input.xyz")
    obabel_err = ""
    try:
        result = subprocess.run(
            ["obabel", pdb_file, "-O", xyz_file], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        obabel_err = result.stderr or ""
    except FileNotFoundError:
        obabel_err = "obabel not found in PATH"

    if not os.path.exists(xyz_file):
        # Fallback: write a minimal XYZ directly from the PDB
        try:
            _pdb_to_xyz(pdb_file, xyz_file)
        except Exception as e:
            with open(xtb_error, 'w') as f:
                f.write(f"Failed to convert PDB to XYZ via obabel. {obabel_err}\n")
                f.write(f"Fallback PDB->XYZ conversion also failed: {e}\n")
            raise RuntimeError("Failed to convert PDB to XYZ:")
    
    # Determine total system charge if not specified
    if system_charge is None:
        system_charge = calculate_formal_charge(pdb_file)
    
    print(f"    Using system charge for {protonation}: {system_charge}")
    
    # Create constraints file using the simplified method with commas instead of spaces
    constraints_file = os.path.join(calc_dir, "constraints.inp")
    create_constraints_file(pdb_file, constraints_file)
    
    # Save constraints file to logs
    shutil.copy(constraints_file, f"{log_file_base}_constraints.inp")
    
    # Enforce engine/mode compatibility and build command
    if engine not in ["xtb", "g-xtb"]:
        raise ValueError(f"Unsupported engine '{engine}'. Choose 'xtb' or 'g-xtb'.")

    if engine == "g-xtb" and mode != 'SP':
        raise ValueError("g-xtb engine is only supported for single point (SP) calculations. Use --mode SP.")

    print(f"    Running {engine} ({mode} mode) with output to {xtb_log}")

    if engine == "xtb":
        # Build command for xtb
        cmd = [
            "xtb", xyz_file,
            "--alpb", solvent,
            "--chrg", str(system_charge),
            "--input", constraints_file,
            "--gfn", "2",
        ]
        # Add optimization only for 'opt' mode
        if mode == 'opt':
            cmd.extend(["--opt", xtbopt])
    else:
        # g-xtb usage: gxtb -c <xyz_file> with optional control files (e.g., .CHRG) in cwd
        # Provide charge via .CHRG in the working directory
        chrg_path = os.path.join(calc_dir, ".CHRG")
        try:
            with open(chrg_path, 'w') as cf:
                cf.write(f"{int(system_charge)}\n")
        except Exception as e:
            with open(xtb_error, 'a') as errf:
                errf.write(f"Failed to write .CHRG file: {e}\n")

        # Build command: coordinate file via -c
        cmd = ["gxtb", "-c", xyz_file]
    
    # Run xTB/g-xtb with conditional optimization
    with open(xtb_log, 'w') as log, open(xtb_error, 'w') as err:
        # Log the exact command for reproducibility
        print(f"    Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, stdout=log, stderr=err, text=True, cwd=calc_dir)
    
    # Extract energy from output (handles xtb and g-xtb variants)
    energy = _parse_energy_from_log(xtb_log)
    if energy is None and engine == "g-xtb":
        # Some g-xtb builds write to files in the working directory, not stdout
        alt_logs = [
            os.path.join(calc_dir, name) for name in (
                "gxtb.out", "g-xtb.out", "gxtb.log", "g-xtb.log", "xtb.out"
            ) if os.path.exists(os.path.join(calc_dir, name))
        ]
        for path in alt_logs:
            energy = _parse_energy_from_log(path)
            if energy is not None:
                # Copy the found log to the central logs dir for convenience
                try:
                    shutil.copy(path, f"{log_file_base}_detected.out")
                except Exception:
                    pass
                break
    
    if energy is None:
        print(f"    ERROR: Could not extract energy from xTB output. See {xtb_log} and {xtb_error} for details.")
        raise ValueError(f"Could not extract energy from xTB output.")
    
    print(f"    {protonation} energy: {energy} Hartree")
    return energy

def _parse_energy_from_log(log_path):
    energy = None
    patterns = [
        re.compile(r"TOTAL\s+ENERGY\s+(-?\d+\.\d+)", re.IGNORECASE),
        re.compile(r"total\s+E\s*=\s*(-?\d+\.\d+)", re.IGNORECASE),
        re.compile(r"Total\s+energy[:\s]+(-?\d+\.\d+)", re.IGNORECASE),
        re.compile(r"Etot\s*=\s*(-?\d+\.\d+)", re.IGNORECASE),
        re.compile(r"FINAL\s+SINGLE\s+POINT\s+ENERGY\s+(-?\d+\.\d+)", re.IGNORECASE),
        re.compile(r"E\s*\(total\)\s*=\s*(-?\d+\.\d+)", re.IGNORECASE),
    ]
    # Additional pattern for g-xtb 1.1.0 summary: 'total   -6806.20592380' at end of file
    total_line_pattern = re.compile(r"^\s*total\s+(-?\d+\.\d+)\s*$", re.IGNORECASE)
    last_total = None
    try:
        with open(log_path, 'r') as f:
            for line in f:
                # Try immediate patterns
                for pat in patterns:
                    m = pat.search(line)
                    if m:
                        try:
                            return float(m.group(1))
                        except Exception:
                            pass
                # Track last 'total <number>' occurrence
                m2 = total_line_pattern.match(line)
                if m2:
                    try:
                        last_total = float(m2.group(1))
                    except Exception:
                        pass
    except FileNotFoundError:
        return None
    # Prefer the last seen 'total' if nothing else matched
    if last_total is not None:
        return last_total
    return None

def _infer_element(atom_name, line):
    # Prefer PDB element column (77-78)
    if len(line) >= 78:
        elem = line[76:78].strip()
        if elem:
            return elem
    # Fallback: derive from atom name
    an = atom_name.strip()
    # Common case: first character is the element (e.g., C, N, O, S, H)
    if not an:
        return 'X'
    e = an[0].upper()
    # Handle two-letter halogens/metals if present at start (rare in protein)
    if len(an) >= 2 and an[1].islower():
        cand = (an[0] + an[1]).capitalize()
        if cand in {"Cl", "Br", "Na", "Mg", "Al", "Si", "Ca", "Fe", "Zn", "Cu", "Mn", "Co", "Ni"}:
            return cand
    return e

def _pdb_to_xyz(pdb_file, xyz_file):
    atoms = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            atom_name = line[12:16]
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            elem = _infer_element(atom_name, line)
            atoms.append((elem, x, y, z))
    if not atoms:
        raise ValueError("No atoms parsed from PDB for XYZ conversion")
    os.makedirs(os.path.dirname(xyz_file), exist_ok=True)
    with open(xyz_file, 'w') as xf:
        xf.write(f"{len(atoms)}\n")
        xf.write("generated from PDB\n")
        for elem, x, y, z in atoms:
            xf.write(f"{elem} {x:.6f} {y:.6f} {z:.6f}\n")

def create_no_hydrogens_pdb(input_pdb, output_pdb):
    """Create a PDB file without hydrogens but with correct histidine names"""
    print(f"Creating hydrogen-free version: {output_pdb}")
    
    with open(input_pdb, 'r') as f_in, open(output_pdb, 'w') as f_out:
        for line in f_in:
            if line.startswith(('ATOM', 'HETATM')):
                # Check if it's a hydrogen using both element and atom name
                element = line[76:78].strip()
                atom_name = line[12:16].strip()
                if element == 'H' or atom_name.startswith('H'):
                    continue
            f_out.write(line)

def qm_flipping_pipeline(input_pdb, output_pdb, args):
    """Main pipeline function with expanded PropKa3 integration for all titratable residues"""
    # Create log directory
    log_dir = "xtb_logs"
    os.makedirs(log_dir, exist_ok=True)
    print(f"xTB logs will be saved to: {os.path.abspath(log_dir)}")
        
    # Run PropKa3 predictions for all titratable residues
    pka_values = run_propka_predictions(input_pdb)
    print(f"PropKa3 found {len(pka_values)} titratable residues")
    print(f"  - ASP/GLU: {sum(1 for (_, info) in pka_values.items() if info['residue'] in ['ASP', 'GLU'])}")
    print(f"  - HIS: {sum(1 for (_, info) in pka_values.items() if info['residue'] == 'HIS')}")
    print(f"  - LYS/ARG: {sum(1 for (_, info) in pka_values.items() if info['residue'] in ['LYS', 'ARG'])}")
    print(f"  - Other: {sum(1 for (_, info) in pka_values.items() if info['residue'] not in ['ASP', 'GLU', 'HIS', 'LYS', 'ARG'])}")
    
    # Process protonation states
    protonation_map = {}
    hydrogen_map = {}
    histidine_results = []
    
    for (chain, resid), info in pka_values.items():
        residue = info['residue']
        pka = info['pKa']
        
        # Determine appropriate protonation state based on pKa
        new_resname = determine_protonation_state(residue, pka, args.ph)
        
        if residue == "HIS":
            # For histidines, we need to handle them specially with QM
            # Store for later processing
            histidine_results.append({
                "chain": chain,
                "residue": resid,
                "Position": f"{chain}:{resid}",
                "Original": residue,
                "pKa": pka,
                "Status": "Pending QM"
            })
        else:
            # For other residues, directly apply the protonation state
            if new_resname != residue:
                protonation_map[(chain, resid)] = new_resname
                print(f"Setting {residue} {chain}:{resid} to {new_resname} (pKa: {pka:.2f}, pH: {args.ph})")

    # Continue with histidine processing per the original pipeline 
    with tempfile.TemporaryDirectory() as temp_dir:
        # Original processing steps for OpenBabel and charged residues
        obabel_pdb = os.path.join(temp_dir, "obabel_processed.pdb")
        process_with_obabel(input_pdb, obabel_pdb)
        
        charged_residues = identify_charged_residues(input_pdb)
        corrected_pdb = os.path.join(temp_dir, "corrected.pdb")
        fix_charged_residues(obabel_pdb, corrected_pdb, charged_residues)
        
        # Find histidines and process them
        histidines = find_histidines(input_pdb)
        print(f"Found {len(histidines)} histidine residues")
        
        results = []
        skipped_histidines = []
        
        for his in histidines:
            chain = his["chain"]
            resid = his["resid"]
            resname = his["resname"]
            
            # Find corresponding entry in histidine_results
            his_result = next((r for r in histidine_results if r["chain"] == chain and r["residue"] == resid), None)
            
            # Initialize result entry
            result_entry = {
                "chain": chain,
                "residue": resid,
                "Position": f"{chain}:{resid}",
                "Original": resname,
                "Optimal": resname,
                "pKa": his_result["pKa"] if his_result else "N/A",
                "HID Energy (H)": "N/A",
                "HIE Energy (H)": "N/A",
                "ΔE (kcal/mol)": "N/A",
                "Note": "No calculation",
                "Status": "Unchanged"
            }
            
            ph = args.ph
            try:
                # Check PropKa predictions
                if his_result:
                    his_pka = his_result["pKa"]
                    
                    # Automatic protonation state assignment
                    if his_pka > ph + 1.0:
                        result_entry.update({
                            "Optimal": "HIP",
                            "Note": f"Auto-assigned HIP ({his_pka:.1f} > {ph}+1)",
                            "Status": "PropKa assigned"
                        })
                        protonation_map[(chain, resid)] = "HIP"  # Add to global map
                        results.append(result_entry)
                        continue
                    # Continue with the original flow for other cases

                # Original QM processing code continues here
                env_traj, target_res = extract_environment(corrected_pdb, his, args.cutoff)
                env_pdb = os.path.join(temp_dir, f"{chain}_{resid}_env.pdb")
                env_traj.save(env_pdb)

                # Create and verify tautomer variants as in original code
                variants = create_tautomer_variants(env_pdb, temp_dir)
                if not variants:
                    raise ValueError("Failed to create tautomer variants")
                
                verify_results = verify_tautomers(variants)
                
                # Run xTB calculations
                energies = {}
                for tautomer, pdb_file in variants.items():
                    try:
                        energies[tautomer] = run_xtb_calculation(
                            pdb_file, temp_dir, tautomer,
                            system_charge=calculate_formal_charge(pdb_file),
                            log_dir=log_dir, xtbopt=args.xtbopt, solvent=args.solvent, mode=args.mode, engine=args.engine
                        )
                    except Exception as e:
                        print(f"  xTB error for {tautomer}: {str(e)}")
                        energies[tautomer] = None

                # Determine optimal protonation
                if energies.get("HID") and energies.get("HIE"):
                    optimal = min(energies, key=energies.get)
                    energy_diff = abs(energies["HID"] - energies["HIE"]) * 627.509
                    
                    result_entry.update({
                        "Optimal": optimal,
                        "HID Energy (H)": f"{energies['HID']:.6f}",
                        "HIE Energy (H)": f"{energies['HIE']:.6f}",
                        "ΔE (kcal/mol)": f"{energy_diff:.2f}",
                        "Note": "QM calculated",
                        "Status": "Changed" if optimal != resname else "Unchanged"
                    })
                    
                    # Add to global protonation map and store hydrogens
                    protonation_map[(chain, resid)] = optimal
                    hydrogen_lines = extract_histidine_hydrogen_lines(variants[optimal], chain, resid)
                    if hydrogen_lines:
                        hydrogen_map[(chain, resid)] = hydrogen_lines
                
                results.append(result_entry)

            except Exception as e:
                print(f"  ERROR processing {chain}:{resid}: {str(e)}")
                skipped_histidines.append((chain, resid, str(e)))
                continue

        # Generate final output
        if results:
            # Create histidine results table
            df = pd.DataFrame(results)
            output_csv = f"{os.path.splitext(output_pdb)[0]}_results.csv"
            df.to_csv(output_csv, index=False)
            
            print("\nHistidine Results:")
            print(tabulate(
                df[["Position", "Original", "Optimal", "pKa", 
                    "HID Energy (H)", "HIE Energy (H)", "ΔE (kcal/mol)", "Status", "Note"]],
                headers=["Position", "Original", "Optimal", "pKa", 
                         "HID Energy", "HIE Energy", "ΔE (kcal/mol)", "Status", "Notes"],
                tablefmt="grid",
                showindex=False
            ))
        
        # Print summary of non-histidine protonation changes
        non_his_changes = []
        for (chain, resid), new_name in protonation_map.items():
            # Skip histidines that were already reported
            if next((r for r in results if r["chain"] == chain and r["residue"] == resid), None):
                continue
                
            # Get original residue name
            orig_res = next((info['residue'] for (c, r), info in pka_values.items() 
                           if c == chain and r == resid), "Unknown")
            pka = next((info['pKa'] for (c, r), info in pka_values.items() 
                       if c == chain and r == resid), None)
            
            non_his_changes.append({
                "Position": f"{chain}:{resid}",
                "Original": orig_res,
                "New": new_name,
                "pKa": pka if pka else "N/A",
                "pH": args.ph
            })
            
        if non_his_changes:
            print("\nNon-Histidine Protonation Changes:")
            print(tabulate(non_his_changes, headers="keys", tablefmt="grid", showindex=False))
        
        # Update PDB with all protonation states and hydrogens
        update_pdb_with_protonations(input_pdb, output_pdb, protonation_map, hydrogen_map)

        # Final consistency pass: enforce nitrogen-hydrogen placement and naming
        # for HIE/HID residues based on their assigned tautomer.
        try:
            _final_out = f"{os.path.splitext(output_pdb)[0]}_final.pdb"
            enforce_histidine_tautomer_hydrogens(output_pdb, _final_out)
            # Replace output with enforced version
            shutil.move(_final_out, output_pdb)
        except Exception as e:
            print(f"Warning: final histidine enforcement failed: {e}")
        
        if skipped_histidines:
            print("\nSkipped Histidines:")
            for chain, resid, reason in skipped_histidines:
                print(f"  {chain}:{resid} - {reason}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize histidine protonation states (HID/HIE) using Open Babel for hydrogen placement and xTB/g-xtb for energy calculations"
    )
    parser.add_argument("input_pdb", help="Input PDB file")
    parser.add_argument("output_pdb", help="Output PDB with optimized protonations")
    parser.add_argument("--cutoff", type=float, default=5.0, 
                        help="Environment cutoff in Å ¦ (default: 5.0)")
    parser.add_argument("--xtbopt", type=str, default='loose',
                        help= "xtb convergence levels ¦ (default: loose)")
    parser.add_argument("--solvent", type=str, default='ether',
                        help='xtb Implicit Solvent (ALBP) ¦ (default: ether)')
    parser.add_argument("--mode", type=str, choices=['opt', 'SP'], default='opt',
                        help='Calculation mode: opt for geometry optimization (default), SP for single point')
    parser.add_argument("--engine", type=str, choices=['xtb', 'g-xtb'], default='xtb',
                        help='Quantum engine: xtb (default) or g-xtb (SP only)')
    parser.add_argument("--ph", type=float, default=7.0,
                    help="Target pH for protonation state prediction (default: 7.0)")


    args = parser.parse_args()
    # Early validation: g-xtb is SP-only
    if args.engine == 'g-xtb' and args.mode != 'SP':
        raise SystemExit("Error: g-xtb engine supports only single point mode. Use '--mode SP' or switch to '--engine xtb'.")
    qm_flipping_pipeline(args.input_pdb, args.output_pdb, args)
