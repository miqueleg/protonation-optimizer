#!/usr/bin/env python3
"""
QM-based Histidine Protonation State Optimizer (HID vs HIE only)
Uses Open Babel for initial hydrogen placement and xTB for energy calculations.
Includes special handling for truncated residues and charge correction for ASP, GLU, LYS and ARG.
"""

import os
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
    """Run PropKa3 and return dictionary of histidine pKa values with proper parsing"""
    print("Running PropKa3 for pKa predictions...")
    result = subprocess.run(
        ["propka3", pdb_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Check for errors
    if result.returncode != 0:
        print(f"PropKa3 Error: {result.stderr}")
        return {}

    pka_values = {}
    with open(pdb_file.split('.')[0]+'.pka', 'r') as propkaout:
        for line in propkaout:
            if "HIS " in line[:5]:
                splitted = line.split()
                if len(splitted) > 15:
                    chain = splitted[2]
                    resid = int(splitted[1])
                    pka = float(splitted[3])
                    pka_values[(chain, resid)] = pka
    return pka_values


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
    subprocess.run(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Then add them back with proper valence
    cmd2 = ["obabel", no_h_pdb, "-O", output_pdb, "-h"]
    subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Clean up temporary file
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
        'K': 1      # Potassium ion
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


def run_xtb_calculation(pdb_file, temp_dir, protonation, system_charge=None, xtbopt='loose', solvent='ether', log_dir="xtb_logs", mode='opt'):
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
    result = subprocess.run(
        ["obabel", pdb_file, "-O", xyz_file], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True
    )
    
    if not os.path.exists(xyz_file):
        with open(xtb_error, 'w') as f:
            f.write(f"Failed to convert PDB to XYZ: {result.stderr}")
        raise RuntimeError(f"Failed to convert PDB to XYZ: {result.stderr}")
    
    # Determine total system charge if not specified
    if system_charge is None:
        system_charge = calculate_formal_charge(pdb_file)
    
    print(f"    Using system charge for {protonation}: {system_charge}")
    
    # Create constraints file using the simplified method with commas instead of spaces
    constraints_file = os.path.join(calc_dir, "constraints.inp")
    create_constraints_file(pdb_file, constraints_file)
    
    # Save constraints file to logs
    shutil.copy(constraints_file, f"{log_file_base}_constraints.inp")
    
    # Display calculation mode
    print(f"    Running xTB ({mode} mode) with output to {xtb_log}")
    
    # Build command based on mode
    cmd = [
        "xtb", xyz_file,
        "--alpb", solvent,
        "--chrg", str(system_charge),
        "--input", constraints_file,
        "--gfn", "2",
    ]
    
    # Add optimization flag only for 'opt' mode
    if mode == 'opt':
        cmd.extend(["--opt", xtbopt])
    
    # Run xTB with conditional optimization
    with open(xtb_log, 'w') as log, open(xtb_error, 'w') as err:
        result = subprocess.run(cmd, stdout=log, stderr=err, text=True, cwd=calc_dir)
    
    # Extract energy from output
    energy = None
    with open(xtb_log, "r") as f:
        for line in f:
            if "TOTAL ENERGY" in line:
                numbers = re.findall(r"-?\d+\.\d+", line)
                if numbers:
                    energy = float(numbers[0])
                    break
    
    if energy is None:
        with open(xtb_log, "r") as f:
            for line in f:
                if "total E" in line:
                    numbers = re.findall(r"-?\d+\.\d+", line)
                    if numbers:
                        energy = float(numbers[0])
                        break
    
    if energy is None:
        print(f"    ERROR: Could not extract energy from xTB output. See {xtb_log} and {xtb_error} for details.")
        raise ValueError(f"Could not extract energy from xTB output.")
    
    print(f"    {protonation} energy: {energy} Hartree")
    return energy

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

def update_pdb_with_protonations(input_pdb, output_pdb, results):
    """Update PDB with optimal protonation states"""
    opt_protonations = {
        (res["chain"], res["residue"]): res["Optimal"]  # Now using correct keys
        for res in results
    }
    
    with open(input_pdb, 'r') as f_in, open(output_pdb, 'w') as f_out:
        for line in f_in:
            if line.startswith('ATOM') and line[17:20].strip() in ["HIS", "HID", "HIE"]:
                chain_id = line[21]
                residue_num = int(line[22:26].strip())
                key = (chain_id, residue_num)
                
                if key in opt_protonations:
                    line = line[:17] + opt_protonations[key].ljust(3) + line[20:]
            f_out.write(line)
 
def qm_flipping_pipeline(input_pdb, output_pdb, args):
    """Main pipeline function with PropKa3 integration and proper table handling"""
    # Create log directory
    log_dir = "xtb_logs"
    os.makedirs(log_dir, exist_ok=True)
    print(f"xTB logs will be saved to: {os.path.abspath(log_dir)}")
    
    # Run PropKa3 predictions
    pka_values = run_propka_predictions(input_pdb)
    print(pka_values)    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Process entire structure with OpenBabel
        obabel_pdb = os.path.join(temp_dir, "obabel_processed.pdb")
        process_with_obabel(input_pdb, obabel_pdb)
        
        # Identify and fix charged residues
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
            
            # Initialize result entry with all possible fields
            result_entry = {
                "chain": chain,
                "residue": resid,
                "Position": f"{chain}:{resid}",
                "Original": resname,
                "Optimal": resname,
                "pKa": pka_values.get((chain, resid), "N/A"),
                "HID Energy (H)": "N/A",
                "HIE Energy (H)": "N/A",
                "ΔE (kcal/mol)": "N/A",
                "Note": "No calculation",
                "Status": "Unchanged"
            }
            ph = args.ph
            try:
                # Check PropKa predictions
                his_pka = pka_values.get((chain, resid))
                if his_pka:
                    result_entry["pKa"] = his_pka
                    
                    # Automatic protonation state assignment
                    if his_pka > ph + 1.0:
                        result_entry.update({
                            "Optimal": "HIP",
                            "Note": f"Auto-assigned HIP ({his_pka:.1f} > {ph}+1)",
                            "Status": "PropKa assigned"
                        })
                        results.append(result_entry)
                        continue
                    elif his_pka < ph - 1.0:
                        result_entry["Note"] = f"Auto-deprotonated (pKa {his_pka:.1f} < pH-1)"
                    else:
                        result_entry["Note"] = f"Buffer zone (pKa {his_pka:.1f}) - QM verification needed"

                # QM processing for non-automatic cases
                env_traj, target_res = extract_environment(corrected_pdb, his, args.cutoff)
                env_pdb = os.path.join(temp_dir, f"{chain}_{resid}_env.pdb")
                env_traj.save(env_pdb)

                # Create tautomer variants
                variants = create_tautomer_variants(env_pdb, temp_dir)
                if not variants:
                    raise ValueError("Failed to create tautomer variants")

                # Verify tautomers
                verify_results = verify_tautomers(variants)
                
                # Run xTB calculations
                energies = {}
                for tautomer, pdb_file in variants.items():
                    try:
                        energies[tautomer] = run_xtb_calculation(
                            pdb_file, temp_dir, tautomer,
                            system_charge=calculate_formal_charge(pdb_file),
                            log_dir=log_dir,xtbopt=args.xtbopt, solvent=args.solvent, mode=args.mode
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
                
                results.append(result_entry)

            except Exception as e:
                print(f"  ERROR processing {chain}:{resid}: {str(e)}")
                skipped_histidines.append((chain, resid, str(e)))
                continue

        # Generate final output
        if results:
            # Create results table
            df = pd.DataFrame(results)
            output_csv = f"{os.path.splitext(output_pdb)[0]}_results.csv"
            df.to_csv(output_csv, index=False)
            
            print("\nFinal Results:")
            print(tabulate(
                df[["Position", "Original", "Optimal", "pKa", 
                    "HID Energy (H)", "HIE Energy (H)", "ΔE (kcal/mol)", "Status", "Note"]],
                headers=["Position", "Original", "Optimal", "pKa", 
                         "HID Energy", "HIE Energy", "ΔE (kcal/mol)", "Status", "Notes"],
                tablefmt="grid",
                showindex=False
            ))
            
            # Update PDB with all results (including PropKa assignments)
            update_pdb_with_protonations(input_pdb, output_pdb, results)
            
        if skipped_histidines:
            print("\nSkipped Histidines:")
            for chain, resid, reason in skipped_histidines:
                print(f"  {chain}:{resid} - {reason}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize histidine protonation states (HID/HIE) using Open Babel for hydrogen placement and xTB for energy calculations"
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
                        help='xTB calculation mode: opt for geometry optimization (default), SP for single point')
    parser.add_argument("--ph", type=float, default=7.4,
                    help="Target pH for protonation state prediction (default: 7.4)")


    args = parser.parse_args()
    qm_flipping_pipeline(args.input_pdb, args.output_pdb, args)


