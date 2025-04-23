# Histidine Protonation State Optimizer

A quantum mechanics-based tool for determining optimal histidine protonation states (HID/HIE) in protein structures for molecular dynamics simulations.

## Description

This tool analyzes protein structures and determines the energetically favorable protonation state for each histidine residue using the xTB semi-empirical quantum chemistry method. It accurately handles the two neutral tautomeric forms of histidine by:

1. Extracting the local environment around each histidine residue
2. Generating both HID (delta-protonated) and HIE (epsilon-protonated) tautomers
3. Special handling of charged residues (ASP, GLU, LYS, ARG) with proper proton positioning
4. Running quantum mechanical energy calculations on each tautomer
5. Comparing energies to determine the optimal protonation state

This tool is particularly useful for preparing protein structures for molecular dynamics simulations where histidine protonation can significantly impact results.

## Dependencies

### Python Packages
- NumPy
- Pandas
- MDTraj
- BioPython (Bio.PDB)
- Tabulate

### External Software
- Open Babel (for hydrogen placement)
- xTB (for quantum mechanical calculations)

## Installation

1. Install the required Python packages:

2. Install Open Babel:

3. Install xTB:

4. Clone this repository:
```
git clone https://github.com/miqueleg/histidine-protonation-optimizer
cd histidine-protonation-optimizer
```

## Usage

Basic usage:
```
python his_protonation_optimizer.py input.pdb output.pdb
```
With custom environment cutoff (default is 5.0 Å):
```
python his_protonation_optimizer.py input.pdb output.pdb --cutoff 6.0
```
Changing Optimization level and Solvent (default loose and ether) :
```
python his_protonation_optimizer.py input.pdb output.pdb --xtbopt crude --solvent water
```


This will:
1. Find all histidine residues in `protein.pdb`
2. For each histidine, extract a 5.0 Å environment
3. Create both HID and HIE tautomers with proper hydrogen placement
4. Run xTB optimization calculations on each tautomer
5. Determine the optimal protonation state based on energy
6. Save the result to `output.pdb` and `output_noH.pdb` (`output_noH.pdb` can be used directly for Leap or PDBfixer)
7. Generate a detailed report of the analysis

## Output Files

The tool generates several outputs:
- An optimized PDB file with the correct histidine protonation states
- A CSV file with energy results for each histidine
- A detailed text report of the analysis
- xTB log files for each calculation in an `xtb_logs` directory

## Special Features

- **Charged Residue Handling**: It handles ASP, GLU, LYS, and ARG by preserving their charged states
- **Constraint-based Optimization**: Allows hydrogen atoms to relax during energy calculations while keeping heavy atoms fixed

## Limitations

- Only handles the two neutral tautomers (HID and HIE), not the protonated form (HIP)
- Processes each histidine independently; for histidines in close proximity, results may not capture cooperative effects (not yet)
- Requires proper installation and configuration of xTB and Open Babel

## Citing This Software

If you use this tool in your research, please cite it as:
```
Estévez-Gay, M. (2025). Histidine Protonation State Optimizer.
GitHub repository: https://github.com/miqueleg/histidine-protonation-optimizer
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request


