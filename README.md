# Protonation State Optimizer

A quantum mechanics-based tool for determining optimal aminoacidic protonation states (focusing on Histidines) in protein structures for molecular dynamics simulations.

## Description

This tool analyzes protein structures and determines the energetically favorable protonation state for each histidine residue using the xTB semi-empirical quantum chemistry method (HID/HIE). It accurately handles the two neutral tautomeric forms of histidine by:

0. Checks pKa using the PropKa3 tool. It determines HIP protonation based on pKa
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
- g-xtb (optional, single point calculations only)
- PropKa3

## Installation

1. Install the required Python packages:

2. Install Open Babel:

3. Install xTB
3.2. Install g-xtb if interested on using g-xtb on SP calculations 

5. Install PropKa3
   
6. Clone this repository:
```
git clone https://github.com/miqueleg/protonation-optimizer
```

## Usage

Basic usage:
```
python protonation_optimizer.py input.pdb output.pdb
```
With custom environment cutoff (default is 5.0 Å):
```
python protonation_optimizer.py input.pdb output.pdb --cutoff 6.0
```
Changing Optimization level and Solvent (default loose and ether) :
```
python protonation_optimizer.py input.pdb output.pdb --xtbopt crude --solvent water
```
Using Single Point calculations instead of Optimization [Faster but less precise] (default opt) :
```
python protonation_optimizer.py input.pdb output.pdb --mode SP
```

Using g-xtb engine for single point calculations (SP only):
```
python protonation_optimizer.py input.pdb output.pdb --engine g-xtb --mode SP
```
Note: g-xtb is supported only for SP calculations. Optimization with g-xtb is not available because gradient calculations are numerical (slow) in version 1.1 of g-xtb.

This will:
0. Run PropKa3 and determine initial protonations
1. Find all histidine residues in `protein.pdb`
2. For each histidine, extract a 5.0 Å environment
3. Create both HID and HIE tautomers with proper hydrogen placement
4. Run xTB optimization or xTB/g-xtb single point calculations on each tautomer
5. Determine the optimal protonation state based on energy
6. Generate a detailed report table of the analysis

## Output Files

The tool generates several outputs:
- An optimized PDB file with the computed protonation states of the AminoAcids in your protein
- A CSV file with energy results for each histidine
- A detailed text report of the analysis
- xTB/g-xtb log files for each calculation in an `xtb_logs` directory

## Special Features

- **Charged Residue Handling**: It handles ASP, GLU, LYS, and ARG by pKa/pH comparison using PropKa3
- **Constraint-based Optimization**: Allows hydrogen atoms to relax during energy calculations while keeping heavy atoms fixed

## Limitations

- Only handles the two neutral tautomers via QM (HID and HIE), not the protonated form (HIP). The latest is only determined via pKa calculation
- Processes each histidine independently; for histidines in close proximity, results may not capture cooperative effects (yet)
- When using `--engine g-xtb`, you must also set `--mode SP`. Geometry optimizations are not supported with g-xtb in this tool.

## Citing This Software

If you use this tool in your research, please cite it as:
```
Estévez-Gay, M. (2025). Protonation State Optimizer.
GitHub repository: https://github.com/miqueleg/protonation-optimizer
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request

