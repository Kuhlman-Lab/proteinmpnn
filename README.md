# ProteinMPNN

This repo includes the Kuhlman Lab fork of ProteinMPNN. It includes all the functionality of the original ProteinMPNN repo (linked [here](https://github.com/dauparas/ProteinMPNN)), with the following additions:
- Improved input parsing for custom design runs
- Multi-state design support
- Additional utilities to provide integration with [EvoPro](https://github.com/Kuhlman-Lab/evopro)

![ProteinMPNN](https://docs.google.com/drawings/d/e/2PACX-1vTtnMBDOq8TpHIctUfGN8Vl32x5ISNcPKlxjcQJF2q70PlaH2uFlj2Ac4s3khnZqG1YxppdMr0iTyk-/pub?w=889&h=358)
Read [ProteinMPNN paper](https://www.biorxiv.org/content/10.1101/2022.06.03.494563v1).

## Installation:

```
git clone git@github.com:Kuhlman-Lab/proteinmpnn.git
cd proteinmpnn
mamba create env -f setup/proteinmpnn.yml
```

## Usage Guidelines:

### General Usage

The different input arguments available for each script can be viewed by adding `-h` to your python call (e.g., `python generate_json.py -h`).

ProteinMPNN accepts PDB files as input and produces FASTA files as output.

Unlike the original repo, our ProteinMPNN organizes the different input options (aka arguments) into `.flag` files:
- `json.flags` is used to specify design constraints, like fixed residues and symmetry
- `proteinmpnn.flags` is used to specify prediction flags, like which sampling temperature and model variant to use.

In general, there are two steps to running ProteinMPNN:
1. Run the `generate_json.py` script and pass it the `json.flags` file.
- This makes a new file called `proteinmpnn_res_specs.json` containing parsed design information.
2. Run the `run_protein_mpnn.py ` script and pass it `proteinmpnn.flags` and `proteinmpnn_res_specs.json` to obtain the actual ProteinMPNN prediction.

### Useful Flags

Used in `json.flags`:

`--default_design_setting`: this is an optional filter to allow/disallow certain residue types during design. By default, it is set to `all`, which allows all 20 amino acids. Possible settings include:
    `all-hydphob`: exclude hydrophobic residues (`CDEHKNPQRSTX`)
    `all-hydphil`: exclude hydrophilic residues (`ACFGILMPVWYX`)
    `all-CLD`: exclude specific amino acids (in this case, Cys, Leu, and Asp)
    `L+polar`: mix-and-match amino acids and categories (in this case, allow all polar amino acids and also Leu)

Used in `proteinmpnn.flags`:
`--model_name`: specifies which ProteinMPNN model checkpoint to use. Possible options include:
    `v_48_002`: vanilla (default) model with k=48 neighbors and 0.02A noise
    `s_48_010`: soluble protein model with k=48 neighbors and 0.1A noise

`--sampling_temp`: specifies the sampling temperature, which changes how diverse the generated sequences will be. Ranges from 0 to 1, inclusive. A temperature of 0 returns the "best" prediction every time (zero diversity), while a temperature of 1 will return completely random samples. Recommended range is 0.0 - 0.3 or so.

`--dump_probs`: if included, ProteinMPNN will save the predicted sequence probability table for each scaffold. This will be a numpy array of shape [L, 21], for a protein of length L. If multiple sequences are generated per scaffold, probabilities will be averaged before saving. A helper script for visualizing these tables is included at `run/helper_scripts/other_tools/view_probs.py`.

### Example Cases

Example input and expected output files, as well as jobscripts and flag files, for many different design tasks are included in `examples/`. For a summary and explanation of each example, see `examples/EXAMPLES.md`. Currently supported protocols include:

1. Monomer Design (with user-friendly parsing of designable residues)
2. Binder Design
2. Oligomer Design (with support for abitrary symmetries in homooligomers)
3. Multi-state Design (with support for multiple complex design constraints)

-----------------------------------------------------------------------------------------------------

## Unit Testing

TODO

## Code organization:
* `run/run_protein_mpnn.py` - the main script to initialialize and run the model.
* `run/generate_json.py` - function to automatically generate json of design constraints.
* `run/helper_scripts/` - helper functions to parse PDBs, assign which chains to design, which residues to fix, adding AA bias, tying residues etc.
* `examples/` - simple example inputs/outputs and runscripts for different tasks.
* `model_weights/` - trained proteinmpnn model weights.
    * `v_48_...` - vanilla proteinmpnn models trained at different noise levels.
    * `s_48_...` - solublempnn models trained at different noise levels.
    * `ca_48_...` - Ca-only models trained at different noise levels.


## License

ProteinMPNN is distributed under an MIT license, which can be found at `proteinmpnn/LICENSE`. See license file for more details.