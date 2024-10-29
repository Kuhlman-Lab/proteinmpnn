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

### Example Cases

ProteinMPNN accepts PDB files as input and produces FASTA files as output.

Unlike the original repo, our ProteinMPNN organizes the different input options (aka arguments) into `.flag` files:
- `json.flags` is used to specify design constraints, like fixed residues and symmetry
- `proteinmpnn.flags` is used to specify prediction flags, like which sampling temperature and model variant to use.

In general, there are two steps to running ProteinMPNN:
1. Run the `generate_json.py` script and pass it the `json.flags` file.
- This makes a new file called `proteinmpnn_res_specs.json` containing parsed design information.
2. Run the `run_protein_mpnn.py ` script and pass it `proteinmpnn.flags` and `proteinmpnn_res_specs.json` to obtain the actual ProteinMPNN prediction.

Example input and expected output files, as well as jobscripts and flag files, for many different design tasks are included in `examples/`.

We also outline each task and its key arguments below.

#### 1. Monomer Design
Location: `examples/monomer/`
Task: Design a single monomeric protein domain. This is the simplest configuration. This example will also be the most detailed, as others follow a similar format.

1A. Standard Monomer Design (`examples/monomer/standard/`)

The input for this run is `6MRR.pdb`, and the output will be sent to the `outs/` folder.

The `json.flags` file includes only two arguments:
```
--pdb_dir ./ # location of the input PDB(s)
--designable_res A1-A68 # which residues to design
```
The `--designable_res` argument should use the chain and numbering used in the input PDB. This field can include:
- Individual residues (e.g., `A1,A2,A3`)
- A range of residues (e.g., `A1-A10`)
- A combination of both (e.g., `A1,A2,A50-A60`)

The `run_protein_mpnn.sh` file shows how to run this example:
```
python ../../../run/generate_json.py @json.flags

python ../../../run/run_protein_mpnn.py @proteinmpnn.flags
```
All subsequent examples will also have a matching `run_protein_mpnn.sh` file for reference.

1B. Mutation Clusters (`examples/monomer/mutation_cluster/`)

This protocol designs "mutation clusters" that are specified by certain seed residue(s) and a cluster radius.
```
--pdb_dir ./
--cluster_center A1,A43 # accepts residues with the same format as designable_res
--cluster_radius 10.0 # cluster radius in Angstrom. all residues within this radius of one or more cluster centers will be designable.
```
- Note: This protocol is only supported for single-state, non-symmetric proteins.

1C. Destabilization (`examples/monomer/destabilize`)

This protocol inverts the probabilities obtained from ProteinMPNN to purposely pick disfavored residues. To do this, simply add the flag `--destabilize` to the `proteinmpnn.flags` file. For more useful negative state design with multiple constraints, see the `Multi-state Design` section.

#### 2. Complex Design
Location: `examples/complex/`
Task: Design proteins with multiple chains, with or without symmetry.

2A. Heterooligomers (`examples/complex/heterooligomer/`)

Heterooligomers can be designed by simply adding multiple sets of designable residues:
`--designable_res A43-A185,B43-B185,C43-C185`. Note that the chain ID is required before each and every residue number.

2B. Homooligomers (`examples/complex/homooligomer/`)

Homooligomers require us to add a new option to the `json.flags` file called `--symmetric_res`. For example:
```
--pdb_dir ./
--designable_res A7-A183,B7-B183
--symmetric_res A7-A183:B7-B183
```
Symmetric sets of residues with arbitrary symmetries are specified by separating them with `:`. Note that ALL residues included in `--symmetric_res` MUST be included in `--designable_res` to be designed, but the inverse is NOT true. In other words, you can design symmetric and non-symmetric residues within the same run if desired.

More examples:
```
--symmetric_res A1-A10:B1-B10:C1-C10 # 3-fold symmetry across chains A/B/C
--symmetric_res A1-A5:B1-B5,A5-A10:C5-C10 # two different 2-fold symmetries: one between chains A/B, another between chains B/C
```

#### 3. Multi-state Design
3A. Single constraint
3B. Multiple constraint
3C. Bidirectional constraint

#### 4. Other Protocols
    4A. Mutation pair sweep


-----------------------------------------------------------------------------------------------------

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