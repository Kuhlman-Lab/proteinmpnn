## Example Cases

### 1. Monomer Design
Location: `examples/monomer/`
Task: Design a single monomeric protein domain. This is the simplest configuration. This example will also be the most detailed, as others follow a similar format.

#### 1A. Standard Monomer Design (`examples/monomer/standard/`)

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

#### 1B. Mutation Clusters (`examples/monomer/mutation_cluster/`)

This protocol designs "mutation clusters" that are specified by certain seed residue(s) and a cluster radius.
```
--cluster_center A1,A43 # accepts residues with the same format as designable_res
--cluster_radius 10.0 # cluster radius in Angstrom. all residues within this radius of one or more cluster centers will be designable.
```
- Note: This protocol is only supported for single-state, non-symmetric proteins.

#### 1C. Destabilization (`examples/monomer/destabilize`)

This protocol inverts the probabilities obtained from ProteinMPNN to purposely pick disfavored residues. To do this, simply add the flag `--destabilize` to the `proteinmpnn.flags` file. For more useful negative state design with multiple constraints, see the `Multi-state Design` section.

### 2. Complex Design
Location: `examples/complex/`
Task: Design proteins with multiple chains, with or without symmetry.

#### 2A. Heterooligomers (`examples/complex/heterooligomer/`)

Heterooligomers can be designed by simply adding multiple sets of designable residues:
`--designable_res A43-A185,B43-B185,C43-C185`. Note that the chain ID is required before each and every residue number.

#### 2B. Homooligomers (`examples/complex/homooligomer/`)

Homooligomers require us to add a new option to the `json.flags` file called `--symmetric_res`. For example:
```
--designable_res A7-A183,B7-B183
--symmetric_res A7-A183:B7-B183
```
Symmetric sets of residues with arbitrary symmetries may be specified by separating them with `:`.  You can design symmetric and non-symmetric residues within the same run if desired using `--designable_res` to specify additional residues. Symmetric residue sets must have equal lengths within each symmetry.

To specify multiple symmetries, separate them with `,`. For example:
```
--symmetric_res A1-A5:B1-B5,A5-A10:C5-C10
```
This flag specifies two different 2-fold symmetries: one between chains A/B, another between chains B/C.

### 3. Multi-state Design (MSD)
Location: `examples/complex/multi_state`
Task: Design proteins to favor and/or disfavor multiple states (structures).

Notes:
- MSD is currently not supported for symmetric assemblies, since both use the same underlying machinery.
- Each individual state should be saved as a separate PDB file prior to running MSD.
- MSD will output a FASTA file for each individual state labeled with the PDB prefixes from the input.
- Residue set args like `--designable_res` require a prefix to specify the state they belong to (e.g., `PDB1:A7-A10`). See examples for details.

#### 3A. Single constraint (`examples/multi_state/single_constraint`)

When specifying residue sets (e.g., using `--designable_res`) in MSD mode, we must specify which state they belong to:
```
# Design residues A7-A183 in 4GYT_dimer.pdb and residues A7-A183 in 4GYT_monomer.pdb
# All states must be in --pdb_dir to be recognized
--designable_res 4GYT_dimer:A7-A183,B7-B183;4GYT_monomer:A7-A183
```

We use *constraints* to tie different states together using the following syntax:
```
--constraints 4GYT_dimer:A7-A183:0.5,4GYT_dimer:B7-B183:0.5,4GYT_monomer:A7-A183:1
```
The above constraint ties together chains A/B of 4GYT_dimer and chain A of 4GYT_monomer. The resulting sequence(s) will be favored by ProteinMPNN to adopt both the dimer and monomer folds. Note that different states are separated by `,`, and individual terms for each state are separated by `:`. 

Constraints cannot be used with the `--symmetric_res` argument at this time. Like symmetry, constraints must use residue sets with equal lengths, otherwise an error will be thrown.

Another technical note: multi-state design runs will generate an intermediate output at `msd/msd.pdb` which integrates all the states into a single PDB for compatibility with ProteinMPNN. This means the `--pdb_dir` flag in `proteinmpnn.flags` should be set to `msd/` instead of the default (which is usually `./`).

#### 3B. Mixed or Negative Design (`examples/multi_state/mixed_design`)

The third term in each state expression is the *weight* assigned to the logits of each state. Lower values will assign less relative importance to a given state. To implement "negative design" (i.e., to disfavor a given state), simply use a negative value such as -1 for the intended negative state. Negative and positive weights can be used together to implement "mixed" design (favoring one state, but not another). For example:
```
--designable_res 12E8:L1-L214,H1-H221;12E8_chainL_monomer:L1-L214
--constraints 12E8:L1-L214:1,12E8_chainL_monomer:L1-L214:-2
```

#### 3C. Multiple Constraints (`examples/multi_state/multiple_constraint`)

Multiple orthogonal constraints can be added by separating them using `;`. For example:
```
--designable_res 1a0o:A1-A128,B1-B70;1a0o_chainA_monomer:A1-A128;1a0o_chainB_monomer:B1-B70
--constraints 1a0o:A1-A128:1,1a0o_chainA_monomer:A1-A128:1;1a0o:B1-B70:1,1a0o_chainB_monomer:B1-B70:-1
```
These flags specify that one constraint should be used to tie 1a0o chain A with 1a0a_chainA_monomer, while another constraint should be used to tie 1a0o chain B with 1a0o_chainB_monomer.

#### 3D. Mutation Clusters (Not implemented)

This feature is not completed and does not function properly. To avoid unexpected behavior, avoid using the `--cluster_center` argument in `generate_json_multi_state.py`. May be revisited and completed at a later date.

#### 3E. Bidirectional Coding Constraint (Experimental)

Ongoing work in the lab involves design of bidirectionally encoded proteins. This is a special case of MSD involving exactly two states with a special sort of sequence symmetry defined by codon/anticodon complementarity. For more info, see the following publication: https://doi.org/10.1074/jbc.M115.642876. To define codon/anticodon complementarity, we use the human codon table at `run/model_weights/bidir_table.pt`

To enable bidirectional coding constraints, simply add the flag `--bidirectional` to the `json.flags` file (see `examples/multi_state/bidirectional`). Sequences will be sampled that attempt to maximize favorability across both states, with the added constraint that the two residues chosen must be bidirectionally compatible. Output sequences will be 100% complementary, though the codons may not be optimal for expression.

For greater flexibility in specifying different codon constraints, an experimental MCMC-based method has been developed. To enable this mode, the `--mcmc` flag may be added to `json.flags` (see `examples/multi_state/bidirectional_MCMC`). This setting is currently not validated or recommended for general use.

### 4. Other Protocols

#### 4A. Mutation pair sweep (`examples/other/mutation_pairs`)

This protocol uses an exhaustive pairwise sweep in order to determine an optimal pair of residues to mutate based on two input residue sets. This setting is currently experimental. Exactly two chains may be specified, and one mutation will be returned from each chain. It also runs quite slowly and scales with N^2 where N is the number of designable residues.

To enable this sampling mode, add the `--pairwise` flag to `proteinmpnn.flags`.