This example uses the mutation pair setting to search an interface for the best pair of mutations.

This feature is experimental and makes several assumptions in order to work properly.

Current implementation (8/2/24):
- add the --pairwise option to the proteinmpnn.flags file to enable mutation pair search
- specify what residues you want to modify in the --designable_residues field of json.flags file

Assumptions:
- ProteinMPNN will separate these based on chain ID - it is expecting two different chains, and it will find one mutation on each chain.
- It will throw an error if you give it more/less than two chains.

Note: this runs horrendously slow (>10s per sample) for sets of more than ~10 residues. May need to revisit to speedup.
