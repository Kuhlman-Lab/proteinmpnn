import json
import re
import sys, os
import argparse
import math
import numpy as np
from typing import Optional, Sequence, Tuple, Dict
from Bio.PDB import PDBParser, PDBIO, Select, Structure
import itertools

from generate_json import FileArgumentParser, ProteinDesignInputFormatter


class MultiStateProteinDesignInputFormatter(ProteinDesignInputFormatter):
    def __init__(self, pdb_dir: str, designable_res: str = '', default_design_setting: str = 'all', 
                 constraints: str = '', gap: float = 1000.) -> None:
        self.CHAIN_IDS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
        self.AA3 = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 
                    'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR', 'XXX']
        self.AA1 = list('ACDEFGHIKLMNPQRSTVWYX')
        self.AA3_to_AA1 = {aa3: aa1 for aa3, aa1 in zip(self.AA3, self.AA1)}
        
        if not os.path.isdir(pdb_dir):
            raise ValueError(f'The pdb_dir {pdb_dir} is does not exist.')
        else:
            pdb_list = [file for file in os.listdir(pdb_dir) if file[-3:]=='pdb']
            if len(pdb_list) < 1:
                raise ValueError(f'The pdb_dir {pdb_dir} does not contain any .pdb files.')
            else:
                self.pdb_dir = pdb_dir
                self.pdb_list = pdb_list
                self.parser = PDBParser(QUIET=True)

        self.design_default = default_design_setting
        
        # combine all PDBs into one shared object
        self.msd_pdb = ''
        self.chain_dict = self.combine_pdbs(gap)

        # update residue specifications to match new chain IDs
        if designable_res:
            self.design_res = self.parse_designable_res(designable_res)
        else:
            self.design_res = []

        # handle parsing multi-state tied residues as if they were symmetry rules 
        self.beta_dict = {}
        if constraints:
            self.symmetric_res = self.parse_constraints(constraints)
        else:
            self.symmetric_res = []


    def _check_res_validity(self, res_item: str) -> Tuple[str, int]:
        split_item = [item for item in re.split('(\d+)', res_item) if item]
        if len(split_item) != 2:
            raise ValueError(f'Unable to parse residue: {res_item}.')
        if split_item[0] not in self.CHAIN_IDS:
            raise ValueError(f'Unknown chain id in residue: {res_item}')
        return (split_item[0], int(split_item[1]))

    def _check_range_validity(self, range_item: str, pdb_name: str) -> Sequence[Tuple[str, int]]:
        split_range = range_item.split('-')
        if len(split_range) != 2:
            raise ValueError(f'Unable to parse residue range: {range_item}')

        start_item, finish_item = split_range[0], split_range[1]

        s_chain, s_idx = self._check_res_validity(start_item)
        f_chain, f_idx = self._check_res_validity(finish_item)
        if s_chain != f_chain:
            raise ValueError(f'Residue ranges cannot span multiple chains: {range_item}')
        if s_idx >= f_idx:
            raise ValueError(f'Residue range starting index must be smaller than the ending index: '
                             f'{range_item}')

        res_range = []
        for i in range(s_idx, f_idx + 1):
            res_range.append( (self.chain_dict[pdb_name][s_chain], i) )

        return res_range
        
    def _check_symmetry_validity(self, symmetric_item: str, pdb_names: Sequence[str]) -> Dict[str, Sequence[Tuple[str, int]]]:        
        split_item = symmetric_item.split(':')
        symmetry_dict = {}
        for subitem, pdb_name in zip(split_item, pdb_names):
            # replace chain label for saving to json
            adj_subitem = subitem.replace(subitem[0], self.chain_dict[pdb_name][subitem[0]])
            if '-' in subitem:
                res_range = self._check_range_validity(subitem, pdb_name)
                symmetry_dict[adj_subitem] = res_range
            else:
                res_ch, res_idx = self._check_res_validity(subitem)
                symmetry_dict[adj_subitem] = [(res_ch, res_idx)]

        item_lens = [len(symmetry_dict[key]) for key in symmetry_dict]
        if math.floor(sum([l == item_lens[0] for l in item_lens])/len(item_lens)) != 1:
            raise ValueError(f'Tied residues and residue ranges must be of the same '
                             f'size for forcing symmetry: {symmetric_item}')

        return symmetry_dict

    def combine_pdbs(self, gap):
        """ Combines list of PDBs into one shared PDB file as needed by MPNN (separated by 1000A each). """
        initial_pdb = self.pdb_list[0]
        parser = PDBParser(QUIET=True)
        target = parser.get_structure('main', initial_pdb)

        init_ch = [c.id for c in target.get_chains()]
        init_dict = {}
        for a, b in zip(init_ch, init_ch):
            init_dict[a] = b

        chain_dict = {initial_pdb[:-4]: init_dict}
        no_duplicates = [c.id for c in target.get_chains()]
        io = PDBIO()
        # sort combos by sum of increments to reduce spread and chance of PDB overflow
        combos = sorted(itertools.product([0, 1, 2, 3, 4, 5], repeat=3), key=lambda x: (sum(x), x))
        chain_inc = 0
        
        for model in target:
            for inc, pdb in enumerate(self.pdb_list[1:]):
                mobile = parser.get_structure('mobile', pdb)
                mobile_dict = {}
                for m in mobile:
                    for chain in m:
                        # rename chains to avoid conflicts b/w files
                        tmp = chain.id
                        while chain.id in no_duplicates:
                            chain.id = self.CHAIN_IDS[chain_inc]
                            chain_inc += 1
                        mobile_dict[tmp] = chain.id
                        no_duplicates.append(chain.id)
                        # add chain to target structure
                        model.add(chain)
                        # increment chain to be far away from other chains
                        inc_3d = np.array(combos[inc + 1]) * gap

                        for residue in chain:
                            for atom in residue:
                                if atom.is_disordered():
                                    try:
                                        atom.disordered_select("A")
                                    except KeyError:
                                        raise ValueError('Failed to resolve disordered residues')
                                atom.set_coord(atom.get_coord() + inc_3d)

                chain_dict[pdb[:-4]] = mobile_dict

        # saving modified PDB for future use
        msd_dir = os.path.join(self.pdb_dir, 'msd')
        if not os.path.isdir(msd_dir):
            os.mkdir(msd_dir)
        
        self.check_structure(target)
        io.set_structure(target)
        io.save(os.path.join(msd_dir, 'msd.pdb'), select=NotDisordered())
        self.msd_pdb = os.path.join(msd_dir, 'msd.pdb')
        return chain_dict
    
    def check_structure(self, target: Structure) -> bool:
        atoms = target.get_atoms()
        for a in atoms:
            coords = a.get_coord()
            for c in coords:
                if c > 9999. or c < -999.:
                    raise ValueError('MSD intermediate is too big for PDB format - try reducing --gap option or number of states used at once')
                
        chains = target.get_chains()
        chains = [c for c in chains]
        if len(chains) > 62:
            raise ValueError('MSD intermediate is too big for PDB format - try reducing number of states used at once')

    def parse_designable_res(self, design_str: str) -> Sequence[str]:
    
        # split per-state with semicolon, per-region with comma
        design_per_state = [d for d in design_str.strip().split(';') if d]
        design_res = []
        for dps in design_per_state:
            pdb_name, des_info = [p for p in dps.strip().split(':') if p]
            design_str = [s for s in des_info.strip().split(",") if s]
            for item in design_str:
                if "-" not in item:
                    item_ch, item_idx = self._check_res_validity(item)
                    # need to swap out chain name for new chain
                    item_ch = self.chain_dict[pdb_name][item_ch]
                    design_res.append( (item_ch, item_idx) )
                else:
                    range_res = self._check_range_validity(item, pdb_name)
                    design_res += range_res
        
        return design_res

    def parse_constraints(self, symmetric_str: str) -> Sequence[str]:
        """Parsing MSD constraints by treating them as symmetry rules with some added complexity."""
        # 4GYT:A44-66:1.0,4GYT_A:A44-66:-0.5;4GYT:B44-66:1.0,4GYT_B:B44-66:-0.5
        symm_per_constraint = [d for d in symmetric_str.strip().split(';') if d]
        
        symmetric_res = []
        for spc in symm_per_constraint:
            symmetric_str = [s for s in spc.strip().split(",") if s]
            # symmetric_str holds a SINGLE symm pair INCLUDING pdb/beta data
            adj_symm_item = []
            pdb_names = []
            betas = []
            for item in symmetric_str:
                pdb_name, symm_res, beta = [si for si in item.strip().split(':') if si]
                adj_symm_item.append(symm_res)
                adj_chain = self.chain_dict[pdb_name][symm_res[0]]
                self.beta_dict[adj_chain] = float(beta)
                pdb_names.append(pdb_name)
                betas.append(beta)

            item = ':'.join(adj_symm_item)
            if ":" not in item:
                raise ValueError(f'No colon detected in symmetric res: {item}.')

            symmetry_dict = self._check_symmetry_validity(item, pdb_names)
            symmetric_res.append(symmetry_dict)
        
        return symmetric_res

    def generate_json(self, out_path: str) -> None:
        """Generating MSD json by running fxn on new msd.pdb file"""
        self.pdb_list = [self.msd_pdb]

        pdb_dicts = []
        for pdb in self.pdb_list:
            pdb_id = pdb[:-4]
            pdb_file = pdb
            # pdb_file = os.path.join(self.pdb_dir, pdb)
            pdb_struct = self.parser.get_structure(id=pdb_id, file=pdb_file)

            chains = list(pdb_struct.get_chains())

            pdbids = {}
            chain_seqs = {}

            for chain in chains:
                chain_seqs[chain.id] = []
                res_index_chain = 1
                residues = list(chain.get_residues())

                for residue in residues:
                    # Get residue PDB number
                    num_id = residue.id[1]

                    # Record pdbid. E.g. A10
                    pdbid = chain.id + str(num_id)

                    # Add gapped residues to dictionary
                    if residue != residues[0]:

                        # Determine number of gapped residues
                        n_gaps = 0
                        while True:
                            prev_res = chain.id + str(num_id - n_gaps - 1)
                            if prev_res not in pdbids:
                                n_gaps += 1
                            else:
                                break

                        for i in range(n_gaps):
                            # Determine pdb id of gap residue
                            prev_res = chain.id + str(num_id - n_gaps + i)

                            # Update residue id dict with (X, residue chain, chain_index)
                            pdbids[prev_res] = (self.AA3_to_AA1['XXX'], chain.id, res_index_chain)

                            # Update chain sequence with X
                            chain_seqs[chain.id].append(self.AA3_to_AA1['XXX'])

                            # Increment
                            res_index_chain += 1

                    # Update dict with (residue name, residue chain, chain index)
                    pdbids[pdbid] = (self.AA3_to_AA1.get(residue.get_resname(), 'XXX'), chain.id, res_index_chain)

                    # Add to the chain_sequence
                    chain_seqs[chain.id].append(self.AA3_to_AA1.get(residue.get_resname(), 'XXX'))

                    # Update the chain index
                    res_index_chain += 1

                # Make the list of AA1s into a single string for the chain sequence
                chain_seqs[chain.id] = "".join([x for x in chain_seqs[chain.id] if x is not None])

            mutable = []
            for resind in self.design_res:
                res_id = resind[0] + str(resind[1])
                if pdbids[res_id][0] != 'X':
                    mutable.append({"chain": pdbids[res_id][1], "resid": pdbids[res_id][2], 
                                    "WTAA": pdbids[res_id][0], "MutTo": self.design_default})

            symmetric = []
            for symmetry in self.symmetric_res:
                values = list(symmetry.values())

                for tied_pos in zip(*values):
                    skip_tie = False
                    sym_res = []
                    for pos in tied_pos:
                        res_id = pos[0] + str(pos[1])
                        if pdbids[res_id][0] == 'X':
                            skip_tie = True
                            break
                        else:
                            sym_res.append( pdbids[res_id][1] + str(pdbids[res_id][2]) )

                    if not skip_tie:
                        symmetric.append(sym_res)

            dictionary = {"sequence" : chain_seqs, "designable": mutable, "symmetric": symmetric, "tied_betas": self.beta_dict, "chain_key": self.chain_dict}
            pdb_dicts.append(dictionary)

        with open(out_path, 'w') as f:
            for pdb_dict in pdb_dicts:
                f.write(json.dumps(pdb_dict, indent=2) + '\n')


class NotDisordered(Select):
    """Checks if atom is disordered, and if so, chooses first entry"""
    def accept_atom(self, atom):
        return not atom.is_disordered() or atom.get_altloc() == "A"


def get_arguments() -> argparse.Namespace:
    """ Uses FileArgumentParser to parse commandline options for multi-state design setup."""

    # Construct the parser and its arguments.
    parser = FileArgumentParser(description='Script that takes a PDB and design'
                                'specifications to create a json file for input'
                                'to ProteinMPNN', 
                                fromfile_prefix_chars='@')

    parser.add_argument('--pdb_dir',
                        type=str,
                        help='Path to the directory containing PDB files.')
    parser.add_argument('--constraints', 
                        type=str,
                        default='',
                        help='comma-separated list of multi-state design constraints. '
                        'E.g. PDB1:A10-15:1+PDB2:A10-15:0.5+PDB3:B5-10:-3,PDB1:A20-25:1+PDB3:B20-25:1')
    parser.add_argument('--designable_res',
                        default='',
                        type=str,
                        help='PDB chain and residue numbers to mutate, separated by '
                        'commas and/or hyphens. E.g. A10,A12-A15')
    parser.add_argument('--default_design_setting',
                        default='all',
                        type=str,
                        help="Default setting amino acid types that residues are "
                        "allowed to mutate to. Use 'all' to allow any amino acid "
                        "to be design. Default is 'all'.")
    parser.add_argument('--out_path',
                        default='proteinmpnn_res_specs.json',
                        type=str,
                        help="Path for output json file. Default is "
                        "proteinmpnn_res_specs.json.")
    parser.add_argument('--gap', 
                        default=1000.,
                        type=float,
                        help="Gap (in Angstrom) between states in MSD intermediate structure. "
                        "Only needed if you hit the PDB overflow limit. Default is 1000.")

    # Parse the provided arguments
    args = parser.parse_args(sys.argv[1:])

    return args


if __name__ == '__main__':
    args = get_arguments()
    pdif = MultiStateProteinDesignInputFormatter(args.pdb_dir, args.designable_res, args.default_design_setting, args.constraints, args.gap)
    pdif.generate_json(args.out_path)