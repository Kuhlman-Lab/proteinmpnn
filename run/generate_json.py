import itertools
import json
from random import Random
import re
import sys, os
import argparse
import math
import numpy as np
from typing import Optional, Sequence, Tuple, Dict
from Bio.PDB import PDBParser, PDBIO, Select, Structure


class FileArgumentParser(argparse.ArgumentParser):
    """Overwrites default ArgumentParser to better handle flag files."""

    def convert_arg_line_to_args(self, arg_line: str) -> Optional[Sequence[str]]:
        """ Read from files where each line contains a flag and its value, e.g.
        '--flag value'. Also safely ignores comments denoted with '#' and 
        empty lines.
        """
        
        # Remove any comments from the line
        arg_line = arg_line.split('#')[0]
        
        # Escape if the line is empty
        if not arg_line:
            return None

        # Separate flag and values
        split_line = arg_line.strip().split(' ')
        
        # If there is actually a value, return the flag value pair
        if len(split_line) > 1:
            return [split_line[0], ' '.join(split_line[1:])]
        # Return just flag if there is no value
        else:
            return split_line


class ProteinDesignInputFormatter(object):
    def __init__(self, pdb_dir: str, designable_res: str = '', default_design_setting: str = 'all', 
                 symmetric_res: str = '', cluster_center: str = '', cluster_radius: float = 10.0, 
                 gap: float = 1000., validation_tries: int = 0, bidirectional: bool = False, 
                 constraints: str = '', multi_state: bool = False) -> None:
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
            elif (len(pdb_list) > 1) and (not multi_state):
                raise ValueError(f'The pdb_dir {pdb_dir} contains more than 1 .pdb file. '
                                'Currently this is not supported unless in multi-state mode.')
            else:
                self.pdb_dir = pdb_dir
                self.pdb_list = pdb_list
                self.parser = PDBParser(QUIET=True)

        self.design_default = default_design_setting
        self.bidirectional = bidirectional
        self.multi_state = multi_state

        # combine all PDBs into one shared file for MSD
        if self.multi_state:
            self.chain_dict = self.combine_pdbs(gap, validation_tries)

        if designable_res:
            self.design_res = self.parse_designable_res(designable_res)
        else:
            self.design_res = []
        
        if cluster_center:
            cluster_mut = self.parse_cluster_center(cluster_center, cluster_radius)
            self.design_res += cluster_mut

        # MSD uses symmetric res args, can't use both
        if symmetric_res:
            self.symmetric_res = self.parse_symmetric_res(symmetric_res)
        elif constraints:
            self.beta_dict = {}
            self.symmetric_res = self.parse_constraints(constraints)
        else:
            self.symmetric_res = []

        # update design res based on cluster_mut and symmetry (untested)
        if cluster_center:
            self.update_design_res(cluster_mut)

        # configure bidirectional symmetry
        if self.bidirectional:
            self.apply_bidirectional()

    def _check_res_validity(self, res_item: str) -> Tuple[str, int]:
        split_item = [item for item in re.split('(\d+)', res_item) if item]
        
        if len(split_item) != 2:
            raise ValueError(f'Unable to parse residue: {res_item}.')
        if split_item[0] not in self.CHAIN_IDS:
            raise ValueError(f'Unknown chain id in residue: {res_item}')
        return (split_item[0], int(split_item[1]))

    def _check_range_validity(self, range_item: str, pdb_name: str = None) -> Sequence[Tuple[str, int]]:
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
            if self.multi_state:
                res_range.append( (self.chain_dict[pdb_name][s_chain], i) )
            else:
                res_range.append( (s_chain, i) )

        return res_range

    def _check_symmetry_validity(self, symmetric_item: str, pdb_names: Sequence[str] = None) -> Dict[str, Sequence[Tuple[str, int]]]:        

        if self.multi_state:
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

        else:
            split_item = symmetric_item.split(':')

            symmetry_dict = {}
            for subitem in split_item:
                if '-' in subitem:
                    res_range = self._check_range_validity(subitem)
                    symmetry_dict[subitem] = res_range
                else:
                    res_ch, res_idx = self._check_res_validity(subitem)
                    symmetry_dict[subitem] = [(res_ch, res_idx)]

            item_lens = [len(symmetry_dict[key]) for key in symmetry_dict]
            if math.floor(sum([l == item_lens[0] for l in item_lens])/len(item_lens)) != 1:
                raise ValueError(f'Tied residues and residue ranges must be of the same '
                                f'size for forcing symmetry: {symmetric_item}')

            return symmetry_dict

    def _get_cluster_neighbors(self, center: str, cluster_radius: float) -> Sequence[str]:
        
        # Chain and residue number for center
        center_ch, center_res = center
        
        for pdb in self.pdb_list:
            # Get pdb structure
            pdb_id = pdb[:-4]
            pdb_file = os.path.join(self.pdb_dir, pdb)
            pdb_struct = self.parser.get_structure(id=pdb_id, file=pdb_file)
            
            # Get list of chains
            chains = list(pdb_struct.get_chains())
            # Determine cluster location
            for chain in chains:
                # Find correct chain
                if chain.id == center_ch:
                    for residue in chain.get_residues():
                        # Find correct residue
                        if residue.id[1] == center_res:
                            for atom in residue.get_atoms():
                                # Find the CA atom and get its coords
                                if atom.get_name() == 'CA':
                                    center_pos = atom.get_coord()
                                    break
                            break
                    break

            # Determine neighbors using cluster radius        
            neighbors = []
            for chain in chains:
                for residue in chain.get_residues():
                    # Residue chain and number
                    res_ch, res_num = chain.id, residue.id[1]
                    
                    for atom in residue.get_atoms():
                        # Find the CA atom and compute distance from center
                        if atom.get_name() == 'CA':
                            dist2center = np.sqrt(np.sum((atom.get_coord() - center_pos) ** 2) + 1e-12)
                            # If distance is less than radius, then add to neighbor list
                            if dist2center <= cluster_radius:
                                neighbors.append( (res_ch, res_num) )
                                
        return neighbors
    
    def parse_cluster_center(self, cluster_center: str, cluster_radius: float) -> Sequence[str]:
        
        if self.multi_state:
            # NOTE: multi-state clusters aren't multi-state aware.
            cluster_per_state = [c for c in cluster_center.strip().split(';') if c]
            centers = []

            for cps in cluster_per_state:
                pdb_name, clus_info = [c for c in cps.strip().split(':') if c]
                cluster_center = [s for s in clus_info.strip().split(',') if s]
                for item in cluster_center:
                    if "-" not in item:
                        item_ch, item_idx = self._check_res_validity(item)
                        item_ch = self.chain_dict[pdb_name][item_ch]
                        centers.append( (item_ch, item_idx) )
                    else:
                        range_res = self._check_range_validity(item, pdb_name)
                        centers += range_res

            design_res = []    
            for center in centers:
                design_res += self._get_cluster_neighbors(center, cluster_radius)
            # remove duplicate residues
            return [*set(design_res)]
        else:
            cluster_center = [s for s in cluster_center.strip().split(',') if s]
            centers = []

            for item in cluster_center:
                if "-" not in item:
                    item_ch, item_idx = self._check_res_validity(item)
                    centers.append( (item_ch, item_idx) )
                else:
                    range_res = self._check_range_validity(item)
                    centers += range_res
            
            design_res = []    
            for center in centers:
                design_res += self._get_cluster_neighbors(center, cluster_radius)
                    
            return design_res

    def parse_designable_res(self, design_str: str) -> Sequence[str]:
    
        if self.multi_state:
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
        
        else:
            design_str = [s for s in design_str.strip().split(",") if s]

            design_res = []
            for item in design_str:
                if "-" not in item:
                    item_ch, item_idx = self._check_res_validity(item)
                    design_res.append( (item_ch, item_idx) )
                else:
                    range_res = self._check_range_validity(item)
                    design_res += range_res
            
            return design_res

    def parse_symmetric_res(self, symmetric_str: str) -> Sequence[str]:

        symmetric_str = [s for s in symmetric_str.strip().split(",") if s]
        
        symmetric_res = []
        for item in symmetric_str:
            if ":" not in item:
                raise ValueError(f'No colon detected in symmetric res: {item}.')

            symmetry_dict = self._check_symmetry_validity(item)
            symmetric_res.append(symmetry_dict)
        
        return symmetric_res
    
    def parse_constraints(self, symmetric_str: str) -> Sequence[str]:
        """Parsing MSD constraints by treating them as symmetry rules with some added complexity."""
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

        pdb_dicts = []
        for pdb in self.pdb_list:
            pdb_id = pdb[:-4]
            pdb_file = os.path.join(self.pdb_dir, pdb)
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

            if self.multi_state:
                dictionary = {"sequence" : chain_seqs, "designable": mutable, "symmetric": symmetric, "tied_betas": self.beta_dict, "chain_key": self.chain_dict}
            else:
                dictionary = {"sequence" : chain_seqs, "designable": mutable, "symmetric": symmetric}
            pdb_dicts.append(dictionary)

        with open(out_path, 'w') as f:
            for pdb_dict in pdb_dicts:
                f.write(json.dumps(pdb_dict, indent=2) + '\n')

    def combine_pdbs(self, gap: float = 1000., valid_tries: int = 0) -> Dict[str, Dict[str, str]]:
        """ 
        Combines list of PDBs into one shared PDB file with new chain IDs.
        Each state is separated by 1000A. Optionally tries to resolve clashes.
        """
        min_dist = 0.
        tries = 0
        # sort combos by sum of increments to reduce spread and chance of PDB overflow
        combos = sorted(itertools.product([0, 1, 2, 3, 4, 5], repeat=3), key=lambda x: (sum(x), x))
        rand = Random(0)  # make seeded random shuffler

        while min_dist < 100.:
            # use first listed PDB as starting structure
            initial_pdb = self.pdb_list[0]
            target = self.parser.get_structure('main', initial_pdb)

            init_ch = [c.id for c in target.get_chains()]
            init_dict = {}
            for a, b in zip(init_ch, init_ch):
                init_dict[a] = b

            chain_dict = {initial_pdb[:-4]: init_dict}
            no_duplicates = [c.id for c in target.get_chains()]
            io = PDBIO()
            chain_inc = 0
            for model in target:  # iterate over model in target pdb
                for inc, pdb in enumerate(self.pdb_list[1:]):
                    mobile = self.parser.get_structure('mobile', pdb)
                    mobile_dict = {}
                    for m in mobile:  # iterate over model in mobile pdb
                        chain_list = [chain for chain in m]
                        [m.detach_child(chain.id) for chain in chain_list]
                        for chain in chain_list:  # iterate over chain in mobile pdb
                            # rename chains to avoid conflicts b/w files
                            original_id = chain.id
                            tmp = original_id
                            while tmp in no_duplicates:
                                chain_inc += 1
                                tmp = self.CHAIN_IDS[chain_inc]
                            chain.id = tmp
                            mobile_dict[original_id] = chain.id
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
            # if validation is disabled, just continue outside of loop
            if valid_tries == 0:
                print('Skipping validation - Multi-state integration complete!')
                break
            # check to see if MSD combination was successful - are all states suitably far apart?
            min_dist = _check_for_clashes(target, chain_dict)
            # randomly reorder distance spread combos if min_dist check fails
            if min_dist < 100.:
                rand.shuffle(combos)
                tries += 1
                if tries >= valid_tries:  # if integration fails 5x, quit to prevent infinite loop
                    raise RuntimeError('Multi-state integration failed all attempts due to clashes between states.')
                print('Multi-state integration failed due to clashes between states - retrying...')

        # saving modified PDB for future use
        msd_dir = os.path.join(self.pdb_dir, 'msd')
        if not os.path.isdir(msd_dir):
            os.mkdir(msd_dir)
        msd_pdb = os.path.join(msd_dir, 'msd.pdb')

        if valid_tries > 0:
            _check_structure(target)
            print('Multi-state integration validated successfully! Saving intermediate at: %s' % (msd_pdb))

        io.set_structure(target)
        io.save(msd_pdb, select=NotDisordered())
        self.pdb_dir = msd_dir
        self.pdb_list = ['msd.pdb']
        return chain_dict

    def apply_bidirectional(self):
        """Check to see if other specs are compatible with bidirectional.
        If so, flip order of symmetry_res to reflect this. If not, ignore them"""
        for sr in self.symmetric_res:
            items = sr.values()
            # to work with bidirectional constraints, symmetry must be dimeric
            if len(items) == 2:
                # flip one of the position lists (doesn't matter which one) so they are designed in opposite directions
                sr[list(sr.keys())[1]] = list(items)[1][::-1]
            else:
                raise ValueError(f"Bidirectional constraints not supported for {len(items)} subunits.")
        return

    def update_design_res(self, cluster_mutations: Sequence[Tuple[str, int]]):
        """Checks to see if any mutations added by cluster selection have constraints on them - if so, add the matching residues to self.design_res"""
        for sr in self.symmetric_res:  # iterate over symmetry constraints
            items = sr.values()
            for cm in cluster_mutations:  # iterate over cluster mutation list
                for it in items:  # iterate over chains within a constraint
                    if cm in it:
                        cm_idx = it.index(cm)  # get idx of mutation for retrieving matching pairs
                        for add_item in items:
                            if cm_idx < len(add_item):  # need to account for chains of different sizes
                                mut_to_add = add_item[cm_idx]
                                if mut_to_add not in self.design_res and mut_to_add[1] == cm[1]:
                                    self.design_res.append(mut_to_add)
        
        # sort self.design_res since cluster mutations will be all jumbled up at the end
        self.design_res = sorted(sorted(self.design_res, key=lambda x: x[1]), key=lambda x: x[0])
        return


class NotDisordered(Select):
    """Checks if atom is disordered, and if so, chooses first entry"""
    def accept_atom(self, atom):
        return not atom.is_disordered() or atom.get_altloc() == "A"


def _check_structure(target: Structure) -> bool:
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


def _check_for_clashes(target: Structure, chain_dict: dict) -> float:
    """Calculates minimum distance between the selected chain and the other states in the multi-state"""
    min_dist = 1E8
    for _, values in chain_dict.items():
        # pull out the chains for each state and check them against all other chains - keep min distance
        values = values.values()
        for model in target:
            all_chains = [c for c in model.get_chains()]
            sel_chains = [c for c in all_chains if c.id in values]
            other_chains = [c for c in all_chains if c not in sel_chains]
            for sc in sel_chains:
                for ot in other_chains:
                    for sel_atom in sc.get_atoms():
                        for ot_atom in ot.get_atoms():
                            dist = sel_atom - ot_atom
                            if dist < min_dist:
                                min_dist = dist

    return min_dist


def get_arguments() -> argparse.Namespace:
    """ Uses FileArgumentParser to parse commandline options to generate a json
    file for input into ProteinMPNN."""

    # Construct the parser and its arguments.
    parser = FileArgumentParser(description='Script that takes a PDB and design'
                                'specifications to create a json file for input'
                                'to ProteinMPNN', 
                                fromfile_prefix_chars='@')

    parser.add_argument('--pdb_dir',
                        type=str,
                        help='Path to the directory containing PDB files.')
    parser.add_argument('--designable_res',
                        default='',
                        type=str,
                        help='PDB chain and residue numbers to mutate, separated by '
                        'commas and/or hyphens. E.g. A10,A12-A15.'
                        'Multi-state requires filename prefix. E.g. PDB1:A10,A12-A15')
    parser.add_argument('--default_design_setting',
                        default='all',
                        type=str,
                        help="Default setting amino acid types that residues are "
                        "allowed to mutate to. Use 'all' to allow any amino acid "
                        "to be design. Default is 'all'.")
    parser.add_argument('--symmetric_res',
                        default='',
                        type=str,
                        help="PDB chain and residue numbers to force symmetric "
                        "design, separated by colons and commas. E.g. to force "
                        "symmetry between residue A1 and A15 use 'A1:A15' and for "
                        "symmetry between residues 1-5 on chains A and B use "
                        "'A1-A5:B1-B15'. (Note that the number of residues on"
                        " each side of the colon must be the same)."
                        "Cannot be used in multi-state mode.")
    parser.add_argument('--cluster_center',
                        default='',
                        type=str,
                        help="PDB chain and residue numbers which will serve as "
                        "mutation centers. Every residue with a CA within "
                        "--cluster_radius Angstroms will be mutatable")
    parser.add_argument('--cluster_radius',
                        default=10.0,
                        type=float,
                        help="Radius from cluster mutation centers in which to "
                        "include residues for mutation. Default is 10.0 A.")
    parser.add_argument('--out_path',
                        default='proteinmpnn_res_specs.json',
                        type=str,
                        help="Path for output json file. Default is "
                        "proteinmpnn_res_specs.json.")

    # multi-state design args
    parser.add_argument('--gap', 
                        default=1000.,
                        type=float,
                        help="Gap (in Angstrom) between states in MSD intermediate structure. "
                        "Only needed if you hit the PDB overflow limit. Default is 1000.")
    parser.add_argument('--validation_tries', 
                        default=0, 
                        type=int, 
                        help="If set to 0, will not check the MSD intermediate structure "
                        "for clashes. Recommended when running MSD on a new system. Very slow!")
    parser.add_argument('--bidirectional', 
                        action='store_true', 
                        help="Turn on bidirectional coding constraints. MSD only. Default is off.")
    parser.add_argument('--constraints', 
                        type=str,
                        default='',
                        help='Semicolon-separated list of multi-state design constraints. '
                        'commas separate individual residue sets within a constraint. '
                        'E.g. PDB1:A10-A15:1,PDB2:A10-A15:0.5;PDB1:A20-A25:1,PDB3:B20-B25:-1'
                        'See examples/multi_state for details.')
    parser.add_argument('--multi_state', 
                        action='store_true', 
                        default=False,
                        help="Enable multi-state design (MSD) parsing.")

    # Parse the provided arguments
    args = parser.parse_args(sys.argv[1:])

    return args
    

if __name__=="__main__":
    args = get_arguments()

    pdif = ProteinDesignInputFormatter(args.pdb_dir, args.designable_res, args.default_design_setting, 
                                       args.symmetric_res, args.cluster_center, args.cluster_radius, 
                                       args.gap, args.validation_tries, args.bidirectional, args.constraints, 
                                       args.multi_state)
    pdif.generate_json(args.out_path)
