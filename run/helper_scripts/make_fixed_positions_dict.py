import argparse
import json

def get_json_list(input_path):
    with open(input_path, 'r') as json_file:
        json_list = list(json_file)

    dict_list = []
    for json_str in json_list:
        dict_list.append(json.loads(json_str))

    return dict_list

def main(json_list, position_list, chain_list):
    
    fixed_list = [[int(item) for item in one.split()] for one in position_list.split(",")]
    global_designed_chain_list = [str(item) for item in chain_list.split()]    
    my_dict = {}
    for result in json_list:
        all_chain_list = [item[-1:] for item in list(result) if item[:9]=='seq_chain']
        fixed_position_dict = {}
        for i, chain in enumerate(global_designed_chain_list):
            fixed_position_dict[chain] = fixed_list[i]
        for chain in all_chain_list:
            if chain not in global_designed_chain_list:       
                fixed_position_dict[chain] = []
        my_dict[result['name']] = fixed_position_dict
    
    return my_dict

def write_json(output_path, my_dict):
    with open(output_path, 'w') as f:
        f.write(json.dumps(my_dict) + '\n')
    
    #e.g. output
    #{"5TTA": {"A": [1, 2, 3, 7, 8, 9, 22, 25, 33], "B": []}, "3LIS": {"A": [], "B": []}}

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_path", type=str, help="Path to the parsed PDBs")
    argparser.add_argument("--output_path", type=str, help="Path to the output dictionary")
    argparser.add_argument("--chain_list", type=str, default='', help="List of the chains that need to be fixed")
    argparser.add_argument("--position_list", type=str, default='', help="Position lists, e.g. 11 12 14 18, 1 2 3 4 for first chain and the second chain")

    args = argparser.parse_args()
    json_list = get_json_list(args.input_path)
    my_dict = main(json_list, args.position_list, args.chain_list)
    write_json(args.output_path, my_dict)

