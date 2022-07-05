import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
from hashlib import sha256
import pickle
import json 

PIX_PAD = 4
CMD_PAD = 3
COORD_PAD = 4
EXT_PAD = 1
EXTRA_PAD = 1
R_PAD = 2

def hash_loop_se(data):
    if len(data['se_ext']) == 0: 
        return '', '' # empty

    pixs = []
    for pix in data['se_pix']:
        pix += EXTRA_PAD  # smallest is 0 => smallest is 1
        pixs.append(pix)
    pixs.append(np.zeros(1).astype(int))  # 0 for End of SE
    pixs = np.hstack(pixs)

    exts = []
    for ext in data['se_ext']:
        ext += EXTRA_PAD  # smallest is 0 => smallest is 1
        exts.append(ext)
    exts.append(np.zeros(1).astype(int))  # 0 for End of SE
    exts = np.hstack(exts)

    final = np.concatenate((pixs, exts))
   
    # Hash the loop parameters
    loop_hash = sha256(np.ascontiguousarray(final)).hexdigest()
    uid = data['uid']
    return loop_hash, uid


def hash_loop_s(data):
    if len(data['se_ext']) == 0: 
        return '', '' # empty

    pixs = []
    for pix in data['se_pix']:
        pix += EXTRA_PAD  # smallest is 0 => smallest is 1
        pixs.append(pix)
    pixs.append(np.zeros(1).astype(int))  # 0 for End of SE
    pixs = np.hstack(pixs)
   
    # Hash the loop parameters
    loop_hash = sha256(np.ascontiguousarray(pixs)).hexdigest()
    uid = data['uid']
    return loop_hash, uid


def hash_loop_e(data):
    if len(data['se_ext']) == 0: 
        return '', '' # empty

    exts = []
    for ext in data['se_ext']:
        ext += EXTRA_PAD  # smallest is 0 => smallest is 1
        exts.append(ext)
    exts.append(np.zeros(1).astype(int))  # 0 for End of SE
    exts = np.hstack(exts)
   
    # Hash the loop parameters
    loop_hash = sha256(np.ascontiguousarray(exts)).hexdigest()
    uid = data['uid']
    return loop_hash, uid
    

def flatten(t):
    return [item for sublist in t for item in sublist]


def parallel_hash_loops(loops, hash_type):
    """ Parallel hash generated data """
    duplicate_groups = {}
    if hash_type =='se':
        objs_iter = Pool(30).imap(hash_loop_se, loops)
    elif hash_type =='s':
        objs_iter = Pool(30).imap(hash_loop_s, loops)
    elif hash_type =='e':
        objs_iter = Pool(30).imap(hash_loop_e, loops)
    for h, uid in tqdm(objs_iter, total=len(loops)):
        if len(h)>0:
            if not h in duplicate_groups:
                duplicate_groups[h] = []
            duplicate_groups[h].append([uid])
    return duplicate_groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--hash_type", type=str, required=True)
    args = parser.parse_args()

    # Load loops
    print('loading data...')
    with open(os.path.join(args.datapath, 'train.pkl'), 'rb') as f:
        loops = pickle.load(f)
   
    # Assign UID
    for idx, data in enumerate(loops):
        data['uid'] = idx
 
    # Hash loops
    print('hashing...')
    gen_len = len(loops)
    gen_groups = parallel_hash_loops(loops, args.hash_type)
    
    # Uniqueness
    print('uniqueness...')
    num_files_in_groups = []
    for g in tqdm(gen_groups.values()):
        num_files_in_group = len(g)
        num_files_in_groups.append(num_files_in_group)
    unique_count = np.sum(np.array(num_files_in_groups)==1)
    unique_percent = (unique_count / gen_len) * 100.0
    print(f"\tUnique Percentage: {unique_percent:.2f}%")

    with open('../data/train_val_test_split.json') as f:  
        data_split = json.load(f)

    print('creating new data...')
    unique_uid = {}

    for g in tqdm(gen_groups.values()):
        uid = g[0][0] # only choose one 
        unique_uid[uid] = True

    trainset = []
    for loop in tqdm(loops):
        uid = loop['uid']
        if uid in unique_uid.keys():
            trainset.append(loop)
        else:
            pass

    with open(os.path.join(args.data_path,"train_unique_"+args.hash_type+".pkl"), "wb") as tf:
        pickle.dump(trainset, tf)

    print("Duplicate Stats:")
    print(f"\tUnique Percentage: {unique_percent:.2f}%")
    print(f"\tUnique Train Dataset Length: {len(trainset)}")
