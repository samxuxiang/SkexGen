import argparse
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
from hashlib import sha256
import pickle


def hash_loop_fake(data):
    # Hash the loop parameters
    loop_hash = sha256(np.ascontiguousarray(data['tokens'])).hexdigest()
    return loop_hash

def hash_loop_real(data):
    pix_tokens = data['se_pix']
    ext_tokens = data['se_ext']
    merged = []
    for pix, ext in zip(pix_tokens, ext_tokens):
        pix += 1  # smallest is 0 => smallest is 1
        ext += 1
        merged.append(pix)
        merged.append(ext)
    merged = np.hstack(merged)
   
    # Hash the loop parameters
    loop_hash = sha256(np.ascontiguousarray(merged)).hexdigest()
    return loop_hash


def parallel_hash_loops(loops, hash_type):
    """ Parallel hash generated data """
    duplicate_groups = {}
    if hash_type == 'real':
        objs_iter = Pool(36).imap(hash_loop_real, loops)
    else:
        hash_type == 'fake'
        objs_iter = Pool(36).imap(hash_loop_fake, loops)

    for h in tqdm(objs_iter, total=len(loops)):
        if len(h)>0:
            if not h in duplicate_groups:
                duplicate_groups[h] = []
            duplicate_groups[h].append([1])
    return duplicate_groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_path", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    args = parser.parse_args()

    print('loading data...')
    with open(args.gen_path, 'rb') as f:
        fake_sketches = pickle.load(f)
    gen_len = len(fake_sketches)
    gen_groups = parallel_hash_loops(fake_sketches, 'fake')

    # Uniqueness
    num_files_in_groups = []
    for g in gen_groups.values():
        num_files_in_group = len(g)
        num_files_in_groups.append(num_files_in_group)
    unique_count = np.sum(np.array(num_files_in_groups)==1)
    unique_percent = (unique_count / gen_len) * 100.0

    with open(args.gt_path, 'rb') as f:
        gt_sketches = pickle.load(f)
    gt_groups = parallel_hash_loops(gt_sketches, 'real')
    
    # Novelness
    novel = 0
    for key, value in gen_groups.items():
        # generated sketch appear in training
        if key not in gt_groups:
            novel += len(value)
    novel_percent = (novel / gen_len) * 100.0

    print(f"Exp {args.gen_path}")
    print(f"\tUnique Percentage: {unique_percent:.3f}%")
    print(f"\tNovel Percentage: {novel_percent:.3f}%")
