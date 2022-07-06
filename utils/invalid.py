import torch
import os
import numpy as np
import pickle 
from utils import CADparser
from multiprocessing import Pool
import argparse
from tqdm import tqdm

EXTRA_PAD = 1
MAX_TOKEN = 5000
NUM_TRHEADS = 36

class SE(torch.utils.data.Dataset):
    """ sketch-extrude dataset """
    def __init__(self, datapath, MAX_LEN):
        with open(datapath, 'rb') as f:
            data = pickle.load(f)
        self.maxlen = MAX_LEN 
        
        # Filter out too long result
        self.data = []
        for index in range(len(data)):
            vec_data = data[index]
            pix_len = vec_data['len_pix']
            ext_len = vec_data['len_ext']
            total_len = pix_len + ext_len + vec_data['num_se']
            if total_len <= self.maxlen:
                self.data.append(vec_data)
        print(len(self.data))
       

    def __len__(self):
        return len(self.data)


    def prepare_batch(self, pixel_v):
        keys = np.ones(len(pixel_v))
        padding = np.zeros(self.maxlen-len(pixel_v)).astype(int)  
        pixel_v_flat = np.concatenate([pixel_v, padding], axis=0)
        pixel_v_mask = 1-np.concatenate([keys, padding]) == 1   
        return pixel_v_flat, pixel_v_mask


    def __getitem__(self, index):
        vec_data = self.data[index]
        pix_tokens = vec_data['se_pix']
        ext_tokens = vec_data['se_ext']
        uid = vec_data['name']

        merged = []
        for pix, ext in zip(pix_tokens, ext_tokens):
            pix += EXTRA_PAD  # smallest is 0 => smallest is 1
            ext += EXTRA_PAD
            merged.append(pix)
            merged.append(ext)
        merged.append(np.zeros(1).astype(int))  # 0 for End of SE
        merged = np.hstack(merged)
        token_flat, token_mask = self.prepare_batch(merged)
        
        return token_flat, token_mask, uid 


def raster(data):   
    pixels, uid = data
    try:
        parser = CADparser(args.bit)
        parsed_data = parser.perform(pixels)
        return True, uid, parsed_data
    except Exception as error_msg:  
        return False, uid, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--bit", type=int, required=True)
    args = parser.parse_args()

    dataset = SE(datapath=os.path.join(args.datapath,'train.pkl'), MAX_LEN=MAX_TOKEN) 
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             shuffle=False, 
                                             batch_size=2048,
                                             num_workers=5)
    
    total_fails = 0
    invalid_uids = []
    with tqdm(dataloader, unit="batch") as batch_data:
        for pixel_v_flat, pixel_v_mask, uids in batch_data:
            iter_data = zip(pixel_v_flat.detach().cpu().numpy(), uids)
            load_iter = Pool(NUM_TRHEADS).imap(raster, iter_data)  # number of threads in your pc
            for valid, data_uid, _ in load_iter:
                if not valid:
                    invalid_uids.append(data_uid)

    # Invalid files 
    with open(os.path.join(args.datapath,'train_invalid.pkl'), "wb") as tf:
        pickle.dump(invalid_uids, tf)
    print(f'Total failures: {100.0*len(invalid_uids)/len(dataset):2f}%')