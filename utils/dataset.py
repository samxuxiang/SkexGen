from utils import process_obj_se
from tqdm import tqdm
from multiprocessing import Pool
import json 
from pathlib import Path
from glob import glob
import itertools
 

class SE():
    """ sketch-extrude dataset """
    def __init__(self, start, end, datapath, bit, threads=16):
        self.start = start
        self.end = end
        self.datapath = datapath
        self.threads = threads
        self.bit = bit

    def load_all_obj(self, train_val_test_split):
        print("Loading obj data...")

        with open(train_val_test_split) as f:
            data_split = json.load(f)
       
        project_folders = []
        for i in range(self.start, self.end):
            cur_dir =  Path(self.datapath) / str(i).zfill(4)
            project_folders += glob(str(cur_dir)+'/*/')

        # Parallel loader
        iter_data = zip(
            project_folders,
            itertools.repeat(self.bit),
        )
        samples = []
        load_iter = Pool(self.threads).imap(process_obj_se, iter_data)
        for data_sample in tqdm(load_iter, total=len(project_folders)):
            samples += data_sample
        
        print('Splitting data...')
        train_samples = []
        test_samples = []
        val_samples = []
        for data in tqdm(samples):
            if data['name'] in data_split['train']:
                train_samples.append(data)
            elif data['name'] in data_split['test']:
                test_samples.append(data)
            elif data['name'] in data_split['validation']:
                val_samples.append(data)
            else:
                train_samples.append(data) # put into training if no match

        print(f"Data Summary")
        print(f"\tTraining data: {len(train_samples)}")
        print(f"\tValidation data: {len(val_samples)}")
        print(f"\tTest data: {len(test_samples)}")
        return train_samples, test_samples, val_samples

