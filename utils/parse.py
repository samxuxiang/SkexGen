import os
import argparse
from dataset import SE
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input folder of the CAD obj (after normalization)")
    parser.add_argument("--bit", type=int, required=True, help='Number of bits for quantization')
    parser.add_argument("--output", type=str, required=True, help="Output folder to save the data")
    args = parser.parse_args()

    # Create training folder
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    # Start creating dataset 
    parser = SE(start=0, end=100, datapath=args.input, bit=args.bit, threads=36) # number of threads in your pc
    train_samples, test_samples, val_samples = parser.load_all_obj()

    # Save to file 
    with open(os.path.join(args.output,"train.pkl"), "wb") as tf:
        pickle.dump(train_samples, tf)
    with open(os.path.join(args.output,"test.pkl"), "wb") as tf:
        pickle.dump(test_samples, tf)
    with open(os.path.join(args.output,"val.pkl"), "wb") as tf:
        pickle.dump(val_samples, tf)
   