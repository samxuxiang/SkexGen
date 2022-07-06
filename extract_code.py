import os
import torch
import argparse
import numpy as np
from model.encoder import PARAMEncoder, CMDEncoder, EXTEncoder
from dataset import SketchExtData
import pickle 
from tqdm import tqdm


def extract(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device("cuda:0")

    dataset = SketchExtData(args.data, args.invalid, args.maxlen)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             shuffle=False, 
                                             batch_size=1024,
                                             num_workers=5)
    # Load pretrained models
    cmd_encoder = CMDEncoder(
        config={
            'hidden_dim': 512,
            'embed_dim': 256,
            'num_layers': 4,
            'num_heads': 8,
            'dropout_rate': 0.1
        },
        max_len=dataset.maxlen_pix,
        code_len = 4,
        num_code = 500,
    )
    cmd_encoder.load_state_dict(torch.load(os.path.join(args.sketch_weight, 'cmdenc_epoch_'+str(args.epoch)+'.pt')))
    cmd_encoder = cmd_encoder.to(device).eval()

    param_encoder = PARAMEncoder(
        config={
            'hidden_dim': 512,
            'embed_dim': 256,
            'num_layers': 4,
            'num_heads': 8,
            'dropout_rate': 0.1
        },
        quantization_bits=args.bit,
        max_len=dataset.maxlen_pix,
        code_len = 2,
        num_code = 1000,
    )
    param_encoder.load_state_dict(torch.load(os.path.join(args.sketch_weight, 'paramenc_epoch_'+str(args.epoch)+'.pt')))
    param_encoder = param_encoder.to(device).eval()

    ext_encoder = EXTEncoder(
        config={
            'hidden_dim': 512,
            'embed_dim': 256,
            'num_layers': 4,
            'num_heads': 8,
            'dropout_rate': 0.1
        },
        quantization_bits=6,
        max_len=dataset.maxlen_ext,
        code_len = 4,
        num_code = 1000,
    )
    ext_encoder.load_state_dict(torch.load(os.path.join(args.ext_weight, 'extenc_epoch_1.pt')))
    ext_encoder = ext_encoder.to(device).eval()

    print('Extracting Code...')
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    total_z = []
    with tqdm(dataloader, unit="batch") as batch_data:
        for cmd, cmd_mask, pix, xy, pix_mask, flag, ext, ext_mask in batch_data:
            with torch.no_grad():
                cmd = cmd.to(device) 
                pix = pix.to(device)
                xy = xy.to(device)
                cmd_mask = cmd_mask.to(device)
                pix_mask = pix_mask.to(device)
                flag = flag.to(device)
                ext = ext.to(device)
                ext_mask = ext_mask.to(device)

                cmd_code = cmd_encoder.get_code(cmd, cmd_mask)
                param_code = param_encoder.get_code(pix, xy, pix_mask) 
                ext_code = ext_encoder.get_code(ext, flag, ext_mask) 
           
            codes = np.concatenate((cmd_code, param_code, ext_code), 1)
            total_z.append(codes)

    code = np.unique(np.vstack(total_z), return_counts=False, axis=0)
   
    print('Saving...')
    with open(os.path.join(args.output, 'code.pkl'), "wb") as tf:
        pickle.dump(code, tf)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--sketch_weight", type=str, required=True)
    parser.add_argument("--ext_weight", type=str, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--maxlen", type=int, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--invalid", type=str, required=True)
    parser.add_argument("--bit", type=int, required=True)
    args = parser.parse_args()

    extract(args)