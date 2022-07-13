import os
import torch
import argparse
from model.encoder import EXTEncoder
from model.decoder import EXTDecoder
from dataset import ExtData
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import sys
sys.path.insert(0, 'utils')
from utils import get_constant_schedule_with_warmup


def train(args):
    # gpu device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda:0")
    
    # Initialize dataset loader
    train_dataset = ExtData(args.train_data, args.maxlen)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                             shuffle=True, 
                                             batch_size=args.batchsize,
                                             num_workers=5,
                                             pin_memory=True)

    val_dataset = ExtData(args.val_data, args.maxlen)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                             shuffle=False, 
                                             batch_size=args.batchsize,
                                             num_workers=5)
   
    # Initialize models
    ext_encoder = EXTEncoder(
        config={
            'hidden_dim': 512,
            'embed_dim': 256,
            'num_layers': 4,
            'num_heads': 8,
            'dropout_rate': 0.1
        },
        quantization_bits=args.bit,
        max_len=train_dataset.maxlen_ext,
        code_len = 4,
        num_code = 1000,
    )
    ext_encoder = ext_encoder.to(device).train()

    ext_decoder = EXTDecoder(
        config={
            'hidden_dim': 512,
            'embed_dim': 256, 
            'num_layers': 4, 
            'num_heads': 8,
            'dropout_rate': 0.1  
        },
        max_len=train_dataset.maxlen_ext,
        quantization_bits=args.bit,
    )
    ext_decoder = ext_decoder.to(device).train()
    
    # Initialize optimizer
    params = list(ext_encoder.parameters()) + list(ext_decoder.parameters()) 
    optimizer = torch.optim.Adam(params, lr=1e-3)
    scheduler = get_constant_schedule_with_warmup(optimizer, 2000)
   
    # logging 
    writer = SummaryWriter(log_dir=args.output)
    
    # Main training loop
    iters = 0
    print('Start training...')

    for epoch in range(200):  # 200 epochs is enough
        with tqdm(train_dataloader, unit="batch") as batch_data:
            for ext_seq, flag_seq, ext_mask in batch_data:
                ext_seq = ext_seq.to(device)
                flag_seq = flag_seq.to(device)
                ext_mask = ext_mask.to(device)
           
                # Pass through encoder 
                latent_z, vq_loss, selection = ext_encoder(ext_seq, flag_seq, ext_mask, epoch) 

                # Pass through decoder 
                ext_pred = ext_decoder(ext_seq[:, :-1], flag_seq[:, :-1], ext_mask[:, :-1], latent_z)
                ext_mask = ~ext_mask.reshape(-1)
                ext_logit = ext_pred.reshape(-1, ext_pred.shape[-1]) 
                ext_target = ext_seq.reshape(-1)
                ext_loss = F.cross_entropy(ext_logit[ext_mask], ext_target[ext_mask])

                total_loss = ext_loss + vq_loss

                # logging
                if iters % 25 == 0:
                    writer.add_scalar("Loss/Total", total_loss, iters)
                    writer.add_scalar("Loss/extrude", ext_loss, iters)
                    writer.add_scalar("Loss/vq", vq_loss, iters)

                if iters % 25 == 0 and selection is not None:
                    writer.add_histogram('selection', selection, iters)

                # Update AE model
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(params, max_norm=1.0)  # clip gradient
                optimizer.step()
                scheduler.step()  # linear warm up to 1e-3
                iters += 1

        writer.flush()

        # save model after n epoch
        if (epoch+1) % 100 == 0:
            torch.save(ext_encoder.state_dict(), os.path.join(args.output,'extenc_epoch_'+str(epoch+1)+'.pt'))
            torch.save(ext_decoder.state_dict(), os.path.join(args.output,'extdec_epoch_'+str(epoch+1)+'.pt'))

        # Validation loss 
        print('Testing...')
        if (epoch+1) % 30 == 0:
            ext_losses = []
            with tqdm(val_dataloader, unit="batch") as batch_data:
                for ext_seq, flag_seq, ext_mask in batch_data:
                    with torch.no_grad():
                        ext_seq = ext_seq.to(device)
                        flag_seq = flag_seq.to(device)
                        ext_mask = ext_mask.to(device)
                
                        # Pass through encoder 
                        latent_z, _, _ = ext_encoder(ext_seq, flag_seq, ext_mask, epoch) 

                        # Pass through decoder 
                        ext_pred = ext_decoder(ext_seq[:, :-1], flag_seq[:, :-1], ext_mask[:, :-1], latent_z)
                        ext_mask = ~ext_mask.reshape(-1)
                        ext_logit = ext_pred.reshape(-1, ext_pred.shape[-1]) 
                        ext_target = ext_seq.reshape(-1)
                        ext_loss = F.cross_entropy(ext_logit[ext_mask], ext_target[ext_mask])
                        ext_losses.append(ext_loss.item())
            avg_ext = np.array(ext_losses).mean()
            print(f'Epoch {epoch}: avg extrude loss is {avg_ext}')

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batchsize", type=int, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--bit", type=int, required=True)
    parser.add_argument("--maxlen", type=int, required=True)
    args = parser.parse_args()

    # Create training folder
    result_folder = args.output
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        
    # Start training 
    train(args)