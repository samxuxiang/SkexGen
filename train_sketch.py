import os
import torch
import argparse
from dataset import SketchData
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter
from model.encoder import PARAMEncoder, CMDEncoder
from model.decoder import SketchDecoder
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
    train_dataset = SketchData(args.train_data, args.invalid, args.maxlen)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                             shuffle=True, 
                                             batch_size=args.batchsize,
                                             num_workers=5,
                                             pin_memory=True)

    val_dataset = SketchData(args.val_data, args.invalid, args.maxlen)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                 shuffle=False, 
                                                 batch_size=args.batchsize,
                                                 num_workers=5)
    
    # Initialize models
    cmd_encoder = CMDEncoder(
        config={
            'hidden_dim': 512,
            'embed_dim': 256,
            'num_layers': 4,
            'num_heads': 8,
            'dropout_rate': 0.1
        },
        max_len=train_dataset.maxlen_pix,
        code_len = 4,
        num_code = 500,
    )
    cmd_encoder = cmd_encoder.to(device).train()

    param_encoder = PARAMEncoder(
        config={
            'hidden_dim': 512,
            'embed_dim': 256,
            'num_layers': 4,
            'num_heads': 8,
            'dropout_rate': 0.1
        },
        quantization_bits=args.bit,
        max_len=train_dataset.maxlen_pix,
        code_len = 2,
        num_code = 1000,
    )
    param_encoder = param_encoder.to(device).train()

    sketch_decoder = SketchDecoder(
        config={
            'hidden_dim': 512,
            'embed_dim': 256, 
            'num_layers': 4, 
            'num_heads': 8,
            'dropout_rate': 0.1  
        },
        pix_len=train_dataset.maxlen_pix,
        cmd_len=train_dataset.maxlen_cmd,
        quantization_bits=args.bit,
    )
    sketch_decoder = sketch_decoder.to(device).train()
   
    # Initialize optimizer
    params = list(sketch_decoder.parameters()) + list(param_encoder.parameters()) + list(cmd_encoder.parameters()) 
    optimizer = torch.optim.Adam(params, lr=1e-3)
    scheduler = get_constant_schedule_with_warmup(optimizer, 2000)
   
    # logging 
    writer = SummaryWriter(log_dir=args.output)
    
    # Main training loop
    iters = 0
    print('Start training...')
    
    for epoch in range(300):  # 300 epochs is usually enough
        with tqdm(train_dataloader, unit="batch") as batch_data:
            for cmd, cmd_mask, pix, xy, mask, pix_aug, xy_aug, mask_aug in batch_data:
                cmd = cmd.to(device)
                cmd_mask = cmd_mask.to(device)
                pix = pix.to(device) 
                xy = xy.to(device)
                mask = mask.to(device)
                pix_aug = pix_aug.to(device) 
                xy_aug = xy_aug.to(device)
                mask_aug = mask_aug.to(device)

                # Pass through encoders
                latent_cmd, cvq_loss, c_selection = cmd_encoder(cmd, cmd_mask, epoch)
                latent_param, pvq_loss, p_selection = param_encoder(pix_aug, xy_aug, mask_aug, epoch) 
                latent_z = torch.cat((latent_cmd, latent_param), 1)
            
                # Pass through decoder
                pix_pred = sketch_decoder(pix[:, :-1], xy[:, :-1, :], mask[:, :-1], latent_z)
                pix_mask = ~mask.reshape(-1)
                pix_logit = pix_pred.reshape(-1, pix_pred.shape[-1]) 
                pix_target = pix.reshape(-1)
                pix_loss = F.cross_entropy(pix_logit[pix_mask], pix_target[pix_mask])

                # Total loss
                total_loss = pix_loss + cvq_loss + pvq_loss
            
                # logging
                if iters % 25 == 0:
                    writer.add_scalar("Loss/Total", total_loss, iters)
                    writer.add_scalar("Loss/sketch", pix_loss, iters)
                    writer.add_scalar("Loss/param_vq", pvq_loss, iters)
                    writer.add_scalar("Loss/cmd_vq", cvq_loss, iters)

                if iters % 25 == 0 and c_selection is not None and p_selection is not None:
                    writer.add_histogram('cmd_selection', c_selection, iters)
                    writer.add_histogram('param_selection', p_selection, iters)
            
                # Update model
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(params, max_norm=1.0)  # clip gradient
                optimizer.step()
                scheduler.step()  # linear warm up to 1e-3
                iters += 1

        writer.flush()

        # save model after n epoch
        if (epoch+1) % 100 == 0:
            torch.save(sketch_decoder.state_dict(), os.path.join(args.output,'sketchdec_epoch_'+str(epoch+1)+'.pt'))
            torch.save(param_encoder.state_dict(), os.path.join(args.output,'paramenc_epoch_'+str(epoch+1)+'.pt'))
            torch.save(cmd_encoder.state_dict(), os.path.join(args.output,'cmdenc_epoch_'+str(epoch+1)+'.pt'))

        # Validation loss 
        print('Testing...')
        if (epoch+1) % 30 == 0:
            pix_losses = []
            with tqdm(val_dataloader, unit="batch") as batch_data:
                for cmd, cmd_mask, pix, xy, mask, pix_aug, xy_aug, mask_aug in batch_data:
                    with torch.no_grad():
                        cmd = cmd.to(device)
                        cmd_mask = cmd_mask.to(device)
                        pix = pix.to(device) 
                        xy = xy.to(device)
                        mask = mask.to(device)
                        pix_aug = pix_aug.to(device) 
                        xy_aug = xy_aug.to(device)
                        mask_aug = mask_aug.to(device)

                        # Pass through encoders
                        latent_cmd, _, _ = cmd_encoder(cmd, cmd_mask, epoch)
                        latent_param, _, _ = param_encoder(pix_aug, xy_aug, mask_aug, epoch) 
                        latent_z = torch.cat((latent_cmd, latent_param), 1)
                    
                        # Pass through decoder
                        pix_pred = sketch_decoder(pix[:, :-1], xy[:, :-1, :], mask[:, :-1], latent_z)
                        pix_mask = ~mask.reshape(-1)
                        pix_logit = pix_pred.reshape(-1, pix_pred.shape[-1]) 
                        pix_target = pix.reshape(-1)
                        pix_loss = F.cross_entropy(pix_logit[pix_mask], pix_target[pix_mask])
                        pix_losses.append(pix_loss.item())
            avg_pix = np.array(pix_losses).mean()
            print(f'Epoch {epoch}: avg pixel loss is {avg_pix}')

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--invalid", type=str, required=True)
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