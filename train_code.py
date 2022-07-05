import os
import torch
import argparse
from model.ar import ARModel
from dataset import ARDataset
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter



def train(args):
    # Initialize gpu device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda:0")
    
    # Initialize dataset loader
    dataset = ARDataset(datapath=args.input, maxlen=args.seqlen) 
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             shuffle=True, 
                                             batch_size=args.batchsize,
                                             num_workers=4)
    # Initialize vertex model
    model = ARModel(
        config={
            'hidden_dim': 512,
            'embed_dim': 256, 
            'num_layers': 4,
            'num_heads': 8,
            'dropout_rate': 0.0
        },
        max_len=args.seqlen,
        classes=args.code,
    )
    model = model.to(device).train()
    
    # Initialize optimizer
    network_parameters = list(model.parameters()) 
    optimizer = torch.optim.Adam(network_parameters, lr=1e-3)
   
    # logging 
    writer = SummaryWriter(log_dir=args.output)
    
    # Main training loop
    iters = 0
    print('Start training...')

    for epoch in range(1000):
        print(epoch)

        for batch in dataloader:
            code = batch
            code = code.to(device)

            # Pass through vertex prediction module 
            logits = model(code[:, :-1])

            c_pred = logits.reshape(-1, logits.shape[-1]) 
            c_target = code.reshape(-1)
            code_loss = F.cross_entropy(c_pred, c_target)
           
            total_loss = code_loss

            # logging
            if iters % 20 == 0:
                writer.add_scalar("Loss/Total", total_loss, iters)

            # Backprop 
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(network_parameters, max_norm=1.0)  # clip gradient
            optimizer.step()
            iters += 1

        writer.flush()
        # save model after n epoch
        if (epoch+1) % 500 == 0:
            torch.save(model.state_dict(), os.path.join(args.output,'ar_epoch_'+str(epoch+1)+'.pt'))

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, help="Output folder to save the data [default: output]")
    parser.add_argument("--batchsize", type=int, help="Training Batch Size")
    parser.add_argument("--bits", type=int, help="Qauntization bit number")
    parser.add_argument("--device", type=str, help="CUDA Device Index")
    parser.add_argument("--seqlen", type=int)
    parser.add_argument("--code", type=int)
    args = parser.parse_args()

    # Create training folder
    result_folder = args.output
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        
    # Start training 
    train(args)
