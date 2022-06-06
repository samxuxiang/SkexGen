import os
import torch
import argparse
import pdb
from multiprocessing import Pool
from dataset import SketchExtData
from model.ar import ARModel
from model.decoder import SketchDecoder, EXTDecoder
from model.encoder import PARAMEncoder, CMDEncoder, EXTEncoder
from utils import CADparser, write_obj
import pickle 



def test(args):
    # Initialize gpu device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device("cuda:0")

    dataset = SketchExtData(args.data, args.invalid, args.maxlen)
  
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
    cmd_encoder.load_state_dict(torch.load(os.path.join(args.weight, 'cmdenc_epoch_'+str(args.epoch)+'.pt')))
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
    param_encoder.load_state_dict(torch.load(os.path.join(args.weight, 'paramenc_epoch_'+str(args.epoch)+'.pt')))
    param_encoder = param_encoder.to(device).eval()

    sketch_decoder = SketchDecoder(
        config={
            'hidden_dim': 512,
            'embed_dim': 256, 
            'num_layers': 4, 
            'num_heads': 8,
            'dropout_rate': 0.1  
        },
        pix_len=dataset.maxlen_pix,
        cmd_len=dataset.maxlen_cmd,
        quantization_bits=args.bit,
    )
    sketch_decoder.load_state_dict(torch.load(os.path.join(args.weight, 'sketchdec_epoch_'+str(args.epoch)+'.pt')))
    sketch_decoder = sketch_decoder.to(device).eval()
    
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
    ext_encoder.load_state_dict(torch.load('proj_log/6bit_4x1000_ip_nodrop_128codedim_maxlen4_flag/extenc_epoch_500.pt'))
    ext_encoder = ext_encoder.to(device).eval()

    ext_decoder = EXTDecoder(
        config={
            'hidden_dim': 512,
            'embed_dim': 256, 
            'num_layers': 4, 
            'num_heads': 8,
            'dropout_rate': 0.1  
        },
        max_len=dataset.maxlen_ext,
        quantization_bits=args.bit,
    )
    ext_decoder.load_state_dict(torch.load('proj_log/6bit_4x1000_ip_nodrop_128codedim_maxlen4_flag/extdec_epoch_500.pt'))
    ext_decoder = ext_decoder.to(device).eval()

    ar_model = ARModel(
        config={
            'hidden_dim': 512,
            'embed_dim': 256, 
            'num_layers': 4,
            'num_heads': 8,
            'dropout_rate': 0.1
        },
        max_len=10,
        classes=1000,
    )
    ar_model.load_state_dict(torch.load(os.path.join(args.weight, 'ucode_remove10_500', 'train_code_ar', 'ar_epoch_1000.pt')))
    ar_model = ar_model.to(device).eval()

    print('Random Generation...')
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    NUM_SAMPLE = 10000
    count = 0
    cad = []
    BS = 1024
    cmd_codebook = cmd_encoder.vq_vae._embedding
    param_codebook = param_encoder.vq_vae._embedding
    ext_codebook = ext_encoder.vq_vae._embedding
  
    while len(cad) < NUM_SAMPLE:
        count += BS

        with torch.no_grad():
            codes = ar_model.sample(n_samples=BS)
            cmd_code = codes[:,:4] 
            param_code = codes[:,4:6] ##
            ext_code = codes[:,6:] ##

            cmd_codes = []
            param_codes = []
            ext_codes = []
            for cmd, param, ext in zip(cmd_code, param_code, ext_code):
                if torch.max(cmd) >= 500 or torch.max(param) >= 1000 or torch.max(ext) >= 1000:
                    continue
                else:
                    cmd_codes.append(cmd)
                    param_codes.append(param)
                    ext_codes.append(ext)
            cmd_codes = torch.vstack(cmd_codes)
            param_codes = torch.vstack(param_codes)
            ext_codes = torch.vstack(ext_codes)

            latent_cmd = cmd_encoder.up(cmd_codebook(cmd_codes))
            latent_param = param_encoder.up(param_codebook(param_codes))
            latent_ext = ext_encoder.up(ext_codebook(ext_codes))
            latent_sketch = torch.cat((latent_cmd, latent_param), 1)
                
        # Parallel Sample Sketches 
        sample_pixels, latent_ext_samples = sketch_decoder.sample(n_samples=latent_sketch.shape[0], latent_z=latent_sketch, latent_ext=latent_ext)
        _latent_ext_ = torch.vstack(latent_ext_samples)
        assert len(_latent_ext_) == len(sample_pixels)

        # Parallel Sample Extrudes 
        sample_merges = ext_decoder.sample(n_samples=len(sample_pixels), latent_z=_latent_ext_, sample_pixels=sample_pixels)
        cad += sample_merges
        print(f'cad:{len(cad)}')
        
    # # Parallel raster OBJ
    gen_data = []
    obj_data = []

    load_iter = Pool(36).imap(raster_cad, cad)
    for data_sample, data_obj in load_iter:
        gen_data += data_sample
        obj_data += data_obj
    print(len(gen_data))

    print('Saving...')
    print('Writting OBJ...')
    for index, value in enumerate(gen_data):
        output = os.path.join(args.output, str(index).zfill(5))
        if not os.path.exists(output):
            os.makedirs(output)
        write_obj(output, value)

    with open(os.path.join(args.output, 'objs.pkl'), "wb") as tf:
        pickle.dump(obj_data, tf)


def raster_cad(pixels):   
    try:
        parser = CADparser(args.bit)
        parsed_data, saved_data = parser.perform(pixels)
        return [parsed_data], [saved_data]
    except Exception as error_msg:  
        return [], []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help="Ouput folder containing the sampled results")
    parser.add_argument("--weight", type=str, required=True, help="Input folder containing the saved model")
    parser.add_argument("--epoch", type=int, required=True, help="weight epoch")
    parser.add_argument("--device", type=int, help="CUDA Device Index")
    parser.add_argument("--maxlen", type=int, help="maximum token length")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--invalid", type=str, required=True)
    parser.add_argument("--bit", type=int, help="quantization bit")
    args = parser.parse_args()

    test(args)

