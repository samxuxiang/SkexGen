import torch
import numpy as np
import pickle 
import random

SKETCH_R = 1
RADIUS_R = 1
EXTRUDE_R = 1.0
SCALE_R =  1.4
OFFSET_R = 0.9
PIX_PAD = 4
CMD_PAD = 3
COORD_PAD = 4
EXT_PAD = 1
EXTRA_PAD = 1
R_PAD = 2
AUG_RANGE = 5
MAX_EXT = 5


class SketchData(torch.utils.data.Dataset):
    """ sketch dataset """
    def __init__(self, data_path, invalid_uid, MAX_LEN):  
        self.maxlen = MAX_LEN 
        self.maxlen_pix = 0 
        self.maxlen_cmd = 0
        self.maxlen_ext = 0
        self.maxlen_se = 0

        with open(invalid_uid, 'rb') as f:
            invalid_uids = pickle.load(f)
        invaliduid = {}
        for invalid in invalid_uids:
            invaliduid[invalid] = True
        
        ######################
        ## Load sketch data ##
        ######################
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        self.data = {}
        for index in range(len(data)):
            vec_data = data[index]
            pix_len = vec_data['len_pix']
            cmd_len = vec_data['len_cmd']
            ext_len = vec_data['len_ext']
            num_se = vec_data['num_se']
            total_len = pix_len + EXTRA_PAD 
            uid = vec_data['name']

            if total_len <= self.maxlen and uid not in invaliduid:
                self.data[uid] = vec_data
                if pix_len+EXTRA_PAD > self.maxlen_pix:
                    self.maxlen_pix = pix_len+EXTRA_PAD
                if cmd_len+EXTRA_PAD > self.maxlen_cmd:
                    self.maxlen_cmd = cmd_len+EXTRA_PAD
                if ext_len+EXTRA_PAD > self.maxlen_ext:
                    self.maxlen_ext = ext_len+EXTRA_PAD
                if num_se > self.maxlen_se:
                    self.maxlen_se = num_se

        self.uids = sorted(list(set(self.data.keys()).intersection(set(self.data.keys()))))
        print(f'Sketch Post-Filter: {len(self.uids)}, Keep Ratio: {100*len(self.uids)/len(data):.2f}%')
        print(f'Max Pix {self.maxlen_pix}, Max CMD {self.maxlen_cmd}, Max SE {self.maxlen_se}')


    def __len__(self):
        return len(self.uids)


    def prepare_batch_sketch(self, pixel_v, xy_v):
        keys = np.ones(len(pixel_v))
        padding = np.zeros(self.maxlen_pix-len(pixel_v)).astype(int)  
        pixel_v_flat = np.concatenate([pixel_v, padding], axis=0)
        pixel_v_mask = 1-np.concatenate([keys, padding]) == 1   
        padding = np.zeros((self.maxlen_pix-len(xy_v), 2)).astype(int)  
        xy_v_flat = np.concatenate([xy_v, padding], axis=0)
        return pixel_v_flat, xy_v_flat, pixel_v_mask


    def prepare_batch_cmd(self, command):
        keys = np.ones(len(command))
        padding = np.zeros(self.maxlen_cmd-len(command)).astype(int)  
        command_pad = np.concatenate([command, padding])
        mask = 1-np.concatenate([keys, padding]) == 1   
        return command_pad, mask


    def __getitem__(self, index):
        uid = self.uids[index]
        vec_data = self.data[uid]
        pix_tokens = vec_data['se_pix']
        xy_tokens = vec_data['se_xy']
        cmd_tokens = vec_data['se_cmd']

        pixs = np.hstack(pix_tokens)+EXTRA_PAD
        pixs = np.concatenate((pixs, np.zeros(1).astype(int)))
        
        xys = np.vstack(xy_tokens)+EXTRA_PAD
        xys = np.concatenate((xys, np.zeros((1,2)).astype(int)))

        cmds = np.hstack(cmd_tokens)+EXTRA_PAD
        cmds = np.concatenate((cmds, np.zeros(1).astype(int)))
    
        pix_seq, xy_seq, mask = self.prepare_batch_sketch(pixs, xys)
        cmd_seq, cmd_mask = self.prepare_batch_cmd(cmds)

        # Quantization augmentation
        aug_xys = []
        for xy in xys:
            if xy[0] <= COORD_PAD and xy[1] <= COORD_PAD:
                aug_xys.append(xy - COORD_PAD - EXTRA_PAD)
            else:
                new_xy = xy - COORD_PAD - EXTRA_PAD 
                new_xy[0] = new_xy[0] + random.randint(-AUG_RANGE, +AUG_RANGE)
                new_xy[1] = new_xy[1] + random.randint(-AUG_RANGE, +AUG_RANGE) 
                new_xy = np.clip(new_xy, a_min=0, a_max=2**6-1)         
                aug_xys.append(new_xy)
        _xys_ = np.vstack(aug_xys) + EXTRA_PAD + COORD_PAD
        
        # # Augment the pix value according to XY
        aug_pix = []
        for xy in aug_xys:
            if xy[0] >= 0 and xy[1] >= 0:
                aug_pix.append(xy[1]*(2**6)+xy[0])
            else:
                aug_pix.append(xy[0])
        _pixs_ = np.hstack(aug_pix) + EXTRA_PAD + PIX_PAD

        pix_seq_aug, xy_seq_aug, mask_aug = self.prepare_batch_sketch(_pixs_, _xys_)

        return cmd_seq, cmd_mask, pix_seq, xy_seq, mask, \
               pix_seq_aug, xy_seq_aug, mask_aug


class ARDataset(torch.utils.data.Dataset):
    """ ar dataset """
    def __init__(self, datapath, maxlen):
        with open(datapath, 'rb') as f:
            self.data = pickle.load(f)
        self.maxlen = maxlen
        print(len(self.data))
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        code = self.data[index]
        return code


class SketchExtData(torch.utils.data.Dataset):
    """ sketch dataset """
    def __init__(self, data, invalid_uid, MAX_LEN):  
        self.maxlen = MAX_LEN 
        self.maxlen_pix = 0 
        self.maxlen_cmd = 0
        self.maxlen_ext = 0

        # Convert list to dictionary, signicantly faster for key indexing
        with open(invalid_uid, 'rb') as f:
            invalid_uids = pickle.load(f)
        invaliduid = {}
        for invalid in invalid_uids:
            invaliduid[invalid] = True
        
        ######################
        ## Load sketch data ##
        ######################
        with open(data, 'rb') as f:
            data = pickle.load(f)

        self.data = {}
        for index in range(len(data)):
            vec_data = data[index]
            pix_len = vec_data['len_pix']
            cmd_len = vec_data['len_cmd']
            ext_len = vec_data['len_ext']
            total_len = pix_len + EXTRA_PAD 
            uid = vec_data['name']
            ext_len = vec_data['len_ext']
           
            if total_len <= self.maxlen and vec_data['num_se']<=MAX_EXT and uid not in invaliduid:
                self.data[uid] = vec_data
                ext_len = vec_data['len_ext']
                if pix_len+EXTRA_PAD > self.maxlen_pix:
                    self.maxlen_pix = pix_len+EXTRA_PAD
                if cmd_len+EXTRA_PAD > self.maxlen_cmd:
                    self.maxlen_cmd = cmd_len+EXTRA_PAD
                if ext_len+EXTRA_PAD > self.maxlen_ext:
                    self.maxlen_ext = ext_len+EXTRA_PAD
        
        self.uids = sorted(list(set(self.data.keys())))
        print(f'Sketch Post-Filter: {len(self.uids)}, Keep Ratio: {100*len(self.uids)/len(data):.2f}%')
        print(f'Max Pix {self.maxlen_pix}, Max CMD {self.maxlen_cmd}')
     

    def __len__(self):
        return len(self.uids)


    def prepare_batch_sketch(self, pixel_v, xy_v):
        keys = np.ones(len(pixel_v))
        padding = np.zeros(self.maxlen_pix-len(pixel_v)).astype(int)  
        pixel_v_flat = np.concatenate([pixel_v, padding], axis=0)
        pixel_v_mask = 1-np.concatenate([keys, padding]) == 1   
        padding = np.zeros((self.maxlen_pix-len(xy_v), 2)).astype(int)  
        xy_v_flat = np.concatenate([xy_v, padding], axis=0)
        return pixel_v_flat, xy_v_flat, pixel_v_mask


    def prepare_batch_cmd(self, command):
        keys = np.ones(len(command))
        padding = np.zeros(self.maxlen_cmd-len(command)).astype(int)  
        command_pad = np.concatenate([command, padding])
        mask = 1-np.concatenate([keys, padding]) == 1   
        return command_pad, mask


    def prepare_batch_extrude(self, ext, flags):
        keys = np.ones(len(ext))
        padding = np.zeros(self.maxlen_ext-len(ext)).astype(int)  
        flag_pad = np.concatenate([flags, padding], axis=0)
        ext_flat = np.concatenate([ext, padding], axis=0)
        ext_mask = 1-np.concatenate([keys, padding]) == 1   
        return ext_flat, flag_pad, ext_mask


    def __getitem__(self, index):
        uid = self.uids[index]
        vec_data = self.data[uid]
        pix_tokens = vec_data['se_pix']
        xy_tokens = vec_data['se_xy']
        cmd_tokens = vec_data['se_cmd']
        ext_tokens = vec_data['se_ext']

        exts = np.hstack(ext_tokens) + EXTRA_PAD
        exts = np.concatenate((exts, np.zeros(1).astype(int)))
        ext_flags = np.hstack([1,1,2,2,2,3,3,3,3,3,3,3,3,3,4,5,6,6,7] * len(ext_tokens))
        ext_flags = np.concatenate((ext_flags, np.zeros(1).astype(int)))
        ext_seq, flag_seq, ext_mask = self.prepare_batch_extrude(exts, ext_flags)

        pixs = np.hstack(pix_tokens)+EXTRA_PAD
        pixs = np.concatenate((pixs, np.zeros(1).astype(int)))
        
        xys = np.vstack(xy_tokens)+EXTRA_PAD
        xys = np.concatenate((xys, np.zeros((1,2)).astype(int)))

        cmds = np.hstack(cmd_tokens)+EXTRA_PAD
        cmds = np.concatenate((cmds, np.zeros(1).astype(int)))
    
        pix_seq, xy_seq, sketch_mask = self.prepare_batch_sketch(pixs, xys)
        cmd_seq, cmd_mask = self.prepare_batch_cmd(cmds)
    
        return cmd_seq, cmd_mask, pix_seq, xy_seq, sketch_mask, flag_seq, ext_seq, ext_mask


class ExtData(torch.utils.data.Dataset):
    """ extrude dataset """
    def __init__(self, data_path, invalid, MAX_LEN):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.maxlen = MAX_LEN 
        self.maxlen_ext = 0 

        with open(invalid, 'rb') as f:
            invalid_uids = pickle.load(f)
        invaliduid = {}
        for invalid in invalid_uids:
            invaliduid[invalid] = True

        # Filter out too long results
        self.data = []
        for index in range(len(data)):
            vec_data = data[index]
            uid = vec_data['name']

            if vec_data['num_se'] <= self.maxlen:  # and uid not in invaliduid:
                self.data.append(vec_data)
                ext_len = vec_data['len_ext']
                if ext_len+EXTRA_PAD > self.maxlen_ext:
                    self.maxlen_ext = ext_len+EXTRA_PAD

        print(f'Sketch Post-Filter: {len(self.data)}, Keep Ratio: {100*len(self.data)/len(data):.2f}%')
        print(f'Max Ext {self.maxlen_ext}')

     
    def __len__(self):
        return len(self.data)


    def prepare_batch_extrude(self, ext, flags):
        keys = np.ones(len(ext))
        padding = np.zeros(self.maxlen_ext-len(ext)).astype(int)  
        flag_pad = np.concatenate([flags, padding], axis=0)
        ext_flat = np.concatenate([ext, padding], axis=0)
        ext_mask = 1-np.concatenate([keys, padding]) == 1   
        return ext_flat, flag_pad, ext_mask


    def __getitem__(self, index):
        vec_data = self.data[index]
        ext_tokens = vec_data['se_ext']
       
        exts = np.hstack(ext_tokens) + EXT_PAD
        exts = np.concatenate((exts, np.zeros(1).astype(int)))
        
        ext_flags = np.hstack([1,1,2,2,2,3,3,3,3,3,3,3,3,3,4,5,6,6,7] * len(ext_tokens))
        ext_flags = np.concatenate((ext_flags, np.zeros(1).astype(int)))
        
        ext_seq, flag_seq, ext_mask = self.prepare_batch_extrude(exts, ext_flags)
        
        return ext_seq, flag_seq, ext_mask
