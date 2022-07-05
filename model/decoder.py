from .layers.transformer import *
from .layers.improved_transformer import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

PIX_PAD = 4
CMD_PAD = 3
COORD_PAD = 4
EXT_PAD = 1
EXTRA_PAD = 1
R_PAD = 2
NUM_FLAG = 9 
SAMPLE_PROB = 0.5 # or 0.95


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=250):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len, dtype=torch.long).unsqueeze(1)
        self.register_buffer('position', position)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.pos_embed.weight, mode="fan_in")

    def forward(self, x):
        pos = self.position[:x.size(0)]
        x = x + self.pos_embed(pos)
        return self.dropout(x)


class SketchDecoder(nn.Module):
  """
  Autoregressive generative model 
  """

  def __init__(self,
               config,
               pix_len,
               cmd_len,
               quantization_bits=8):
    """
    Initializes FaceModel.
    """
    super(SketchDecoder, self).__init__()
    self.pix_len = pix_len
    self.cmd_len = cmd_len
    self.embed_dim = config['embed_dim']
    self.bits = quantization_bits

    # Sketch encoders
    self.coord_embed_x = Embedder(2**self.bits+COORD_PAD+EXTRA_PAD, self.embed_dim)
    self.coord_embed_y = Embedder(2**self.bits+COORD_PAD+EXTRA_PAD, self.embed_dim)

    self.pixel_embed = Embedder(2**self.bits * 2**self.bits+PIX_PAD+EXTRA_PAD, self.embed_dim)
    self.pos_embed = PositionalEncoding(max_len=self.pix_len, d_model=self.embed_dim)
    self.logit_fc = nn.Linear(self.embed_dim, 2**self.bits * 2**self.bits+PIX_PAD+EXTRA_PAD)
    self.mempos = Embedder(2, self.embed_dim)
    
    decoder_layers = TransformerDecoderLayerImproved(d_model=self.embed_dim, 
                        dim_feedforward= config['hidden_dim'], nhead=config['num_heads'], dropout=config['dropout_rate'])
    decoder_norm = LayerNorm(self.embed_dim)
    self.decoder = TransformerDecoder(decoder_layers, config['num_layers'], decoder_norm)


  def forward(self, pixel_v, xy_v, pixel_mask, latent_z, is_training=True):
    """ forward pass """
    if pixel_v[0] is None:
      c_bs = len(pixel_v)
      c_seqlen = 0
    else:
      c_bs, c_seqlen = pixel_v.shape[0], pixel_v.shape[1]  
 
    # Context embedding values
    context_embedding = torch.zeros((1, c_bs, self.embed_dim)).cuda() # [1, bs, dim]

    # Data input embedding
    if c_seqlen > 0:
      coord_embed = self.coord_embed_x(xy_v[...,0]) + self.coord_embed_y(xy_v[...,1]) # [bs, vlen, dim]
      pixel_embed = self.pixel_embed(pixel_v)
      embed_inputs = pixel_embed + coord_embed
      
      #### REMOVED ####
      # if is_training:  
      #   # dropout (optional)
      #   drop_index = np.random.uniform(low=0.0, high=1.0, size=(embed_inputs.shape[0], embed_inputs.shape[1])) < DROP_RATE
      #   embed_inputs[drop_index, :] = 0.0

      embeddings = torch.cat([context_embedding, embed_inputs.transpose(0,1)], axis=0)
      decoder_inputs = self.pos_embed(embeddings) 

    else:
      decoder_inputs = self.pos_embed(context_embedding)
  
    memory = latent_z.transpose(0,1)
    pos_mem = self.mempos((torch.Tensor([0]*4 + [1]*2)).long().cuda()).unsqueeze(1).repeat(1, c_bs, 1)  
    memory_encode = memory + pos_mem
    
    nopeak_mask = torch.nn.Transformer.generate_square_subsequent_mask(c_seqlen+1).cuda()  # masked with -inf
    if pixel_mask is not None:
      pixel_mask = torch.cat([(torch.zeros([c_bs, context_embedding.shape[0]])==1).cuda(), pixel_mask], axis=1)    
    decoder_out = self.decoder(tgt=decoder_inputs, memory=memory_encode, memory_key_padding_mask=None,
                               tgt_mask=nopeak_mask, tgt_key_padding_mask=pixel_mask)

    # Logits fc
    logits = self.logit_fc(decoder_out)  # [seqlen, bs, dim] 
    return logits.transpose(1,0)
    

  def sample(self, n_samples,  latent_z, latent_ext):
    """ sample from distribution (top-k, top-p) """
    pix_samples = []
    xy_samples = []
    latent_ext_samples = []
    top_k = 0
    top_p = SAMPLE_PROB

    left_list = np.array(list(range(n_samples)))
    done_list = []

    # Mapping from pixel index to xy coordiante
    pixel2xy = {}
    x=np.linspace(0, 2**self.bits-1, 2**self.bits)
    y=np.linspace(0, 2**self.bits-1, 2**self.bits)
    xx,yy=np.meshgrid(x,y)
    xy_grid = (np.array((xx.ravel(), yy.ravel())).T).astype(int)
    for pixel, xy in enumerate(xy_grid):
      pixel2xy[pixel] = xy+COORD_PAD+EXT_PAD

    # Sample per token
    for k in range(self.pix_len):
      if k == 0:
        pixel_seq = [None] * n_samples
        xy_seq = [None, None] * n_samples
      
      # pass through model
      with torch.no_grad():
        p_pred = self.forward(pixel_seq, xy_seq, None, latent_z, is_training=False)
        p_logits = p_pred[:, -1, :]

      next_pixels = []
      # Top-p sampling of next pixel
      for logit in p_logits: 
        filtered_logits = top_k_top_p_filtering(logit, top_k=top_k, top_p=top_p)
        next_pixel = torch.multinomial(F.softmax(filtered_logits, dim=-1), 1)
        next_pixels.append(next_pixel.item())

      # Convert pixel index to xy coordinate
      next_xys = []
      for pixel in next_pixels:
        if pixel >= PIX_PAD+EXT_PAD:
          xy = pixel2xy[pixel-PIX_PAD-EXT_PAD]
        else:
          xy = np.array([pixel, pixel]).astype(int)
        next_xys.append(xy)
      next_xys = np.vstack(next_xys)  # [BS, 2]
      next_pixels = np.vstack(next_pixels)  # [BS, 1]
        
      # Add next tokens
      nextp_seq = torch.LongTensor(next_pixels).view(len(next_pixels), 1).cuda()
      nextxy_seq = torch.LongTensor(next_xys).unsqueeze(1).cuda()
      
      if pixel_seq[0] is None:
        pixel_seq = nextp_seq
        xy_seq = nextxy_seq
      else:
        pixel_seq = torch.cat([pixel_seq, nextp_seq], 1)
        xy_seq = torch.cat([xy_seq, nextxy_seq], 1)
      
      # Early stopping
      done_idx = np.where(next_pixels==0)[0]
      if len(done_idx) > 0:
        done_pixs = pixel_seq[done_idx] 
        done_xys = xy_seq[done_idx]
        done_ext = latent_ext[done_idx]
        done_list.append(left_list[done_idx])
       
        for pix, xy, ext in zip(done_pixs, done_xys, done_ext):
          pix = pix.detach().cpu().numpy()
          xy = xy.detach().cpu().numpy()
          pix_samples.append(pix)
          xy_samples.append(xy)
          latent_ext_samples.append(ext.unsqueeze(0))
  
      left_idx = np.where(next_pixels!=0)[0]
      if len(left_idx) == 0:
        break # no more jobs to do
      else:
        pixel_seq = pixel_seq[left_idx]
        xy_seq = xy_seq[left_idx]
        left_list = left_list[left_idx]
        if latent_z is not None:
          latent_z = latent_z[left_idx]
          latent_ext = latent_ext[left_idx]
    
    return pix_samples, latent_ext_samples, np.hstack(done_list)


class EXTDecoder(nn.Module):
  """
  Autoregressive generative model 
  """

  def __init__(self,
               config,
               quantization_bits=8,
               max_len=200):
    """
    Initializes FaceModel.
    """
    super(EXTDecoder, self).__init__()
    self.max_len = max_len
    self.embed_dim = config['embed_dim']
    self.bits = quantization_bits

    self.flag_embed = Embedder(NUM_FLAG, self.embed_dim)
    self.ext_embed = Embedder(2**self.bits+EXT_PAD+EXTRA_PAD, self.embed_dim)
    self.pos_embed = PositionalEncoding(max_len=self.max_len, d_model=self.embed_dim)
    self.logit_fc = nn.Linear(self.embed_dim, 2**self.bits+EXT_PAD+EXTRA_PAD)
    
    decoder_layers = TransformerDecoderLayerImproved(d_model=self.embed_dim, 
                        dim_feedforward= config['hidden_dim'],
                        nhead=config['num_heads'], dropout=config['dropout_rate'])
    decoder_norm = LayerNorm(self.embed_dim)
    self.decoder = TransformerDecoder(decoder_layers, config['num_layers'], decoder_norm)
      

  def forward(self, ext_v, flags, ext_mask, code=None, is_training=True):
    """ forward pass """
    if ext_v[0] is None:
      c_bs = len(ext_v)
      c_seqlen = 0
    else:
      c_bs, c_seqlen = ext_v.shape[0], ext_v.shape[1]  
    
    # Context embedding values
    context_embedding = torch.zeros((1, c_bs, self.embed_dim)).cuda() # [1, bs, dim]

    # Data input embedding
    embeddings = None
    if c_seqlen > 0:
      ext_embed = self.ext_embed(ext_v)
      flag_embed = self.flag_embed(flags)
      embed_inputs = flag_embed+ext_embed
      embeddings = torch.cat([context_embedding, embed_inputs.transpose(0,1)], axis=0)
      decoder_inputs = self.pos_embed(embeddings) 
     
    else:
      decoder_inputs = self.pos_embed(context_embedding)

    # Pass through decoder
    memory = code.transpose(0,1)
  
    nopeak_mask = torch.nn.Transformer.generate_square_subsequent_mask(c_seqlen+1).cuda()  # masked with -inf
    if ext_mask is not None:
      ext_mask = torch.cat([(torch.zeros([c_bs, 1])==1).cuda(), ext_mask], axis=1)
    decoder_out = self.decoder(tgt=decoder_inputs, memory=memory, 
                            tgt_mask=nopeak_mask, tgt_key_padding_mask=ext_mask)

    # Logits fc
    logits = self.logit_fc(decoder_out)  # [seqlen, bs, dim] 
    return logits.transpose(1,0)
    

  def sample(self, n_samples, latent_z=None, sample_pixels=None):
    """ sample from distribution (top-k, top-p) """
    samples = []
    top_k = 0
    top_p = SAMPLE_PROB

    left_list = np.array(list(range(n_samples)))
    done_list = []

    # Sample per token
    for k in range(self.max_len):
      if k == 0:
        pixel_seq = [None] * n_samples
        flag_seq = [None] * n_samples

      # pass through model
      with torch.no_grad():
        p_pred = self.forward(pixel_seq, flag_seq, None, latent_z, is_training=False)
        p_logits = p_pred[:, -1, :]

      next_pixels = []
      # Top-p sampling of next pixel
      for idx, logit in enumerate(p_logits): 
        filtered_logits = top_k_top_p_filtering(logit, top_k=top_k, top_p=top_p)
        next_pixel = torch.multinomial(F.softmax(filtered_logits, dim=-1), 1)
        next_pixels.append(next_pixel.item())
        
      # Add next tokens
      next_pixels = np.vstack(next_pixels)  # [BS, 1]
      nextp_seq = torch.LongTensor(next_pixels).view(len(next_pixels), 1).cuda()
      
      if pixel_seq[0] is None:
        pixel_seq = nextp_seq
        flag_seq = torch.LongTensor(np.vstack([1] * n_samples)).cuda()
      
      else:
        pixel_seq = torch.cat([pixel_seq, nextp_seq], 1)
        next_flag = np.zeros(len(nextp_seq)).astype(int)
      
        # Add flag
        for index, nextp in enumerate(nextp_seq): 
          start_idx = len(flag_seq[index]) % 19
          e_flag = [1,1,2,2,2,3,3,3,3,3,3,3,3,3,4,5,6,6,7]
          next_flag[index] = e_flag[start_idx]

        next_flag = next_flag.reshape(-1, 1)
        flag_seq = torch.cat([flag_seq, torch.LongTensor(next_flag).cuda()], 1)
      
      # Early stopping
      done_idx = np.where(next_pixels==0)[0]
      if len(done_idx) > 0:
        done_exts = pixel_seq[done_idx] 
        done_pixs = [sample_pixels[x] for x in done_idx]
        done_list.append(left_list[done_idx])
        
        for pix_job, ext_job in zip(done_pixs, done_exts):
          ext_job = ext_job.detach().cpu().numpy()
          pix_clean = pix_job[:np.where(pix_job==0)[0][0]]
          ext_clean = ext_job[:np.where(ext_job==0)[0][0]]
          pix_splits = np.split(pix_clean, np.where(pix_clean==1)[0]+1)[:-1]
          ext_splits = np.split(ext_clean, np.where(ext_clean==1)[0]+1)[:-1]

          merged = []
          for pix_split, ext_split in zip(pix_splits, ext_splits):
            merged.append(pix_split)
            merged.append(ext_split)
          merged.append(np.zeros(1).astype(int)) 
          merged = np.hstack(merged)
          samples.append(merged)
      
      
      left_idx = np.where(next_pixels!=0)[0]
      if len(left_idx) == 0:
        break # no more jobs to do
      else:
        pixel_seq = pixel_seq[left_idx]
        flag_seq = flag_seq[left_idx]
        left_list = left_list[left_idx]
        if latent_z is not None:
          latent_z = latent_z[left_idx]
          sample_pixels = [sample_pixels[x] for x in left_idx]

    return samples, np.hstack(done_list)
