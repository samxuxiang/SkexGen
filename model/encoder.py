import torch.nn as nn
import torch
import torch.nn.functional as F
from .layers.transformer import *
from .layers.improved_transformer import *

PIX_PAD = 4
CMD_PAD = 3
COORD_PAD = 4
EXT_PAD = 1
EXTRA_PAD = 1
R_PAD = 2
NUM_FLAG = 9 
INITIAL_PASS = 30


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


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim) 
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon


    def forward(self, inputs):
        seqlen, bs = inputs.shape[0], inputs.shape[1]
        
        # Flatten input
        flat_input = inputs.reshape(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
       
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).reshape(seqlen, bs, self._embedding_dim)

        encodings_flat = encodings.reshape(inputs.shape[0], inputs.shape[1], -1)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach() 
        # convert quantized from BHWC -> BCHW
        return loss, quantized.contiguous(), encodings_flat, encoding_indices


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)


class CMDEncoder(nn.Module):

  def __init__(self,
               config,
               code_len = 4,
               max_len = 80,
               num_code = 500,
               ):
    """Initializes Encoder Model.

    Args:
    """
    super(CMDEncoder, self).__init__()
    self.embed_dim = config['embed_dim']
    self.dropout = config['dropout_rate']
    self.max_len = max_len

    self.code_len = code_len
    self.const_embed = Embedder(self.code_len, self.embed_dim)

    self.c_embed = Embedder(3+CMD_PAD+EXT_PAD, self.embed_dim)
    self.pos_embed = PositionalEncoding(d_model=self.embed_dim, max_len=self.max_len+code_len)
   
    # Transformer encoder
    encoder_layers = TransformerEncoderLayerImproved(d_model=self.embed_dim, nhead=config['num_heads'], 
                                             dim_feedforward=config['hidden_dim'], dropout=self.dropout)
    encoder_norm = LayerNorm(self.embed_dim)
    self.encoder = TransformerEncoder(encoder_layers, config['num_layers'], encoder_norm)
    self.fc = nn.Linear(256, 256)
    
    commitment_cost = 0.25
    decay = 0.99
    self.codebook_dim = 128
    self.vq_vae = VectorQuantizerEMA(num_code, self.codebook_dim, commitment_cost, decay)
    self.down = nn.Linear(self.embed_dim, self.codebook_dim)
    self.up = nn.Linear(self.codebook_dim, self.embed_dim)
    

  def forward(self, command, mask, epoch):
    """ forward pass """
    bs, seq_len = command.shape[0], command.shape[1]

    # Command embedding 
    c_embeds = self.c_embed(command.flatten()).view(bs, seq_len, -1) 

    embeddings = c_embeds.transpose(0,1)
    z_embed = self.const_embed(torch.arange(0, self.code_len).long().cuda()).unsqueeze(1).repeat(1, bs, 1) 
    embed_input = torch.cat([z_embed, embeddings], dim=0)
    encoder_input = self.pos_embed(embed_input) 

    # Pass through transformer encoder
    if mask is not None:
        mask = torch.cat([(torch.zeros([bs, self.code_len])==1).cuda(), mask], axis=1)
    outputs = self.encoder(src=encoder_input, src_key_padding_mask=mask)  
    z_encoded = outputs[0:self.code_len]

    if epoch < INITIAL_PASS:
      vq_loss = 0.0 
      selection = None 
      quantized_up = self.up(self.down(z_encoded)).transpose(0,1)
    else:
      vq_loss, quantized, _, selection = self.vq_vae(self.down(z_encoded))
      quantized_up = self.up(quantized).transpose(0,1)
    return quantized_up, vq_loss, selection 


  def get_code(self, command, mask, return_np=True):
    """ forward pass """
    bs, seq_len = command.shape[0], command.shape[1]

    # Command embedding 
    c_embeds = self.c_embed(command.flatten()).view(bs, seq_len, -1) 

    embeddings = c_embeds.transpose(0,1)
    z_embed = self.const_embed(torch.arange(0, self.code_len).long().cuda()).unsqueeze(1).repeat(1, bs, 1) 
    embed_input = torch.cat([z_embed, embeddings], dim=0)
    encoder_input = self.pos_embed(embed_input) 

    # Pass through transformer encoder
    if mask is not None:
        mask = torch.cat([(torch.zeros([bs, self.code_len])==1).cuda(), mask], axis=1)
    outputs = self.encoder(src=encoder_input, src_key_padding_mask=mask)  
    z_encoded = outputs[0:self.code_len]

    _, _, one_hot, _ = self.vq_vae(self.down(z_encoded))  # [seqlen, bs, one-hot] 
    labels = torch.argmax(one_hot, dim=2).transpose(0,1)  # [bs, seqlen]
    if return_np:
      return labels.detach().cpu().numpy().astype(int)
    else:
      return labels


class PARAMEncoder(nn.Module):

  def __init__(self,
               config,
               quantization_bits,
               num_code = 100,
               code_len = 4,
               max_len = 80,
               ):
    """Initializes Encoder Model.

    Args:
    """
    super(PARAMEncoder, self).__init__()
    self.embed_dim = config['embed_dim']
    self.bits = quantization_bits
    self.dropout = config['dropout_rate']
    self.max_len = max_len

    commitment_cost = 0.25
    decay = 0.99
    self.codebook_dim = 128
    self.vq_vae = VectorQuantizerEMA(num_code, self.codebook_dim, commitment_cost, decay)

    self.code_len = code_len
    self.const_embed = Embedder(self.code_len, self.embed_dim)

    self.coord_embed_x = Embedder(2**self.bits+COORD_PAD+EXTRA_PAD, self.embed_dim)
    self.coord_embed_y = Embedder(2**self.bits+COORD_PAD+EXTRA_PAD, self.embed_dim)

    self.pixel_embed = Embedder(2**self.bits * 2**self.bits+PIX_PAD+EXTRA_PAD, self.embed_dim)
    self.pos_embed = PositionalEncoding(max_len=max_len+code_len, d_model=self.embed_dim)
   
    # Transformer encoder
    encoder_layers = TransformerEncoderLayerImproved(d_model=self.embed_dim, nhead=config['num_heads'], 
                                             dim_feedforward=config['hidden_dim'], dropout=self.dropout)
    encoder_norm = LayerNorm(self.embed_dim)
    self.encoder = TransformerEncoder(encoder_layers, config['num_layers'], encoder_norm)
    self.down = nn.Linear(self.embed_dim, self.codebook_dim)
    self.up = nn.Linear(self.codebook_dim, self.embed_dim)
    

  def forward(self, pixel_v, xy_v, mask, epoch):
    """ forward pass """
    bs, seqlen = pixel_v.shape[0], pixel_v.shape[1]

    # embedding 
    coord_embed = self.coord_embed_x(xy_v[...,0]) + self.coord_embed_y(xy_v[...,1]) # [bs, vlen, dim]
    pixel_embed = self.pixel_embed(pixel_v)
    embeddings = (coord_embed+pixel_embed).transpose(0,1) 
    z_embed = self.const_embed(torch.arange(0, self.code_len).long().cuda()).unsqueeze(1).repeat(1, bs, 1) 
    embed_input = torch.cat([z_embed, embeddings], dim=0)
    encoder_input = self.pos_embed(embed_input) 

    # Pass through encoder
    if mask is not None:
        mask = torch.cat([(torch.zeros([bs, self.code_len])==1).cuda(), mask], axis=1)
    outputs = self.encoder(src=encoder_input, src_key_padding_mask=mask)  # [seq_len, bs, dim].transpose(1,0)
    z_encoded = outputs[0:self.code_len]

    if epoch < INITIAL_PASS:
      vq_loss = 0.0
      selection = None 
      quantized_up = self.up(self.down(z_encoded)).transpose(0,1)
    else:
      vq_loss, quantized, _, selection = self.vq_vae(self.down(z_encoded))
      quantized_up = self.up(quantized).transpose(0,1)
    return quantized_up, vq_loss, selection 


  def get_code(self, pixel_v, xy_v, mask, return_np=True):
    bs, seqlen = pixel_v.shape[0], pixel_v.shape[1]

    # embedding 
    coord_embed = self.coord_embed_x(xy_v[...,0]) + self.coord_embed_y(xy_v[...,1]) # [bs, vlen, dim]
    pixel_embed = self.pixel_embed(pixel_v)
    embeddings = (coord_embed+pixel_embed).transpose(0,1)
    z_embed = self.const_embed(torch.arange(0, self.code_len).long().cuda()).unsqueeze(1).repeat(1, bs, 1) 
    embed_input = torch.cat([z_embed, embeddings], dim=0)
    encoder_input = self.pos_embed(embed_input) 

    # Pass through encoder
    if mask is not None:
        mask = torch.cat([(torch.zeros([bs, self.code_len])==1).cuda(), mask], axis=1)
    outputs = self.encoder(src=encoder_input, src_key_padding_mask=mask)  # [seq_len, bs, dim].transpose(1,0)
    
    z_encoded = outputs[0:self.code_len]
    _, _, one_hot, _ = self.vq_vae(self.down(z_encoded))  # [seqlen, bs, one-hot] 
    labels = torch.argmax(one_hot, dim=2).transpose(0,1)  # [bs, seqlen]
    if return_np:
      return labels.detach().cpu().numpy().astype(int)
    else:
      return labels


class EXTEncoder(nn.Module):

  def __init__(self,
               config,
               quantization_bits,
               num_code = 100,
               code_len = 4,
               max_len = 80,
               ):
    """Initializes Encoder Model.

    Args:
    """
    super(EXTEncoder, self).__init__()
    self.embed_dim = config['embed_dim']
    self.bits = quantization_bits
    self.dropout = config['dropout_rate']
    self.max_len = max_len

    commitment_cost = 0.25
    decay = 0.99
    self.codebook_dim = 128
    self.vq_vae = VectorQuantizerEMA(num_code, self.codebook_dim, commitment_cost, decay)

    self.code_len = code_len
    self.const_embed = Embedder(self.code_len, self.embed_dim)

    self.ext_embed = Embedder(2**self.bits+EXT_PAD+EXTRA_PAD, self.embed_dim)
    self.flag_embed = Embedder(8, self.embed_dim)
    self.pos_embed = PositionalEncoding(max_len=self.max_len+self.code_len, d_model=self.embed_dim)
   
    # Transformer encoder
    encoder_layers = TransformerEncoderLayerImproved(d_model=self.embed_dim, nhead=config['num_heads'], 
                                             dim_feedforward=config['hidden_dim'], dropout=self.dropout)
    encoder_norm = LayerNorm(self.embed_dim)
    self.encoder = TransformerEncoder(encoder_layers, config['num_layers'], encoder_norm)
    self.down = nn.Linear(256, self.codebook_dim)
    self.up = nn.Linear(self.codebook_dim, 256)
   

  def forward(self, ext_seq, flag_seq, mask, epoch):
    """ forward pass """
    bs, seqlen = ext_seq.shape[0], ext_seq.shape[1]

    # embedding 
    ext_embeds = self.ext_embed(ext_seq)
    flag_embeds = self.flag_embed(flag_seq)
    embeddings = (ext_embeds+flag_embeds).transpose(0,1) #ext_embeds.transpose(0,1) #
    z_embed = self.const_embed(torch.arange(0, self.code_len).long().cuda()).unsqueeze(1).repeat(1, bs, 1) 
    embed_input = torch.cat([z_embed, embeddings], dim=0)
    encoder_input = self.pos_embed(embed_input) 

    # Pass through encoder
    if mask is not None:
        mask = torch.cat([(torch.zeros([bs, self.code_len])==1).cuda(), mask], axis=1)
    outputs = self.encoder(src=encoder_input, src_key_padding_mask=mask)  # [seq_len, bs, dim].transpose(1,0)
    z_encoded = outputs[0:self.code_len]
    
    if epoch < INITIAL_PASS:
      vq_loss = 0.0
      selection = None 
      quantized_up = self.up(self.down(z_encoded)).transpose(0,1)
    else:
      vq_loss, quantized, _, selection = self.vq_vae(self.down(z_encoded))
      quantized_up = self.up(quantized).transpose(0,1)

    return quantized_up, vq_loss, selection 


  def get_code(self, ext_seq, flag_seq, mask, return_np=True):
    """ forward pass """
    bs, seqlen = ext_seq.shape[0], ext_seq.shape[1]

    # embedding 
    ext_embeds = self.ext_embed(ext_seq)
    flag_embeds = self.flag_embed(flag_seq)
    embeddings = (ext_embeds+flag_embeds).transpose(0,1)
    z_embed = self.const_embed(torch.arange(0, self.code_len).long().cuda()).unsqueeze(1).repeat(1, bs, 1) 
    embed_input = torch.cat([z_embed, embeddings], dim=0)
    encoder_input = self.pos_embed(embed_input) 

    # Pass through encoder
    if mask is not None:
        mask = torch.cat([(torch.zeros([bs, self.code_len])==1).cuda(), mask], axis=1)
    outputs = self.encoder(src=encoder_input, src_key_padding_mask=mask)  # [seq_len, bs, dim].transpose(1,0)
    
    z_encoded = outputs[0:self.code_len]
    _, _, one_hot, _ = self.vq_vae(self.down(z_encoded))  # [seqlen, bs, one-hot] 
    labels = torch.argmax(one_hot, dim=2).transpose(0,1)  # [bs, seqlen]
    if return_np:
      return labels.detach().cpu().numpy().astype(int)
    else:
      return labels

