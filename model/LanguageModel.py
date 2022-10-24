import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from fast_transformers.transformers import *
import numpy as np

from jukebox.transformer.ops import Conv1D
from model.autoregressive import ConditionalAutoregressive2D

def nucleus(probs, p=0.5):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    probs /= (sum(probs) + 1e-5)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word

def make_local_mask(seq_length, width, device):
    mask = torch.zeros(seq_length, seq_length)
    for i in range(seq_length):
        for j in range(seq_length):
            if j < i + width and j > i - width:
                mask[i][j] = 1
    return mask.bool().to(device)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)

from jukebox.hparams import setup_hparams
from jukebox.transformer.ops import Conv1D, LayerNorm


class JukeTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.make_juke_prior(args)
        self.use_tokens = args.name != 'lm9' and args.name != 'lm10'
        print(f'use tokens:{self.use_tokens}')
        self.prime_state_proj = Conv1D(args.d_model, args.d_model)
        self.prime_state_ln = LayerNorm(args.d_model)
        self.binfo_type = args.binfo_type
        if self.binfo_type == 'low':
            self.binfo_state_proj = Conv1D(50, args.d_model)
        elif self.binfo_type == 'mid':
            self.onset_emb = nn.Embedding(2, args.d_model)
        elif self.binfo_type == 'high':
            self.beat_emb = nn.Embedding(3, args.d_model)
        elif self.binfo_type is None:
            pass
        else:
            raise RuntimeError('No matchedbeat information type')

    def get_prime_loss(self, encoder_kv, prime_t):
        if self.use_tokens:
            encoder_kv = encoder_kv.float()
            encoder_kv = self.prime_x_out(encoder_kv)
            prime_loss = nn.functional.cross_entropy(encoder_kv.view(-1, 512), prime_t.view(-1)) / np.log(2.)
        else:
            prime_loss = torch.tensor(0.0, device='cuda')
        return prime_loss

    def get_encoder_kv(self, prime, binfo, fp16=False):
        if self.use_tokens:
            N = prime.shape[0]
            prime_acts = self.prime_prior(prime, binfo, None, None, fp16=fp16, isroll=False)
            assert prime_acts.dtype == torch.float, f'Expected torch.float, got {prime_acts.dtype}'
            encoder_kv = self.prime_state_ln(self.prime_state_proj(prime_acts))
            assert encoder_kv.dtype == torch.float, f'Expected torch.float, got {encoder_kv.dtype}'
        else:
            encoder_kv = None
        return encoder_kv
    
    def binfo_conditioner(self, binfo):
        if self.binfo_type == 'low':
            binfo = F.interpolate(binfo.unsqueeze(1), size=(self.prior.encoder_dims, binfo.size(-1))).squeeze(1)
            binfo = self.binfo_state_proj(binfo)
        elif self.binfo_type == 'mid':
            binfo = self.onset_emb(binfo.long())
        elif self.binfo_type == 'high':
            binfo = binfo.double()
            binfo = torch.where(binfo > 1, 2., binfo)
            binfo = self.beat_emb(binfo.long())
        elif self.binfo_type is None:
            binfo = None
        return binfo

    def forward(self, tgz, otz, binfo=None):
        binfo = self.binfo_conditioner(binfo)
        encoder_kv = self.get_encoder_kv(otz, binfo)
        loss, pred = self.prior(tgz, x_cond=binfo, y_cond=None, encoder_kv=encoder_kv, fp16=False, loss_full=False,
                    encode=False, get_preds=True, get_acts=False, get_sep_loss=False)
        return loss, pred, 
    
    def sample(self, n_samples, otz, binfo, vqvae, temp=1.0, top_k=0, top_p=0.0):
        self.eval()
        with torch.no_grad():
            binfo = self.binfo_conditioner(binfo)
            encoder_kv = self.get_encoder_kv(otz, binfo)
            pred = self.prior.sample(n_samples, x_cond=binfo, y_cond=None,
                encoder_kv=encoder_kv, fp16=False, temp=temp, top_k=top_k, top_p=top_p,
                get_preds=False, sample_tokens=None, device=otz.device
            )
            pred = vqvae.decode(pred)
        return pred
    
    def make_juke_prior(self, args):
        sequence_length = 4096 // np.prod(args.upsample_ratios)
        hps = setup_hparams('small_sep_enc_dec_prior', dict())
        hps['prior_depth'] = args.enc_layers
        hps['n_ctx'] = sequence_length
        hps['blocks'] = args.blocks
        hps['prior_width'] = args.d_model
        hps['attn_dropout'] = 0.3
        hps['resid_dropout']= 0.3 
        hps['emb_dropout'] = args.dropout
        hps['m_mlp'] = 1
        hps['heads'] = args.heads
        hps['attn_order'] = 8 if args.name != 'lm9' and args.name != 'lm10' else 2
        hps['prime_width']= args.d_model
        hps['prime_depth']=9
        hps['prime_heads']=2
        hps['prime_attn_order']=2
        hps['prime_blocks']= args.blocks
        hps['n_vocab'] = 1024
        hps['prime_attn_dropout'] = hps['attn_dropout']
        hps['prime_resid_dropout']= hps['resid_dropout'] 
        hps['prime_emb_dropout'] = 0
        hps['c_res'] = 0
        prior_kwargs = dict(input_shape=(hps.n_ctx,), bins=args.codebook_size,
                                width=hps.prior_width, depth=hps.prior_depth, heads=hps.heads,
                                attn_order=hps.attn_order, blocks=hps.blocks, spread=hps.spread,
                                attn_dropout=hps.attn_dropout, resid_dropout=hps.resid_dropout, emb_dropout=hps.emb_dropout,
                                zero_out=hps.zero_out, res_scale=hps.res_scale, pos_init=hps.pos_init,
                                init_scale=hps.init_scale,
                                m_attn=hps.m_attn, m_mlp=hps.m_mlp,
                                checkpoint_res=hps.c_res if hps.train else 0, checkpoint_attn=hps.c_attn if hps.train else 0, checkpoint_mlp=hps.c_mlp if hps.train else 0)
        
        prime_kwargs = dict(input_shape=(hps.n_ctx,), bins=hps.n_vocab,
                                width=hps.prime_width, depth=hps.prime_depth, heads=hps.prime_heads,
                                attn_order=hps.prime_attn_order, blocks=hps.prime_blocks, spread=hps.prime_spread,
                                attn_dropout=hps.prime_attn_dropout, resid_dropout=hps.prime_resid_dropout,
                                emb_dropout=hps.prime_emb_dropout,
                                zero_out=hps.prime_zero_out, res_scale=hps.prime_res_scale,
                                pos_init=hps.prime_pos_init, init_scale=hps.prime_init_scale,
                                m_attn=hps.prime_m_attn, m_mlp=hps.prime_m_mlp,
                                checkpoint_res=hps.prime_c_res if hps.train else 0, checkpoint_attn=hps.prime_c_attn if hps.train else 0,
                                checkpoint_mlp=hps.prime_c_mlp if hps.train else 0)
        
        self.hps = hps
        self.prior = ConditionalAutoregressive2D(x_cond=args.binfo_type is not None, y_cond=False, encoder_dims = sequence_length, 
                                                    pos_emb=None, **prior_kwargs)
        self.prime_prior = ConditionalAutoregressive2D(x_cond=args.binfo_type is not None, y_cond=False, only_encode=True, 
                                                    mask=False, pos_emb=None, **prime_kwargs)


