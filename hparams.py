HPARAMS_REGISTRY = {}
MODEL_LIST = {
    1: ('vq1', 'lm1'),
    2: ('vq1', 'lm2'),
    3: ('vq1', 'lm3'),
    4: ('vq1', 'lm4'),
    5: ('vq1', 'lm5'),
    6: ('vq1', 'lm6'),
    7: ('vq1', 'lm7'),
    8: ('vq1', 'lm8'),
    9: ('vq2', 'lm1'),
    10: ('vq3', 'lm1'),
    11: ('vq1', 'lm9'),
    12: ('vq1', 'lm10'),
    21: ('vq4', 'lm1'),
    22: ('vq4', 'lm9')
}
class Hyperparams(dict):
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

LM_DEFAULTS = Hyperparams(
    batch_size=16,
    sample_step=10,
    wandb=True,
    ckpt_dir='',
    path='',
)

VQ_DEFAULTS = Hyperparams(
    batch_size=32,
    sample_step=50,
    wandb=True,
    ckpt_dir='',
    path='',
)

def setup_lm_hparams(hparam_set_names):
    # hparam_set_names = (vqvae_hparam, lm_hparam)
    H = Hyperparams()
    vqvae_sets = HPARAMS_REGISTRY[hparam_set_names[0].strip()]
    lm_sets = HPARAMS_REGISTRY[hparam_set_names[1].strip()]
    H.update(LM_DEFAULTS)    
    H.update(vqvae_sets)
    H['vq_name'] = H['name']
    H.update(lm_sets)

    return H

def setup_vq_hparams(hparam_set_name):
    # hparam_set_name = vqvae_hparam
    print(f'set up {hparam_set_name} hparam')
    H = Hyperparams()
    vqvae_sets = HPARAMS_REGISTRY[hparam_set_name.strip()]
    H.update(VQ_DEFAULTS) 
    H.update(vqvae_sets)

    return H


OPT = Hyperparams(
    epochs=1000,
    lr=0.0003,
    clip=1.0,
    beta1=0.9,
    beta2=0.999,
    ignore_grad_norm=0,
    weight_decay=0.0,
    eps=1e-08,
    lr_warmup=100.0,
    lr_decay=10000000.0,
    lr_gamma=1.0,
    lr_scale=1.0,
    lr_use_linear_decay=False,
    lr_start_linear_decay=0,
    lr_use_cosine_decay=False,
    fp16_opt = False,
    prior = True,
    restore_prior = '',
    fp16 = False,
)

MEL = Hyperparams(
    n_fft = 1024,
    hop_length = 256,
    win_length = 1024,
    sampling_rate = 44100,
    n_mel_channels = 80,
    extension = '.wav',
    mel_fmin=0.0,
    mel_fmax=None,
)

vq1 = Hyperparams(
    # codebook 32 | 1024 
    name = 'vq1',
    codebook_size = 32,
    upsample_ratios = [2, 2],
    downsample_ratios = [0.5, 0.5],
    commit_beta = 0.02,
)
HPARAMS_REGISTRY['vq1'] = vq1

vq2 = Hyperparams(
    # codebook 64 | 512
    name = 'vq2',
    codebook_size = 64,
    upsample_ratios = [4, 2],
    downsample_ratios = [0.5, 0.25],
    commit_beta = 0.02,
)
HPARAMS_REGISTRY['vq2'] = vq2

vq3 = Hyperparams(
    # codebook 32 | 2048 
    name = 'vq3',
    batch_size=6,
    codebook_size = 32,
    upsample_ratios = [2, 1],
    downsample_ratios = [1, .5],
    commit_beta = 0.02,
)
HPARAMS_REGISTRY['vq3'] = vq3

vq4 = Hyperparams(
    # codebook 32 | 1024, indentical to vq1 but differnet dataset 
    name = 'vq4',
    codebook_size = 32,
    upsample_ratios = [2, 2],
    downsample_ratios = [0.5, 0.5],
    commit_beta = 0.02,
)
HPARAMS_REGISTRY['vq4'] = vq4

lm1 = Hyperparams(
    # Raw beat activation w/o in-attention
    name = 'lm1',
    enc_layers = 20,
    dec_layers= 9,
    d_model= 512,
    dropout = 0.2,
    heads = 2,
    blocks = 16,

    bact_type = 'raw'
)
HPARAMS_REGISTRY['lm1'] = lm1

lm2 = Hyperparams(
    # No beat activation w/o in-attention
    name = 'lm2',
    enc_layers = 20,
    dec_layers= 9,
    d_model= 512,
    dropout = 0.2,
    heads = 2,
    blocks = 16,

    bact_type = None,
)
HPARAMS_REGISTRY['lm2'] = lm2

lm3 = Hyperparams(
    # Raw beat activation w/ in-attention
    name = 'lm3',
    enc_layers = 20,
    dec_layers= 9,
    d_model= 512,
    dropout = 0.2,
    heads = 2,
    blocks = 16,

    bact_type = 'raw',
)
HPARAMS_REGISTRY['lm3'] = lm3

lm4 = Hyperparams(
    # lm1 but 1024
    name = 'lm4',
    batch_size = 8,
    enc_layers = 20,
    dec_layers= 9,
    d_model= 1024,
    dropout = 0.2,
    heads = 2,
    blocks = 16,

    bact_type = 'raw',
)
HPARAMS_REGISTRY['lm4'] = lm4

lm5 = Hyperparams(
    # beat / downbeat / non-beat 3 token beat embed w/o in-attention
    name = 'lm5',
    enc_layers = 20,
    dec_layers= 9,
    d_model= 512,
    dropout = 0.2,
    heads = 2,
    blocks = 16,

    bact_type = 'token',
)
HPARAMS_REGISTRY['lm5'] = lm5

lm6 = Hyperparams(
    # beat / downbeat / non-beat 3 token beat embed w/ in-attention
    name = 'lm6',
    enc_layers = 20,
    dec_layers= 9,
    d_model= 512,
    dropout = 0.2,
    heads = 2,
    blocks = 16,

    bact_type = 'token',
)
HPARAMS_REGISTRY['lm6'] = lm6

lm7 = Hyperparams(
    # onset/ non-onset 2 tokens beat embed w/o in-attention
    name = 'lm7',
    enc_layers = 20,
    dec_layers= 9,
    d_model= 512,
    dropout = 0.2,
    heads = 2,
    blocks = 16,

    bact_type = 'onset',
)
HPARAMS_REGISTRY['lm7'] = lm7

lm8 = Hyperparams(
    # onset/ non-onset 2 tokens beat embed w/ in-attention
    name = 'lm8',
    enc_layers = 20,
    dec_layers= 9,
    d_model= 512,
    dropout = 0.2,
    heads = 2,
    blocks = 16,

    bact_type = 'onset',
)
HPARAMS_REGISTRY['lm8'] = lm8

lm9 = Hyperparams(
    # No encoder w/ raw beat info
    name = 'lm9',
    enc_layers = 20,
    dec_layers= 9,
    d_model= 512,
    dropout = 0.2,
    heads = 2,
    blocks = 16,

    bact_type = 'raw',
)
HPARAMS_REGISTRY['lm9'] = lm9

lm10 = Hyperparams(
    # No encoder w/o beat info
    name = 'lm10',
    enc_layers = 20,
    dec_layers= 9,
    d_model= 512,
    dropout = 0.2,
    heads = 2,
    blocks = 16,

    bact_type = None,
)
HPARAMS_REGISTRY['lm10'] = lm10



    
