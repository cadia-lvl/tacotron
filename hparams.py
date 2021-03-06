
from tensorflow.contrib.training import HParams

from text.characters import chars

# Default hyperparameters:
hparams = HParams(
    # Comma-separated list of cleaners to run on text prior to training and eval. For non-English
    # text, you may want to use "basic_cleaners" or "transliteration_cleaners"
    cleaners='basic_cleaners',

    # Audio:
    num_mels = 80,
    num_freq = 1025, # note: (num_freq - 1) * 2 = n_fft
    n_fft = 2048, # the number of points in the Fourier transformation
    sample_rate = 24000,
    frame_length_ms = 50,
    frame_shift_ms = 12.5,
    preemphasis = 0.97,
    min_level_db = -100,
    ref_level_db = 20,


    # Model:
    embedded_depth = 256,
    outputs_per_step = 5,
    # -- encoder cbhg
    prenet_depths = [256, 128],
    encoder_K = 16,
    encoder_bank_num_filters = 128,
    encoder_pooling_stride = 1,
    encoder_pooling_width = 2,
    encoder_proj_num_filters = [128, 128],
    encoder_proj_filter_width = 3,
    encoder_num_highway_layers = 4,
    encoder_highway_depth = 128,
    encoder_gru_num_cells = 128,
    # -- decoder
    decoder_K = 8,
    decoder_bank_num_filters = 128,
    decoder_pooling_stride = 1,
    decoder_pooling_width = 2,
    decoder_proj_num_filters = [256, 80],
    decoder_proj_filter_width = 3,
    decoder_num_highway_layers = 4,
    decoder_highway_depth = 128,
    decoder_gru_num_cells = 128,
    
    postnet_depth = 256,
    attention_depth = 256,
    decoder_depth = 256,

    # More model related:
    pad_value = 0,

    # Training:
    batch_size = 32,
    superbatch_size = 32,
    adam_beta1 = 0.9,
    adam_beta2 = 0.999,
    initial_learning_rate = 0.002,
    decay_learning_rate = True,

    # Eval:
    max_iters = 200,
    griffin_lim_iters = 60,
    power = 1.5, # Power to raise magnitudes to prior to Griffin-Lim
)
