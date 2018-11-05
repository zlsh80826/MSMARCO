data_config = {
    'word_size'            : 16,
    'word_count_threshold' : 10,
    'char_count_threshold' : 50,
    'pickle_file'          : '../data/vocabs.pkl',
    'elmo_embedding'       : '../data/elmo_embedding.bin',
    'glove_embedding'      : '../data/glove.840B.300d.txt'
}

model_config = {
    'elmo_dim'          : 1024,
    'hidden_dim'     	: 300,
    'char_convs'     	: 300,
    'char_emb_dim'   	: 16,
    'dropout'        	: 0.25,
    'highway_layers' 	: 2,
    'two_step'          : True,
    'use_cudnn'         : True,
}

training_config = {
    'minibatch_size'    : 1800,     # in samples when using ctf reader, per worker
    'epoch_size'        : 473093,   # in sequences, when using ctf reader
    'log_freq'          : 10000,    # in minibatchs
    'max_epochs'        : 20,
    'lr'                : 1,
    'train_data'        : 'train.ctf',  # or 'train.tsv'
    'val_data'          : 'dev.ctf',
    'val_interval'      : 1,       # interval in epochs to run validation
    'stop_after'        : 1,       # num epochs to stop if no CV improvement
    'minibatch_seqs'    : 16,      # num sequences of minibatch, when using tsv reader, per worker
    'distributed_after' : 0,       # num sequences after which to start distributed training
}
