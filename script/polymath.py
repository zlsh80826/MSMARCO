import cntk as C
import numpy as np
from helpers import *
import pickle
import importlib
import os
import math
import h5py


class PolyMath:
    def __init__(self, config_file):
        data_config = importlib.import_module(config_file).data_config
        model_config = importlib.import_module(config_file).model_config

        self.word_count_threshold = data_config['word_count_threshold']
        self.char_count_threshold = data_config['char_count_threshold']
        self.word_size = data_config['word_size']
        self.abs_path = os.path.dirname(os.path.abspath(__file__))
        pickle_file = os.path.join(self.abs_path, data_config['pickle_file'])

        with open(pickle_file, 'rb') as vf:
            known, self.vocab, self.chars = pickle.load(vf)

        self.wg_dim = known
        self.wn_dim = len(self.vocab) - known
        self.c_dim = len(self.chars)
        self.a_dim = 1

        self.hidden_dim = model_config['hidden_dim']
        self.convs = model_config['char_convs']
        self.dropout = model_config['dropout']
        self.char_emb_dim = model_config['char_emb_dim']
        self.highway_layers = model_config['highway_layers']
        self.two_step = model_config['two_step']
        self.use_cudnn = model_config['use_cudnn']
        self.use_sparse = True

        print('dropout', self.dropout)
        print('use_cudnn', self.use_cudnn)
        print('use_sparse', self.use_sparse)

    def charcnn(self, x):
        embedding = C.layers.Embedding(self.char_emb_dim)
        dropout = C.layers.Dropout(self.dropout),
        conv2d = C.layers.Convolution2D((5, self.char_emb_dim), 
                self.convs, 
                activation=C.relu, 
                init=C.glorot_uniform(), 
                bias=True, 
                init_bias=0, 
                name='charcnn_conv')
        conv_out = C.layers.Sequential([
            embedding,
            dropout,
            conv2d])(x)
        return C.reduce_max(conv_out, axis=1)

    def embed(self):
        npglove = np.zeros((self.wg_dim, 1024 + 300), dtype=np.float32)
        hf = h5py.File(os.path.join(self.abs_path, '../data/elmo_embedding.bin'), 'r')
        #f5_exception = dict()

        ## add vocab_use map
        #vocabs_use = dict()
        #for v in self.vocab:
        #    vocabs_use[v] = False

        #with open(os.path.join(self.abs_path, '../dataset/elmo_out_of_glove.txt'), encoding='utf-8') as f:
        #    for line in f:
        #        parts = line.split()
        #        f5_exception[parts[0]] = np.asarray([float(p) for p in parts[-1024:]])

        with open(os.path.join(self.abs_path, '../data/glove.840B.300d.txt'), encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                word = parts[0].lower()
                if word in self.vocab:
                    try:
                        if len(parts) == 301:
                            npglove[self.vocab[word],:300] = np.asarray([float(p) for p in parts[-300:]])
                            npglove[self.vocab[word],300:] = np.average(hf[word][:], axis=0)
#                            vocabs_use[word] = True
                    except:
                        npglove[self.vocab[word],300:] = np.average(hf['<UNK>'][:], axis=0)
                        print(word)

        #for v in vocabs_use:
        #    if vocabs_use[v] == False:
        #        try:
        #            print(v)
        #            npglove[self.vocab[word], 300:] = f5_exception[v]
        #        except:
        #            print('except: ', v)

        glove = C.constant(npglove)
        nonglove = C.parameter(shape=(self.wn_dim, 1024 + 300), init=C.glorot_uniform(), name='TrainableE')
        
        def func(wg, wn):
            return C.times(wg, glove) + C.times(wn, nonglove)

        return func

    def input_layer(self,cgw,cc,qgw,qc,qnw,cnw):
        cgw_ph = C.placeholder()
        cnw_ph = C.placeholder()
        cc_ph  = C.placeholder()
        qgw_ph = C.placeholder()
        qnw_ph = C.placeholder()
        qc_ph  = C.placeholder()

        input_chars = C.placeholder(shape=(1,self.word_size,self.c_dim))
        input_glove_words = C.placeholder(shape=(self.wg_dim,))
        input_nonglove_words = C.placeholder(shape=(self.wn_dim,))

        embedded = C.splice(
            C.reshape(self.charcnn(input_chars), self.convs),
            self.embed()(input_glove_words, input_nonglove_words), name='splice_embed')

        highway = HighwayNetwork(dim=1024 + 600, highway_layers=self.highway_layers)(embedded)
        highway_drop = C.layers.Dropout(self.dropout)(highway)
        processed = OptimizedRnnStack(self.hidden_dim,
             num_layers=1,
             bidirectional=True,
             use_cudnn=self.use_cudnn,
             name='input_rnn')(highway_drop)
        
        qce = C.one_hot(qc_ph, num_classes=self.c_dim, sparse_output=self.use_sparse)
        cce = C.one_hot(cc_ph, num_classes=self.c_dim, sparse_output=self.use_sparse)
                
        q_processed = processed.clone(C.CloneMethod.share, 
            {input_chars:qce, input_glove_words:qgw_ph, input_nonglove_words:qnw_ph})
        c_processed = processed.clone(C.CloneMethod.share, 
            {input_chars:cce, input_glove_words:cgw_ph, input_nonglove_words:cnw_ph})

        return C.as_block(
            C.combine([c_processed, q_processed]),
            [(cgw_ph, cgw), (cc_ph, cc), (qgw_ph, qgw), (qc_ph, qc), (qnw_ph, qnw), (cnw_ph, cnw)],
            'input_layer',
            'input_layer')

    def scale_dot_product_attention_block(self, contextQ, contextV, contextK, name):

        Q = C.placeholder(shape=(2*self.hidden_dim,), dynamic_axes=[self.b_axis, self.q_axis])
        V = C.placeholder(shape=(2*self.hidden_dim,), dynamic_axes=[self.b_axis, self.q_axis])
        K = C.placeholder(shape=(2*self.hidden_dim,), dynamic_axes=[self.b_axis, self.q_axis])

        Ql = C.layers.Dense(100)(Q)
        Vl = C.layers.Dense(100)(V)
        Kl = C.layers.Dense(100)(K)

        kvw, kvw_mask = C.sequence.unpack(Kl, padding_value=0).outputs
        vvw, _ = C.sequence.unpack(Vl, padding_value=0).outputs
        KT = C.swapaxes(kvw)

        S = C.reshape(C.times(Ql, KT)/math.sqrt(100), -1) 
        kvw_mask_expanded = C.sequence.broadcast_as(kvw_mask, Ql)
        S = C.softmax(C.element_select(kvw_mask_expanded, S, C.constant(-1e+30)))
        att = C.times(S, vvw)

        return C.as_block(
            att,
            [(Q, contextQ), (V, contextV), (K, contextK)],
            'sdp_attention_block' + name,
            'sdp_attention_block' + name)

    def multi_head_attention(self, contextQ, contextV, contextK, name):
        Q = C.placeholder(shape=(2*self.hidden_dim,), dynamic_axes=[self.b_axis, self.q_axis])
        V = C.placeholder(shape=(2*self.hidden_dim,), dynamic_axes=[self.b_axis, self.q_axis])
        K = C.placeholder(shape=(2*self.hidden_dim,), dynamic_axes=[self.b_axis, self.q_axis])

        att0 = self.scale_dot_product_attention_block(Q, V, K, '0')
        att1 = self.scale_dot_product_attention_block(Q, V, K, '1')
        att2 = self.scale_dot_product_attention_block(Q, V, K, '2')
        att3 = self.scale_dot_product_attention_block(Q, V, K, '3')
        att4 = self.scale_dot_product_attention_block(Q, V, K, '4')
        att5 = self.scale_dot_product_attention_block(Q, V, K, '5')

        att = C.splice(att0, att1, att2, att3, att4, att5)
        att_residual = att + Q

        return C.as_block(
            att_residual,
            [(Q, contextQ), (V, contextV), (K, contextK)],
            'multi_head_attention_layer' + name,
            'multi_head_attention_layer' + name)


    def attention_layer(self, context, query, layer):

        q_processed = C.placeholder(shape=(2*self.hidden_dim,))
        p_processed = C.placeholder(shape=(2*self.hidden_dim,))

        qvw, qvw_mask = C.sequence.unpack(q_processed, padding_value=0).outputs

        wq = C.parameter(shape=(2*self.hidden_dim, 2*self.hidden_dim), init=C.glorot_uniform())
        wp = C.parameter(shape=(2*self.hidden_dim, 2*self.hidden_dim), init=C.glorot_uniform())
        wg = C.parameter(shape=(8*self.hidden_dim, 8*self.hidden_dim), init=C.glorot_uniform())
        v = C.parameter(shape=(2*self.hidden_dim, 1), init=C.glorot_uniform())

        # seq[tensor[2d]] p_len x 2d
        wpt = C.reshape(C.times(p_processed, wp), (-1, 2*self.hidden_dim))

        # q_len x 2d
        wqt = C.reshape(C.times(qvw, wq), (-1, 2*self.hidden_dim))
        
        # seq[tensor[q_len]]
        S = C.reshape(C.times(C.tanh(C.sequence.broadcast_as(wqt, p_processed) + wpt), v), (-1))

        qvw_mask_expanded = C.sequence.broadcast_as(qvw_mask, p_processed)

        # seq[tensor[q_len]]
        S = C.element_select(qvw_mask_expanded, S, C.constant(-1e+30))
        
        # seq[tensor[q_len]]
        A = C.softmax(S, axis=0)

        # seq[tensor[2d]]
        swap_qvw = C.swapaxes(qvw)
        cq = C.reshape(C.reduce_sum(A * C.sequence.broadcast_as(swap_qvw, A), axis=1), (-1))

        # seq[tensor[4d]]
        uc_concat = C.splice(p_processed, cq, p_processed * cq, cq * cq)
        
        # seq[tensor[4d]]
        gt = C.tanh(C.times(uc_concat, wg))
        
        # seq[tensor[4d]]
        uc_concat_star = gt * uc_concat
 
        # seq[tensor[4d]]
        vp = C.layers.Sequential([
            C.layers.Dropout(self.dropout),
            OptimizedRnnStack(self.hidden_dim, bidirectional=True, 
                use_cudnn=self.use_cudnn, name=layer+'_attention_rnn')])(uc_concat_star)
        
        return C.as_block(
            vp,
            [(p_processed, context), (q_processed, query)],
            'attention_layer',
            'attention_layer')

    def rnet_output_layer(self, attention_context, query):

        att_context = C.placeholder(shape=(2*self.hidden_dim,))
        q_processed = C.placeholder(shape=(2*self.hidden_dim,))

        wuq = C.parameter(shape=(2*self.hidden_dim, 2*self.hidden_dim), init=C.glorot_uniform())
        whp = C.parameter(shape=(2*self.hidden_dim, 2*self.hidden_dim), init=C.glorot_uniform())
        wha = C.parameter(shape=(2*self.hidden_dim, 2*self.hidden_dim), init=C.glorot_uniform())
        v = C.parameter(shape=(2*self.hidden_dim, 1), init=C.glorot_uniform())
        bias = C.parameter(shape=(2*self.hidden_dim), init=C.glorot_uniform())

        whp_end = C.parameter(shape=(2*self.hidden_dim, 2*self.hidden_dim), init=C.glorot_uniform())
        wha_end = C.parameter(shape=(2*self.hidden_dim, 2*self.hidden_dim), init=C.glorot_uniform())
        v_end = C.parameter(shape=(2*self.hidden_dim, 1), init=C.glorot_uniform())

        # sequence[tensor[1]] q_len x 1
        s0 = C.times(C.tanh(C.times(q_processed, wuq) + bias), v)
        a0 = C.sequence.softmax(s0)
        rQ = C.sequence.reduce_sum(a0 * q_processed)
        
        # sequence[tensor[1]] plen x 1 
        ts = C.reshape(C.times(C.tanh(
            C.times(att_context, whp) + C.times(C.sequence.broadcast_as(rQ, att_context), wha)), v), (-1))

        # sequence[tensor[1]]
        ta = C.sequence.softmax(ts)

        # sequence[2d] 1 x 2d
        c0 = C.reshape(C.sequence.reduce_sum(ta * att_context), (2*self.hidden_dim))
        
        # sequence[tensor[2d]]
        ha1 = C.layers.blocks.GRU(2*self.hidden_dim)(rQ, c0)

        # sequence[tensor[1]] plen x 1
        s1 = C.reshape(C.times(C.tanh(C.times(att_context, whp_end) + C.times(
            C.sequence.broadcast_as(ha1, att_context), wha_end)), v_end), (-1))

        # sequence[tensor[1]] plen x 1
        a1 = C.sequence.softmax(s1)

        return C.as_block(
            C.combine([ts, s1]),
            [(att_context, attention_context), (q_processed, query)],
            'output_layer',
            'output_layer')

    def model(self):
        c = C.Axis.new_unique_dynamic_axis('c')
        q = C.Axis.new_unique_dynamic_axis('q')
        b = C.Axis.default_batch_axis()
        self.c_axis = c
        self.q_axis = q
        self.b_axis = b
        cgw = C.input_variable(self.wg_dim, dynamic_axes=[b,c], is_sparse=self.use_sparse, name='cgw')
        cnw = C.input_variable(self.wn_dim, dynamic_axes=[b,c], is_sparse=self.use_sparse, name='cnw')
        qgw = C.input_variable(self.wg_dim, dynamic_axes=[b,q], is_sparse=self.use_sparse, name='qgw')
        qnw = C.input_variable(self.wn_dim, dynamic_axes=[b,q], is_sparse=self.use_sparse, name='qnw')
        cc = C.input_variable((1,self.word_size), dynamic_axes=[b,c], name='cc')
        qc = C.input_variable((1,self.word_size), dynamic_axes=[b,q], name='qc')
        ab = C.input_variable(self.a_dim, dynamic_axes=[b,c], name='ab')
        ae = C.input_variable(self.a_dim, dynamic_axes=[b,c], name='ae')

        #input layer
        c_processed, q_processed = self.input_layer(cgw,cc,qgw,qc,qnw,cnw).outputs

        # attention layer
        att_context_0 = self.attention_layer(c_processed, q_processed, 'attention0')
        att_context_1 = self.attention_layer(att_context_0, q_processed, 'attention1')
        att_context_2 = self.attention_layer(att_context_1, q_processed, 'attention2')

        # output layer
        start_logits, end_logits = self.rnet_output_layer(att_context_2, q_processed).outputs

        # loss
        loss = all_spans_loss(start_logits, ab, end_logits, ae)

        return C.combine([start_logits, end_logits]), loss
