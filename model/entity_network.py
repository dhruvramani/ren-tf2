"""
entity_network.py

Model definition class for the Recurrent Entity Network. Defines Input Encoder,
Dynamic Memory Cell, and Readout components.
"""
from collections import namedtuple
from tflearn.activations import sigmoid, softmax
import functools
import tensorflow as tf
import tflearn

PAD_ID = 0

def prelu_func(features, initializer=None, scope=None):
    """
    Implementation of [Parametric ReLU](https://arxiv.org/abs/1502.01852) borrowed from Keras.
    """
    with tf.variable_scope(scope, 'PReLU', initializer=initializer):
        alpha = tf.get_variable('alpha', features.get_shape().as_list()[1:])
        pos = tf.nn.relu(features)
        neg = alpha * (features - tf.abs(features)) * 0.5
        return pos + neg
prelu = functools.partial(prelu_func, initializer=tf.constant_initializer(1.0))


class EntityNetwork():
    def __init__(self, vocabulary, story_len, batch_size, memory_slots, embedding_size, mask_dim, labels_dim,
                 learning_rate, decay_steps, decay_rate, clip_gradients=40.0, 
                 initializer=tf.random_normal_initializer(stddev=0.1)):
        """
        Initialize an Entity Network with the necessary hyperparameters.

        :param vocabulary: Word Vocabulary for given model.
        :param story_len: Maximum length of a story.
        """
        self.vocab_sz, self.story_len, self.mask_dim = vocabulary, story_len, mask_dim
        self.embed_sz, self.memory_slots, self.init, self.labels_dim = embedding_size, memory_slots, initializer, labels_dim
        self.bsz, self.lr, self.decay_steps, self.decay_rate = batch_size, learning_rate, decay_steps, decay_rate
        self.clip_gradients = clip_gradients

        # Setup Placeholders
        self.S = tf.placeholder(tf.float32, [None, self.story_len, self.embed_sz], name="Story")
        #self.S_len = tf.placeholder(tf.int32, [None], name="Story_Length")
        self.labels = tf.placeholder(tf.float32, [None, self.labels_dim], name="Labels")
        self.mask = tf.placeholder(tf.int32, [None], name="Mask")

        self.labels_embedding = tf.placeholder(tf.float32, [1, self.labels_dim, self.embed_sz], name="LabelEmbeds")
        self.bias_adj = tf.placeholder(tf.float32, [1, self.labels_dim, self.labels_dim], name="AdjacencyBias") # for GAT
        self.adj_m = tf.placeholder(tf.float32, [self.labels_dim, self.labels_dim], name="Adjacency")

        # Setup Global, Epoch Step 
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        #self.ground_truth = tf.reshape(self.ground_truth, [-1, self.mask_dim * sekf.labels_dim])
        # Instantiate Network Weights
        self.instantiate_weights()

        # Build Inference Pipeline
        self.logits = self.inference()

        self.ground_truth = tf.gather(self.labels, self.mask, axis=0)
        self.logits = tf.gather(self.logits, self.mask, axis=0)


        # Build Loss Computation
        self.loss_val = self.loss()

        # Build Training Operation
        self.train_op = self.train()

        # Create operations for computing the accuracy
        #correct_prediction = tf.equal(tf.argmax(self.logits, 1), self.A)
        #self.f1, self.precision, self.recall =  self.accuracy()
    
    def instantiate_weights(self):
        """
        Instantiate Network Weights, including all weights for the Input Encoder, Dynamic
        Memory Cell, as well as Output Decoder.
        """
        # Create Embedding Matrix, with 0 Vector for PAD_ID (0)
        #E = tf.get_variable("Embedding", [self.vocab_sz, self.embed_sz], initializer=self.init)
        #zero_mask = tf.constant([0 if i == 0 else 1 for i in range(self.vocab_sz)], dtype=tf.float32, shape=[self.vocab_sz, 1])
        #self.E = E * zero_mask
        
        # Create Learnable Mask
        # sentence_len == no. of words
        self.story_mask = tf.get_variable("Story_Mask", [self.sentence_len, 1], initializer=self.init) #tf.constant_initializer(1.0))
        #self.query_mask = tf.get_variable("Query_Mask", [self.sentence_len, 1], initializer=tf.constant_initializer(1.0))

        # Create Memory Cell Keys [IF DESIRED - TIE KEYS HERE]
        self.keys = [tf.get_variable("Key_%d" % i, [self.embed_sz], initializer=self.init) 
                         for i in range(self.memory_slots)]

        # Create Memory Cell
        self.cell = DynamicMemory(self.memory_slots, self.embed_sz, self.keys)

        self.output_w1 = tf.get_variable("OP_W1", [self.memory_slots, self.embed_sz], initializer=self.init)
        # TODO : SEE WHAT TO DO WITH THIS - Output Module Variables
        # self.H = tf.get_variable("H", [self.embed_sz, self.embed_sz], initializer=self.init) # TODO debug shape here
        self.R = tf.get_variable("R", [self.embed_sz, self.labels_dim], initializer=self.init) # TODO : Bring it to [None, mask_dim, labels_dim]

    def inference(self):
        """
        Build inference pipeline, going from the story and question, through the memory cells, to the
        distribution over possible answers.  
        """

        # Send Story through Memory Cell
        initial_state = self.cell.zero_state(self.bsz, dtype=tf.float32)
        memory_traces, memories = tf.nn.dynamic_rnn(self.cell, self.S, initial_state=initial_state) # sequence_length=self.story_len,
        
        stacked_memories = tf.stack(memory_traces, axis=2)
        memories = tf.reshape(stacked_memories, (-1, self.mask_dim, self.embed_sz))
        
        #print(memories.get_shape().as_list(), len(self.keys), self.keys[0].get_shape().as_list(), story_embeddings.get_shape().as_list())
        # [237, 40, 100] 8 [100] [None, 5, 100]
        # map each memory output into label_dim
        op_embedd = tf.reshape(memories, (-1, self.embed_sz))
        op_embedd = tf.matmul(op_embedd, self.R)
        logits = tf.reshape(op_embedd, (self.bsz * self.mask_dim,  self.labels_dim, 1))  # Shape: [batch_size, mask_dim, label_dim]

        logits = self.gat_main(logits, self.bias_adj, hid_units=[50], n_heads=[8, 1], nb_classes=1) # Shape : [1, labels_dim, labels_dim]
        logits = tf.reshape(logits, (self.bsz * self.mask_dim, self.labels_dim))

        # NOTE : @devamanyu
        '''
            Currently doing GAT on labels_embedding and multiplying to logits. Not very useful.

        '''
        return logits
        



        # Output Module 
        # stacked_memories = tf.stack(memories, axis=1)
        # op_embedd = tf.reduce_sum(tf.multiply(stacked_memories, self.output_w1), axis=[2]) # shape : [None, mem_slots]
        # op_embedd = tf.matmul(op_embedd, self.H)
        # logits = tf.matmul(op_embedd, self.R)
        # logits = tf.reshape(logits, [-1, self.mask_dim, self.labels_dim])

        # return logits

        # TODO : Change the model here
        # Generate Memory Scores
        '''
        p_scores = softmax(tf.reduce_sum(tf.multiply(stacked_memories,        # Shape: [None, mem_slots]
                                                     tf.expand_dims(query_embedding, 1)), axis=[2]))
        
        # Subtract max for numerical stability (softmax is shift invariant)
        p_max = tf.reduce_max(p_scores, axis=-1, keep_dims=True)
        attention = tf.nn.softmax(p_scores - p_max)       
        attention = tf.expand_dims(attention, 2)                              # Shape: [None, mem_slots, 1]

        # Weight memories by attention vectors
        u = tf.reduce_sum(tf.multiply(stacked_memories, attention), axis=1)   # Shape: [None, embed_sz]

        # Output Transformations => Logits
        hidden = prelu(tf.matmul(u, self.H) + query_embedding)                # Shape: [None, embed_sz]
        logits = tf.matmul(hidden, self.R)                                    # Shape: [None, vocab_sz]

        return logits
        '''

    def loss(self):
        """
        Build loss computation - softmax cross-entropy between logits, and correct answer. 
        """
        
        # TODO : Might have to implement my own loss
        return tf.losses.sigmoid_cross_entropy(self.ground_truth, self.logits)
    '''
    def accuracy(self):
        #ground_truth = tf.gather(self.ground_truth, self.mask_reshaped, axis=0)
        #logits = tf.nn.sigmoid(tf.gather(self.logits, self.mask_reshaped, axis=0))

        f1_score = tf.contrib.metrics.f1_score(labels=self.ground_truth, predictions=self.logits)
        precision = tf.metrics.precision(labels=self.ground_truth, predictions=self.logits)
        recall = tf.metrics.recall(labels=self.ground_truth, predictions=self.logits)
        return f1_score[0], precision[0], recall[0]
    '''
    def train(self):
        """
        Build ADAM Optimizer Training Operation.
        """
        learning_rate = tf.train.exponential_decay(self.lr, self.global_step, self.decay_steps, 
                                                   self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, 
                                                   learning_rate=learning_rate, optimizer="Adam",
                                                   clip_gradients=self.clip_gradients)
        return train_op

    def gat_main(self, inputs, bias_mat, hid_units, n_heads, nb_classes, attn_drop=0.6, ffd_drop=0.6, activation=tf.nn.elu, residual=False):
        attns = []
        for _ in range(n_heads[0]):
            attns.append(self.gat_attn_head(inputs, bias_mat=bias_mat,
                out_sz=hid_units[0], activation=activation,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(self.gat_attn_head(h_1, bias_mat=bias_mat,
                    out_sz=hid_units[i], activation=activation,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)
        out = []
        for i in range(n_heads[-1]):
            out.append(self.gat_attn_head(h_1, bias_mat=bias_mat,
                out_sz=nb_classes, activation=lambda x: x,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]

        return logits


    def gat_attn_head(self, seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
        # Source : https://github.com/PetarV-/GAT/blob/master/utils/layers.py
        with tf.name_scope('my_attn'):
            if in_drop != 0.0:
                seq = tf.nn.dropout(seq, 1.0 - in_drop)

            seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

            # simplest self-attention possible
            f_1 = tf.layers.conv1d(seq_fts, 1, 1)
            f_2 = tf.layers.conv1d(seq_fts, 1, 1)
            logits = f_1 + tf.transpose(f_2, [0, 2, 1])
            coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

            if coef_drop != 0.0:
                coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
            if in_drop != 0.0:
                seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

            vals = tf.matmul(coefs, seq_fts)
            ret = tf.contrib.layers.bias_add(vals)

            # residual connection
            if residual:
                if seq.shape[-1] != ret.shape[-1]:
                    ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
                else:
                    ret = ret + seq

            return activation(ret)  # activation

class DynamicMemory(tf.contrib.rnn.RNNCell):
    def __init__(self, memory_slots, memory_size, keys, activation=prelu,
                 initializer=tf.random_normal_initializer(stddev=0.1), attention=True):
        """
        Instantiate a DynamicMemory Cell, with the given number of memory slots, and key vectors.

        :param memory_slots: Number of memory slots to initialize. 
        :param memory_size: Dimensionality of memories => tied to embedding size. 
        :param keys: List of keys to seed the Dynamic Memory with (can be random).
        :param initializer: Variable Initializer for Cell Parameters.
        """ 
        self.m, self.mem_sz, self.keys = memory_slots, memory_size, keys
        self.activation, self.init = activation, initializer
        self.attention = attention

        # Instantiate Dynamic Memory Parameters => CONSTRAIN HERE
        # NOTE : Change dim back for not attention
        self.U = None
        if(self.attention):
            self.U = tf.get_variable("U", [237, self.mem_sz], initializer=self.init)
            self.AttentW = tf.get_variable("AttentW", [237, 1896], initializer=self.init)
        else:
            self.U = tf.get_variable("U", [self.mem_sz, self.mem_sz], initializer=self.init)
        self.V = tf.get_variable("V", [self.mem_sz, self.mem_sz], initializer=self.init)
        self.W = tf.get_variable("W", [self.mem_sz, self.mem_sz], initializer=self.init)
        #self.SoftAttenW = tf.get_variable("SoftAttenW", [237, self.mem_sz], initializer=self.init)
    
    @property
    def state_size(self):
        """
        Return size of DynamicMemory State - for now, just M x d. 
        """
        return [self.mem_sz for _ in range(self.m)]
    
    @property
    def output_size(self):
        return [self.mem_sz for _ in range(self.m)]
    
    def zero_state(self, batch_size, dtype):
        """
        Initialize Memory to start as Key Values
        """
        return [tf.tile(tf.expand_dims(key, 0), [batch_size, 1]) for key in self.keys]

    def __call__(self, inputs, state, scope=None):
        """
        Run the Dynamic Memory Cell on the inputs, updating the memories with each new time step.

        :param inputs: 2D Tensor of shape [bsz, mem_sz] representing a story sentence.
        :param states: List of length M, each with 2D Tensor [bsz, mem_sz] => h_j (starts as key).
        """
        new_states = []
        for block_id, h in enumerate(state):

            # New State Candidate
            w_component = tf.matmul(tf.expand_dims(self.keys[block_id], 0), self.V)      # Shape: [1, mem_sz]
            s_component = tf.matmul(inputs, self.W)                                      # Shape: [bsz, mem_sz]

            if(self.attention):
                # V = h_component, Q = s_component, K = w_component
                all_h = tf.stack(new_states[:block_id] + state[block_id:])
                h_component = all_h * self.U
                d_k = tf.cast(tf.shape(w_component)[-1], dtype=tf.float32)
                soft = tf.nn.softmax(tf.matmul(s_component, tf.transpose(w_component)) / d_k)
                attent = tf.multiply(h_component, soft)
                attentshape = attent.get_shape().as_list()
                attent = tf.reshape(attent, [attentshape[0] * attentshape[1], attentshape[2]])
                attent = tf.matmul(self.AttentW, attent)
                new_states.append(attent)
            else:
                h_component = tf.matmul(h, self.U)                                           # Shape: [bsz, mem_sz]
                candidate = self.activation(h_component + w_component + s_component)         # Shape: [bsz, mem_sz]
                # Gating Function            
                content_g = tf.reduce_sum(tf.multiply(inputs, h), axis=[1])                  # Shape: [bsz]
                address_g = tf.reduce_sum(tf.multiply(inputs, 
                                          tf.expand_dims(self.keys[block_id], 0)), axis=[1]) # Shape: [bsz]
                g = sigmoid(content_g + address_g)

                # State Update
                new_h = h + tf.multiply(tf.expand_dims(g, -1), candidate)                    # Shape: [bsz, mem_sz]

                # Unit Normalize State 
                new_h_norm = tf.nn.l2_normalize(new_h, -1)                                   # Shape: [bsz, mem_sz]
            
                new_states.append(new_h_norm)
        
        return new_states, new_states
