""" Core script for building, valing, and evaluating a Recurrent Entity Network. """

from config import *

from model.entity_network import EntityNetwork
from preprocessor.reader import parse, init_glove

import os
import json
import pickle
import tflearn
import datetime
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support


FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

pickle_path = DIR + "data.pkl"
metadata_path = DIR + "metadata.json"
partition_path = DIR + "storyid_partition.txt"
annotation_path = DIR + "json_version/annotations.json"

if Path(metadata_path).is_file:
    with tf.io.gfile.GFile(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)

classes = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]

# Model Details
tf.app.flags.DEFINE_integer("embedding_size", EMB_DIM, "Dimensionality of word embeddings.")
tf.app.flags.DEFINE_integer("memory_slots", 8, "Number of dynamic memory slots.")

# Training Details
tf.app.flags.DEFINE_integer("batch_size", 237, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("eval_batch_size", 237, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("num_epochs", 50, "Number of Training Epochs.")
tf.app.flags.DEFINE_float("learning_rate", .1, "Learning rate for ADAM Optimizer.")
tf.app.flags.DEFINE_integer("decay_epochs", 25, "Number of epochs to run before learning rate decay.")
tf.app.flags.DEFINE_float("decay_rate", 0.5, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_float("clip_gradients", 40.0, 'Norm to clip gradients to.')

# Logging Details
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")
# tf.app.flags.DEFINE_float("validation_threshold", .95, "Validation threshold for early stopping.")


def check_if_glove_prepared():
    """
    Checks whether init_glove has been previously called. Calls if didn't earlier.
    """
    is_glove_setup = Path('{}/27B.{}_words.pkl'.format(GLOVE_PATH, EMB_DIM)).is_file()
    if not is_glove_setup:
        print("Setting up glove")
        init_glove()


# def metrics(logits, truth):
#     logits[logits > 0.5] = 1
#     logits[logits < 0.5] = 1
#     prec = precision_score(truth, logits, average='macro')
#     recall = recall_score(truth, logits, average='macro')
#     fscore = f1_score(truth, logits, average='macro')
#     return fscore, prec, recall

def adj_to_bias(adj, sizes, nhood=1):
    '''
        Source : https://github.com/PetarV-/GAT/blob/master/utils/process.py
        Prepare adjacency matrix by expanding up to a given neighbourhood.
        This will insert loops on every node.
        Finally, the matrix is converted to bias vectors.
        Expected shape: [graph, nodes, nodes]
    '''
    nb_graphs = 1
    mt = np.eye(adj.shape[1])
    
    for _ in range(nhood):
        mt = np.matmul(mt, (adj + np.eye(adj.shape[0])))
        
    for i in range(sizes):
        for j in range(sizes):
            if mt[i][j] > 0.0:
                mt[i][j] = 1.0
    return -1e9 * (1.0 - mt)



def main(load=True):

    # Get Vectorized Forms
    train, test, val = parse()
    train_text_arr, train_all_labels, train_mask_arr, labels_embedding, adj_m = train
    val_text_arr, val_all_labels, val_mask_arr, _, _ = val
    test_text_arr, test_all_labels, test_mask_arr, _, _ = test

    # Framing adjacency matrix as bias
    adj_bias = adj_to_bias(adj_m, adj_m.shape[0], nhood=1)
    labels_embedding = labels_embedding[np.newaxis] # (nb_graphs, nb_nodes, inp_fts)
    adj_bias = adj_bias[np.newaxis]  # (nb_graphs, nb_nodes, nb_nodes)

    # Setup Checkpoint + Log Paths
    ckpt_dir = "./checkpoints/"
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)


    # Build Model
    with tf.Session() as sess:
        # Instantiate Model
        entity_net = EntityNetwork(metadata['vocab_size'], metadata['max_sentence_length'], FLAGS.batch_size,
                                   FLAGS.memory_slots, FLAGS.embedding_size, metadata['mask_dim'], metadata['labels_dim'], FLAGS.learning_rate, 
                                   FLAGS.decay_epochs * (metadata['dataset_size'] / FLAGS.batch_size), FLAGS.decay_rate)
        
        # Initialize Saver
        saver = tf.train.Saver()

        # Initialize all Variables
        if os.path.exists(ckpt_dir + "checkpoint") and load == True:
            print('Restoring Variables from Checkpoint!')
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
            with open(ckpt_dir + "training_logs.pik", 'rb') as f:
                train_loss_history, train_metric, val_loss_history, val_metric = pickle.load(f)
        else:
            print('Initializing Variables!')
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            train_loss_history, train_metric_history, val_loss_history, val_metric_history = {}, {}, {}, {}
        
        # Get Current Epoch
        curr_epoch = sess.run(entity_net.epoch_step)

        # Start Training Loop
        n, test_n, val_n, bsz, best_val_loss = train_text_arr.shape[0], test_text_arr.shape[0], val_text_arr.shape[0], FLAGS.batch_size, np.inf
        eval_bsz = FLAGS.eval_batch_size
        best_val_epoch = -1

        for epoch in range(curr_epoch, FLAGS.num_epochs):

            train_loss, y_true, y_pred = [], [], []
            for batch_i, (start, end) in enumerate(tqdm(zip(range(0, n, bsz), range(bsz, n, bsz)), ncols=80)):
                
                # generate mask index
                mask = np.reshape(np.array(train_mask_arr[start:end]),-1)
                mask_index = np.array([idx for idx, val in enumerate(mask) if val == 1])
                
                labels_unrolled = np.reshape(train_all_labels[start:end], (-1, metadata['labels_dim']))
                curr_loss, ground_truth, logits, _ = sess.run([entity_net.loss_val, entity_net.ground_truth, tf.nn.sigmoid(entity_net.logits), entity_net.train_op], # = [] 
                                                  feed_dict={entity_net.S: train_text_arr[start:end], 
                                                             entity_net.labels: labels_unrolled,
                                                             entity_net.mask: mask_index,
                                                             entity_net.bias_adj : adj_bias})

                train_loss.append(curr_loss)
                ground_truth = ground_truth.astype("int")
                predictions = (logits >= 0.5).astype("int")
                
                [y_true.append(label) for label in ground_truth]
                [y_pred.append(label) for label in predictions]

                tqdm.write("Epoch: {}, iter {}: loss = {:.3f}".format(epoch + 1, batch_i, np.mean(train_loss)))

            # Add train loss, train metric to data
            train_loss, train_metric = np.mean(train_loss), precision_recall_fscore_support(np.array(y_true), np.array(y_pred), average="micro")[:3]
            train_loss_history[epoch] = train_loss
            train_metric_history[epoch] = train_metric

            tqdm.write(f"Train loss: {train_loss:.3f}   ;   [P, R, F-score]: {train_metric}")

            # Increment Epoch
            sess.run(entity_net.epoch_increment)
            
            # Validate every so often
            if epoch % FLAGS.validate_every == 0:
                val_loss, val_metric = do_eval(val_n, bsz, sess, entity_net, val_text_arr, val_all_labels, val_mask_arr, adj_bias)
                
                
                # Add val loss, val acc to data 
                tqdm.write(f"Val loss: {val_loss:.3f}   ;   [P, R, F-score]: {val_metric}")
                val_loss_history[epoch] = val_loss
                val_metric_history[epoch] = val_metric

                # Update best_val
                if val_loss < best_val_loss:
                    tqdm.write("\nBest val loss")
                    best_val_loss = val_loss
                    best_val_metric = val_metric
                    best_val_epoch = epoch
                    test_loss, test_metric = do_eval(test_n, eval_bsz, sess, entity_net, test_text_arr, test_all_labels, test_mask_arr, adj_bias)
                    tqdm.write(f"Test loss: {test_loss}   ;   [P, R, F-score]: {test_metric}")


                # Save Model
                saver.save(sess, ckpt_dir + "model.ckpt", global_step=entity_net.epoch_step)
                with open(ckpt_dir + "training_logs.txt", 'wb+') as f:
                    pickle.dump((train_loss, train_metric, val_loss, val_metric), f)

                # Early Stopping Condition
                # if best_val > FLAGS.validation_threshold:
                #     break
            
        
        tqdm.write("Train Loss:")
        print([train_loss_history[epoch] for epoch in range(FLAGS.num_epochs)])
        tqdm.write("Val Loss:")
        print([val_loss_history[epoch] for epoch in range(FLAGS.num_epochs)])

        # Test Loop
        tqdm.write(f"Best Val loss: {best_val_loss}   ;   [P, R, F-score]: {best_val_metric}   ;   Best val epoch: {best_val_epoch}")
        tqdm.write(f"Test loss: {test_loss}   ;   [P, R, F-score]: {test_metric}")
        

def do_eval(n, bsz, sess, entity_net, text_arr, labels, mask, adj_bias):
    """Perform an Evaluation Epoch on the Given Data"""

    eval_loss, y_true, y_pred = [], [], []
    for start, end in zip(range(0, n+1, bsz), range(bsz, n+1, bsz)):
        # generate mask index
        mask = np.reshape(np.array(mask[start:end]),-1)
        mask_index = np.array([idx for idx, val in enumerate(mask) if val == 1])
        
        labels_unrolled = np.reshape(labels[start:end], (-1,metadata['labels_dim']))

        curr_eval_loss, ground_truth, logits = sess.run([entity_net.loss_val, entity_net.ground_truth, tf.nn.sigmoid(entity_net.logits)],
                                                 feed_dict={entity_net.S: text_arr[start:end],
                                                            entity_net.labels: labels_unrolled,
                                                            entity_net.mask: mask_index,
                                                            entity_net.bias_adj : adj_bias})
        eval_loss.append(curr_eval_loss)    
        ground_truth = ground_truth.astype("int")
        predictions = (logits >= 0.5).astype("int")
        [y_true.append(label) for label in ground_truth]
        [y_pred.append(label) for label in predictions]
    
    eval_loss, eval_metric = np.mean(eval_loss), precision_recall_fscore_support(np.array(y_true), np.array(y_pred), average="micro")[:3]
    return eval_loss, eval_metric
        
    

if __name__ == "__main__":
    check_if_glove_prepared()
    tf.compat.v1.app.run()
