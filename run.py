"""
run.py

Core script for building, training, and evaluating a Recurrent Entity Network.
"""
from model.entity_network import EntityNetwork
from preprocessor.reader import parse, init_glove
import datetime
import os
import pickle
import tensorflow as tf
import tflearn
import json
from sklearn.metrics import precision_score, recall_score, f1_score


FLAGS = tf.app.flags.FLAGS

_DIR = "/home/nevronas/Projects/Personal-Projects/Dhruv/NeuralDialog-CVAE/data/commonsense/"
_GLOVE_PATH = '/home/nevronas/word_embeddings/glove_twitter'
#_DIR = "/mnt/data/devamanyu/work/StoryCommonSense/storycommonsense_data/"
#_GLOVE_PATH = '/mnt/data/devamanyu/work/glove_twitter'
_EMB_DIM = 100
_MAX_WLEN = 18
_VOCAB = -1

pickle_path = _DIR + "data.pkl"
metadata_path = _DIR + "metadata.json"
partition_path = _DIR + "storyid_partition.txt"
annotation_path = _DIR + "json_version/annotations.json"

classes = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]

# Run Details
#tf.app.flags.DEFINE_integer("task_id", 1, "ID of Task to Train/Evaluate [1 - 20].")
#tf.app.flags.DEFINE_string("data_path", "tasks/en-valid-10k", "Path to Training Data")

# Model Details
tf.app.flags.DEFINE_integer("embedding_size", _EMB_DIM, "Dimensionality of word embeddings.")
tf.app.flags.DEFINE_integer("memory_slots", 8, "Number of dynamic memory slots.")

# Training Details
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("num_epochs", 200, "Number of Training Epochs.")
tf.app.flags.DEFINE_float("learning_rate", .01, "Learning rate for ADAM Optimizer.")
tf.app.flags.DEFINE_integer("decay_epochs", 25, "Number of epochs to run before learning rate decay.")
tf.app.flags.DEFINE_float("decay_rate", 0.5, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_float("clip_gradients", 40.0, 'Norm to clip gradients to.')

# Logging Details
tf.app.flags.DEFINE_integer("validate_every", 10, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_float("validation_threshold", .95, "Validation threshold for early stopping.")

def metrics(logits, truth):
    logits[logits > 0.5] = 1
    logits[logits < 0.5] = 1
    prec = precision_score(truth, logits, average='macro')
    recall = recall_score(truth, logits, average='macro')
    fscore = f1_score(truth, logits, average='macro')
    return fscore, prec, recall

def main(load=True):
    # Get Vectorized Forms of Stories, Questions, and Answers
    train, test, val = parse()
    train_text_arr, train_all_labels, train_mask_arr = train
    val_text_arr, val_all_labels, val_mask_arr = val
    test_text_arr, test_all_labels, test_mask_arr = test

    # Setup Checkpoint + Log Paths
    ckpt_dir = "./checkpoints/"
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    
    with tf.gfile.Open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)

    # Build Model
    with tf.Session() as sess:
        # Instantiate Model
        entity_net = EntityNetwork(metadata['vocab_size'], metadata['max_word_length'], metadata['max_sentence_length'], FLAGS.batch_size,
                                   FLAGS.memory_slots, FLAGS.embedding_size, metadata['mask_dim'], metadata['labels_dim'], FLAGS.learning_rate, 
                                   FLAGS.decay_epochs * (metadata['dataset_size'] / FLAGS.batch_size), FLAGS.decay_rate)
        
        # Initialize Saver
        saver = tf.train.Saver()

        # Initialize all Variables
        if os.path.exists(ckpt_dir + "checkpoint") and load == True:
            print('Restoring Variables from Checkpoint!')
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
            with open(ckpt_dir + "training_logs.pik", 'rb') as f:
                train_loss, train_acc, val_loss, val_acc = pickle.load(f)
        else:
            print('Initializing Variables!')
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            train_loss, train_acc, val_loss, val_acc = {}, {}, {}, {}
        
        # Get Current Epoch
        curr_epoch = sess.run(entity_net.epoch_step)

        # Start Training Loop
        n, test_n, val_n, bsz, best_val = train_text_arr.shape[0], test_text_arr.shape[0], val_text_arr.shape[0], FLAGS.batch_size, 0.0
        for epoch in range(curr_epoch, FLAGS.num_epochs):
            loss, f1, prec, recall, counter = 0.0, 0.0, 0.0, 0.0, 0
            for start, end in zip(range(0, n, bsz), range(bsz, n, bsz)):
                curr_loss, ground_truth, logits, _ = sess.run([entity_net.loss_val, entity_net.ground_truth, tf.nn.sigmoid(entity_net.logits), entity_net.train_op], # = [] 
                                                  feed_dict={entity_net.S: train_text_arr[start:end], 
                                                             entity_net.labels: train_all_labels[start:end],
                                                             entity_net.mask: train_mask_arr[start:end]})
                curr_f1, curr_prec, curr_rec = metrics(logits, ground_truth)
                loss, f1, prec, recall, counter = loss + curr_loss, f1 + curr_f1, prec + curr_prec, recall + curr_rec, counter + 1
                print("Epoch %d\tBatch %d \t Train Loss: %.3f \t Train Metrics [F, P, R] : %.3f, %.3f, %.3f" % (epoch, counter, loss / float(counter), f1 / float(counter), prec / float(counter), recall / float(counter)), end="\r")
                if counter % 100 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss: %.3f\tTrain Metrics [F, P, R]: %.3f, %.3f, %.3f" % (epoch, counter, loss / float(counter), f1 / float(counter), prec / float(counter), recall / float(counter)))
            if(epoch % 10 == 0):
                print("")
            # Add train loss, train acc to data
            train_loss[epoch], train_acc[epoch] = loss / float(counter), [f1 / float(counter), prec / float(counter), recall / float(counter)]

            # Increment Epoch
            sess.run(entity_net.epoch_increment)
            
            # Validate every so often
            if epoch % FLAGS.validate_every == 0:
                val_loss_val, val_acc_val = do_eval(val_n, bsz, sess, entity_net, val_text_arr, val_all_labels, val_mask_arr)
                print("VAL Epoch %d Test Loss: %.3f\Tes F1: Metrics [F, P, R] : %.3f, %.3f, %.3f" % (epoch, val_loss_val, val_acc_val[0], val_acc_val[1], val_acc_val[2]))
                
                # Add val loss, val acc to data 
                val_loss[epoch], val_acc[epoch] = val_loss_val, val_acc_val

                # Update best_val
                if val_acc[epoch][0] > best_val:
                    best_val = val_acc[epoch][0]

                # Save Model
                saver.save(sess, ckpt_dir + "model.ckpt", global_step=entity_net.epoch_step)
                with open(ckpt_dir + "training_logs.txt", 'wb+') as f:
                    pickle.dump((train_loss, train_acc, val_loss, val_acc), f)

                # Early Stopping Condition
                if best_val > FLAGS.validation_threshold:
                    break
            
        # Test Loop
        test_loss, test_acc = do_eval(test_n, bsz, sess, entity_net, test_text_arr, test_all_labels, test_mask_arr)
        

        # Print and Write Test Loss/Accuracy
        print("Test Loss: %.3f\tTest Metrics [F, P, R] : %.3f, %.3f, %.3f" % (test_loss, test_acc[0], test_acc[1], test_acc[2]))
        with open(ckpt_dir + "output.txt", 'w') as g:
            g.write("Test Loss: %.3f\tTest Metrics [F, P, R] : %.3f, %.3f, %.3f\n" % (test_loss, test_acc[0], test_acc[1], test_acc[2]))

def do_eval(n, bsz, sess, entity_net, text_arr, labels, mask):
    """Perform an Evaluation Epoch on the Given Data"""
    eval_loss, eval_f1, eval_prec, eval_recall, eval_counter = 0.0, 0.0, 0.0, 0.0, 0
    for start, end in zip(range(0, n, bsz), range(bsz, n, bsz)):
        curr_eval_loss, ground_truth, logits = sess.run([entity_net.loss_val, entity_net.ground_truth, tf.nn.sigmoid(entity_net.logits)],
                                                 feed_dict={entity_net.S: text_arr[start:end],
                                                            entity_net.labels: labels[start:end],
                                                            entity_net.mask: mask[start:end]})
        cf1, cpr, cre = metrics(logits, ground_truth)
        eval_loss, eval_f1, eval_prec, eval_recall, eval_counter = eval_loss + curr_eval_loss, eval_f1 + cf1, eval_prec + cpr, eval_recall + cre, eval_counter + 1
    return eval_loss / float(eval_counter), [eval_f1 / float(eval_counter), eval_prec / float(eval_counter), eval_recall / float(eval_counter)]
    

if __name__ == "__main__":
    tf.app.run()
