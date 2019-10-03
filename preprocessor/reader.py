"""
reader.py

Core script containing preprocessing logic - reads bAbI Task Story, and returns
vectorized forms of the stories, questions, and answers.
"""
from config import *

import re
import os
import json
import bcolz
import pickle
import numpy as np
import jsonlines
from tqdm import tqdm
from collections import OrderedDict
from preprocessor.tokenizer import Tokenizer
from bert_embedding import BertEmbedding


PAD_ID = 0

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_ID, UNK_ID, SOS_ID, EOS_ID = [0, 1, 2, 3]


# t
tokenizer = Tokenizer('spacy')


glove = {}


LOC = "BERT" if USE_BERT else "GLOVE"
if not os.path.exists(DIR+LOC):
    os.makedirs(DIR+LOC)
train_pickle_path = "{}/train_data.pkl".format(DIR + LOC)
test_pickle_path = "{}/test_data.pkl".format(DIR + LOC)
val_pickle_path = "{}/val_data.pkl".format(DIR + LOC)
metadata_path = "{}/metadata.json".format(DIR + LOC)

partition_path = DIR + "storyid_partition.txt"
annotation_path = DIR + "json_version/annotations.json"


classes = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]




def init_glove(glove_path=GLOVE_PATH): # Run only first time
    """
    Sets up the glove embedding dictionary.
    First 4 indices are reserved for: '<pad>', '<unk>', '<sos>', '<eos>'
    """
    words, word2idx = [], {}
    idx = 4
    vectors = bcolz.carray(
        np.zeros(1), rootdir='{}/27B.{}d.dat'.format(glove_path, GLOVE_EMB_DIM), mode='w')

    words.append(PAD_TOKEN)
    words.append(UNK_TOKEN)
    words.append(SOS_TOKEN)
    words.append(EOS_TOKEN)

    word2idx[PAD_TOKEN] = PAD_ID
    word2idx[UNK_TOKEN] = UNK_ID
    word2idx[SOS_TOKEN] = SOS_ID
    word2idx[EOS_TOKEN] = EOS_ID

    vectors.append(np.zeros(GLOVE_EMB_DIM).astype(np.float))
    random_vector = np.random.uniform(-0.25, 0.25, 100).astype(np.float)
    vectors.append(random_vector.copy())
    vectors.append(random_vector.copy())
    vectors.append(random_vector.copy())

    with open('{}/glove.twitter.27B.{}d.txt'.format(glove_path, GLOVE_EMB_DIM), 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors.reshape((1193518, GLOVE_EMB_DIM)),
                           rootdir='{}/27B.{}.dat'.format(glove_path, GLOVE_EMB_DIM), mode='w')
    vectors.flush()
    pickle.dump(words, open(
        '{}/27B.{}_words.pkl'.format(glove_path, GLOVE_EMB_DIM), 'wb'))
    pickle.dump(word2idx, open(
        '{}/27B.{}_idx.pkl'.format(glove_path, GLOVE_EMB_DIM), 'wb'))
    return idx

def load_glove():
    """
        Load glove dictionary
        TODO: try 300 dimensional glove embeddings
    """
    global VOCAB
    vectors = bcolz.open('{}/27B.{}.dat'.format(GLOVE_PATH, GLOVE_EMB_DIM))[:]
    words = pickle.load(open('{}/27B.{}_words.pkl'.format(GLOVE_PATH, GLOVE_EMB_DIM), 'rb'))
    word2idx = pickle.load(open('{}/27B.{}_idx.pkl'.format(GLOVE_PATH, GLOVE_EMB_DIM), 'rb'))

    if(VOCAB == -1):
        VOCAB = len(words)

    return {w: vectors[word2idx[w]] for w in words}


def get_bert(sentences):
    """
        Returns BERT sentence embeddings for provided sentence text
    """

    BERT_LAYERS = 4

    with open("/tmp/input.txt", "w") as f:
        for sentence in sentences:
            f.write(sentence + "\n")

    os.system("python3 ./preprocessor/bert/extract_features.py \
              --input_file=/tmp/input.txt \
              --output_file=/tmp/output.jsonl \
              --vocab_file={}/vocab.txt \
              --bert_config_file={}/bert_config.json \
              --init_checkpoint={}/bert_model.ckpt \
              --layers=-1,-2,-3,-4 \
              --max_seq_length=128 \
              --batch_size=60".format(BERT_BASE_DIR, BERT_BASE_DIR, BERT_BASE_DIR))

    sentence_embeddings = []
    with jsonlines.open('/tmp/output.jsonl') as reader:
        for obj in reader:
            # 0 is for first toke -> [CLS]
            cls_layers = obj["features"][0]["layers"]
            embedding = np.zeros((BERT_EMB_DIM))
            for i in range(BERT_LAYERS):
                embedding += np.asarray(cls_layers[i]["values"])
            embedding /= BERT_LAYERS
            sentence_embeddings.append(embedding)
    return np.array(sentence_embeddings)


def prepare_for_bert(text_arr):
    """
        Function first flattens all sentences into a single list, gets the bert embeddings 
        and then reshapes them into story-wise setting
    """
    print("Preparing for BERT feature extraction")

    # Flatten all sentences across all stories 
    all_sentences = []
    cumulative_sentence_nums = []
    start, end = 0,0

    cnt=0
    for sentences in text_arr:
        for sent in sentences:
            all_sentences.append(sent)
        start = end
        end = end + len(sentences)
        cnt+= len(sentences)
        cumulative_sentence_nums.append((start, end))

    # get bert embeddings of all the sentences
    # numpy matrix: [# total_sentences, 768]
    bert_embeddings = get_bert(all_sentences)

    # Reshape sentences into story-wise manner
    sentence_embeddings = []
    for start, end in cumulative_sentence_nums:
        sentence_embeddings.append(bert_embeddings[start:end])
    
    return sentence_embeddings





def get_labels(charay):
    """
        Filtering labels as per details mention in original dataset release notes.
    """
    ann = []
    for i in range(3):
        try :
            ann.append(charay["emotion"]["ann{}".format(i)]["plutchik"])
        except:
            # print("ann{} ignored".format(i))
            continue

    if(len(ann) == 0): # NOTE - change this maybe
        return [0 for _ in classes]

    final_dict = dict()
    for classi in classes:
        final_dict[classi] = [1, 1, 1]

    for idx in range(len(ann)):
        for i in ann[idx]:
            if(i[:-2] in final_dict.keys()):
                final_dict[i[:-2]][idx] = int(i[-1])

    majority = []
    for key in final_dict.keys():
        if(int(sum(final_dict[key]) / 3) >= 2):
            majority.append(key) #[key if(floor(sum(final_dict[key]) / 3) >= 2) for key in final_dict.keys()]

    onehot = [1 if i in majority else 0 for i in classes]
    return onehot






def create_dataset(data_type="train"):
    global annotation_path, partition_path

    data_type = "dev" if data_type == "train" else data_type
    annotation_file = open(annotation_path, "r")
    raw_data = json.load(annotation_file, object_pairs_hook=OrderedDict)

    if not USE_BERT:
        glove = load_glove()

    text_arr, all_labels, char_arr, mask_arr = [], [], [], []


    for line in open(partition_path, "r"):
        id_key = line.split("\t")[0]

        if id_key not in raw_data.keys():
            print("Here")
            continue

        story = raw_data[id_key]
        
        if(story["partition"] != data_type):
            continue 

        sentences = story["lines"]
        characters = sentences['1']["characters"]

        
        s_dim, c_dim, count = len(sentences.keys()), len(characters.keys()), 0
        mask_dim = s_dim * c_dim
        sentence_embeddings, labels, mask = [], [], [0] * mask_dim

        for si in range(s_dim):
            sent = sentences[str(si + 1)]
            text = sent["text"]
            text = tokenizer(text)
            
            if USE_BERT:
                # Accumulate normal text and get BERT representations later in batches
                sentence_rep = " ".join(text) 
            else:
                # Glove averaging
                sentence_rep = [glove.get(word, glove['unk']) for word in text]
                sentence_rep = np.mean(np.array(sentence_rep), axis=0)

            sentence_embeddings.append(sentence_rep)
            
            charecs = list(sent["characters"].keys())
            labels.append([])
            for cj in range(c_dim):
                char = sent["characters"][charecs[cj]]
                one_hot = get_labels(char)
                labels[si].append([])
                labels[si][cj] = one_hot
                if(1 in one_hot):
                    mask[count] = 1
                count += 1
        
        # If BERT, then we delay the sentence representation extraction to be done at once for faster time
        embeddings = np.asarray(sentence_embeddings) if not USE_BERT else sentence_embeddings

        mask = np.asarray(mask)
        labels = np.asarray(labels)
        labels = labels.reshape(labels.shape[0] * labels.shape[1], labels.shape[2])
        
        all_labels.append(labels)   # Shape : [stories, s_d * c_d, labels_dim]
        mask_arr.append(mask)       # Shape : [stories, s_d * c_d]
        text_arr.append(embeddings) # Shape : [stories, s_d, embedding_dim]
        char_arr.append(c_dim)      # Shape : [stories, 1]  - No. of chars. - to find upper bound    

    if USE_BERT:
        text_arr = prepare_for_bert(text_arr)

    return text_arr, all_labels, mask_arr, char_arr



def get_adjacency(all_labels):
    # TODO: use BERT for label embedding too

    glove = load_glove()

    labels_embedding = np.asarray([glove.get(label, glove['unk']) for label in classes])


    # data driven adjacency
    adj_m = np.zeros((len(classes), len(classes)))

    for i in range(len(all_labels)):
        for j in range(all_labels[i].shape[0]):
            idx = np.where(all_labels[i][j] > 0)
            adj_m[idx] += all_labels[i][j]


    
    # self edges are added later, so remove them now
    for i in range(adj_m.shape[0]):
        adj_m[i] = adj_m[i]/adj_m[i, i]  # p(A|B) = P(A,B)/P(B) = #(A,B)/#(B) 
        adj_m[i, i] = 0

    adj_m[adj_m >= ADJACENCY_THRESHOLD] = 1
    adj_m[adj_m < ADJACENCY_THRESHOLD] = 0
    print(adj_m)

    return labels_embedding, adj_m






def pad_stories(text_arr, all_labels, mask_arr, max_sentence_length, max_char_length):
    
    for i in range(len(text_arr)):
        story = text_arr[i]
        shape = story.shape
        sentence_pad = max_sentence_length - shape[0]
        # FML
        new_story = []
        story_type = type(story[0])
        if(story_type != list):
            for j in range(shape[0]):
                a = story[j].tolist()
                new_story.append(a)

        if(story_type != list):
            story = np.asarray([new_story[k] for k in range(shape[0])])
        else:
            story = np.asarray([story[k] for k in range(shape[0])])
        if(sentence_pad != 0):
            text_arr[i] = np.pad(story, ((0, sentence_pad), (0, 0), (0, 0)), 'constant')
        else :
            text_arr[i] = story
    
    for i in range(len(all_labels)):
        label = all_labels[i]
        shape = label.shape
        pad_length = max_sentence_length * max_char_length - shape[0]
        all_labels[i] = np.pad(label, ((0, pad_length), (0,0)), 'constant')

    for i in range(len(mask_arr)):
        mask = mask_arr[i]
        shape = mask.shape
        pad_length = max_sentence_length * max_char_length - shape[0]
        mask_arr[i] = np.pad(mask, ((0, pad_length)), 'constant')

    mask_arr = np.asarray(mask_arr, dtype='int')     # Shape : [max_sentence_length, s_d * c_d]
    text_arr = np.asarray(text_arr)     # Shape : [max_sentence_length, embedding_dim]
    all_labels = np.asarray(all_labels) # Shape : [max_sentence_length, s_d * c_d, labels_dim]

    mask_arr = np.expand_dims(mask_arr, 2)
    return text_arr, all_labels, mask_arr









def parse_data(load=True):
    
    def load_pickle_data(pickle_path):
        if(os.path.isfile(pickle_path) and load == True):
            file = open(pickle_path, 'rb+')
            data = pickle.load(file)
            file.close()
            return data
        return None
    
    train_data = load_pickle_data(train_pickle_path)
    val_data = load_pickle_data(val_pickle_path)
    test_data = load_pickle_data(test_pickle_path)

    if((train_data is not None) and (val_data is not None) and (test_data is not None) ):
        print(train_data[0].shape, train_data[1].shape,
              train_data[2].shape, test_data[0].shape)
        return train_data, test_data, val_data


        

    data = {"train" : (), "test" :()}
    msl, mcl, mask_dim, labels_dim, embedding_dim = 0, 0, 0, 0, EMB_DIM

    for dtype in data.keys():
        text_arr, all_labels, mask_arr, char_arr= create_dataset(data_type=dtype)

        
        dataset_size = len(text_arr)
        sentence_lengths = [story.shape[0] for story in text_arr]

        max_sentence_length = max(sentence_lengths)
        max_char_length = max(char_arr)
        msl = max(msl, max_sentence_length)
        mcl = max(mcl, max_char_length)

        _, labels_dim = len(all_labels[0]), len(all_labels[0][0])
        mask_dim = max_sentence_length*max_char_length
        data[dtype] = (text_arr, all_labels,  mask_arr)

    print(msl, mcl)

    for dtype in data.keys():
        text_arr, all_labels, mask_arr= data[dtype]
        text_arr, all_labels, mask_arr = pad_stories(text_arr, all_labels, mask_arr, msl, mcl)
        print(text_arr.shape)
        data[dtype] = (text_arr, all_labels,  mask_arr)





    # Splitting data in train and validation sets
    nsamples = int(text_arr.shape[0] * 0.8)
    idx = [i for i in range(data["train"][0].shape[0])]
    np.random.shuffle(idx)

    data["val"] = (data["train"][0][idx[nsamples:]], data["train"][1][idx[nsamples:]], data["train"][2][idx[nsamples:]])
    data["train"] = (data["train"][0][idx[:nsamples]], data["train"][1][idx[:nsamples]], data["train"][2][idx[:nsamples]])

    for dtype in data.keys():
        text_arr, all_labels, mask_arr = data[dtype]
        print(dtype, text_arr.shape, all_labels.shape, mask_arr.shape)

    with open(metadata_path, 'w') as f:
        metadata = {
            'max_char_length': mcl,
            'max_sentence_length': msl,
            'mask_dim' : mask_dim,
            'labels_dim' : labels_dim,
            'emedding_dim' : embedding_dim,
            'vocab_size': VOCAB,
            'dataset_size': dataset_size,
        }
        json.dump(metadata, f)

    with open(train_pickle_path, "wb+") as handle:
        pickle.dump(data["train"], handle)

    with open(test_pickle_path, "wb+") as handle:
        pickle.dump(data["test"], handle)

    with open(val_pickle_path, "wb+") as handle:
        pickle.dump(data["val"], handle)

    exit()

    return data["train"], data["test"], data["val"]



if __name__ == '__main__':
    parse(load=False)
