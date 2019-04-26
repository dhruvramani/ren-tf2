"""
reader.py

Core script containing preprocessing logic - reads bAbI Task Story, and returns
vectorized forms of the stories, questions, and answers.
"""
import re
import os
import json
import bcolz
import pickle
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

FORMAT_STR = "qa%d_"
PAD_ID = 0
SPLIT_RE = re.compile('(\W+)?')
DATA_TYPES = ['train', 'valid', 'test']

_DIR = "/home/nevronas/Projects/Personal-Projects/Dhruv/NeuralDialog-CVAE/data/commonsense/"
_GLOVE_PATH = '/home/nevronas/word_embeddings/glove_twitter'
#_DIR = "/mnt/data/devamanyu/work/StoryCommonSense/storycommonsense_data/"
#_GLOVE_PATH = '/mnt/data/devamanyu/work/glove_twitter'
_EMB_DIM = 100
_MAX_WLEN = 18
_VOCAB = -1

train_pickle_path = _DIR + "train_data.pkl"
test_pickle_path = _DIR + "test_data.pkl"
val_pickle_path = _DIR + "val_data.pkl"
partition_path = _DIR + "storyid_partition.txt"
metadata_path = _DIR + "metadata.json"
annotation_path = _DIR + "json_version/annotations.json" 

classes = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]

def init_glove(glove_path=_GLOVE_PATH): # Run only first time
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir='{}/27B.{}d.dat'.format(glove_path, _EMB_DIM), mode='w')
    with open('{}/glove.twitter.27B.{}d.txt'.format(glove_path, _EMB_DIM), 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors.reshape((1193514, _EMB_DIM)), rootdir='{}/27B.{}.dat'.format(glove_path, _EMB_DIM), mode='w')
    vectors.flush()
    pickle.dump(words, open('{}/27B.{}_words.pkl'.format(glove_path, _EMB_DIM), 'wb'))
    pickle.dump(word2idx, open('{}/27B.{}_idx.pkl'.format(glove_path, _EMB_DIM), 'wb'))
    return idx

def tokenize(sentence):
    "Tokenize a string by splitting on non-word characters and stripping whitespace."
    return [token.strip().lower() for token in re.split(SPLIT_RE, sentence) if token.strip()]

def load_glove():
    global _VOCAB
    vectors = bcolz.open('{}/27B.{}.dat'.format(_GLOVE_PATH, _EMB_DIM))[:]
    words = pickle.load(open('{}/27B.{}_words.pkl'.format(_GLOVE_PATH, _EMB_DIM), 'rb'))
    word2idx = pickle.load(open('{}/27B.{}_idx.pkl'.format(_GLOVE_PATH, _EMB_DIM), 'rb'))

    if(_VOCAB == -1):
        _VOCAB = len(words)

    return {w: vectors[word2idx[w]] for w in words}

def get_labels(charay):
        ann = []
        for i in range(3):
            try :
                ann.append(charay["emotion"]["ann{}".format(i)]["plutchik"])
            except:
                print("ann{} ignored".format(i))

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
    glove = load_glove()

    text_arr, all_labels, char_arr, mask_arr = [], [], [], []
    stories_dat = []
    with open(partition_path, "r") as partition_file:
        for line in partition_file:
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
            embeddings, labels, mask = [], [], [0] * mask_dim

            for si in range(s_dim):
                sent = sentences[str(si + 1)]
                text = sent["text"]
                
                embed_string = re.sub(r"[^a-zA-Z]+", ' ', text)
                embedding = [glove.get(word, glove['unk']) for word in embed_string.split(" ")]
                embeddings.append(embedding)
                
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


            mask = np.asarray(mask)
            labels = np.asarray(labels)
            embeddings = np.asarray(embeddings)
            labels = labels.reshape(labels.shape[0] * labels.shape[1], labels.shape[2])
            
            all_labels.append(labels)   # Shape : [stories, s_d * c_d, labels_dim]
            mask_arr.append(mask)       # Shape : [stories, s_d * c_d]
            text_arr.append(embeddings) # Shape : [stories, s_d, words, embedding_dim]
            char_arr.append(c_dim)      # Shape : [stories, 1]  - No. of chars. - to find upper bound
            
            # OR - decide
            #stories_dat.append((embeddings, labels, mask, c_dim))

    return text_arr, all_labels, mask_arr, char_arr # stories_dat # - ALL ARE LISTS

def pad_stories(text_arr, all_labels, mask_arr, max_sentence_length, max_word_length, max_char_length):
    
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
        for j in range(shape[0]):
            if(len(story[j]) != max_word_length):
                if(story_type != list):
                    a = new_story[j]
                    for i in range(max_word_length - len(a)):
                        a.append([0] * _EMB_DIM)
                    new_story[j] = np.asarray(a)
                else :
                    story[j] = story[j] + [[0] * _EMB_DIM] * (max_word_length - len(list(story[j])))
                    story[j] = np.asarray(story[j])
        if(story_type != list):
            story = np.asarray([new_story[k] for k in range(shape[0])])
        else:
            story = np.asarray([story[k] for k in range(shape[0])])
        if(sentence_pad != 0):
            text_arr[i] = np.pad(story, ((0, sentence_pad), (0, 0), (0, 0)), 'constant')
        else :
            text_arr[i] = story

    for i in range(len(text_arr)):
        if(text_arr[i].shape[1] != max_word_length):
            pad_length = max_word_length - text_arr[i].shape[1]
            text_arr[i] = np.pad(text_arr[i], ((0, 0), (0, pad_length), (0,0)), 'constant')
    
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
    text_arr = np.asarray(text_arr)     # Shape : [max_sentence_length, max_word_length, embedding_dim]
    all_labels = np.asarray(all_labels) # Shape : [max_sentence_length, s_d * c_d, labels_dim]

    mask_arr = np.expand_dims(mask_arr, 2)
    return text_arr, all_labels, mask_arr

def parse(load=True):
    train_data, test_data, val_data = (), (), ()
    
    if(os.path.isfile(train_pickle_path) and load == True):
        file = open(train_pickle_path, 'rb+')
        train_data = pickle.load(file)
        file.close()

    if(os.path.isfile(test_pickle_path) and load == True):
        file = open(test_pickle_path, 'rb+')
        test_data = pickle.load(file)
        file.close()

    if(len(train_data) != 0 and len(test_data) != 0):
        print(train_data[0].shape, train_data[1].shape, train_data[2].shape, test_data[0].shape)
        return train_data, test_data, None, None

    data = {"train" : (), "test" :()}
    msl, mwl, mcl, mask_dim, labels_dim, embedding_dim = 0, 0, 0, 0, 0, _EMB_DIM

    for dtype in data.keys():
        text_arr, all_labels, mask_arr, char_arr = create_dataset(data_type=dtype)
    
        dataset_size = len(text_arr)
        sentence_lengths = [story.shape[0] for story in text_arr]
        word_lengths = [len(story[ss]) for story in text_arr for ss in range(story.shape[0])]

        max_sentence_length = max(sentence_lengths)
        max_word_length = max(word_lengths)
        max_char_length = max(char_arr)

        if(max_sentence_length > msl):
            msl = max_sentence_length
        if(max_word_length > mwl):
            mwl = max_word_length
        if(max_char_length > mcl):
            mcl = max_char_length

        _, labels_dim = len(all_labels[0]), len(all_labels[0][0])
        mask_dim = max_sentence_length*max_char_length
        data[dtype] = (text_arr, all_labels,  mask_arr)
        # TODO : MOVE
    print(msl, mwl, mcl)  

    for dtype in data.keys():
        text_arr, all_labels, mask_arr = data[dtype]
        text_arr, all_labels, mask_arr = pad_stories(text_arr, all_labels, mask_arr, msl, mwl, mcl)
        data[dtype] = (text_arr, all_labels,  mask_arr)
        print(text_arr.shape, all_labels.shape, mask_arr.shape)

    with open(metadata_path, 'w') as f:
        metadata = {
            'max_char_length': mcl,
            'max_word_length': mwl,
            'max_sentence_length': msl,
            'mask_dim' : mask_dim,
            'labels_dim' : labels_dim,
            'emedding_dim' : embedding_dim,
            'vocab_size': _VOCAB,
            'dataset_size': dataset_size,
        }
        json.dump(metadata, f)

    with open(train_pickle_path, "wb+") as handle:
        pickle.dump(data["train"], handle)

    with open(test_pickle_path, "wb+") as handle:
        pickle.dump(data["test"], handle)

    return data["train"], data["test"], None, None # TODO : Change this

def tokenize(sentence):
    """
    Tokenize a string by splitting on non-word characters and stripping whitespace.
    """
    return [token.strip().lower() for token in re.split(SPLIT_RE, sentence) if token.strip()]

if __name__ == '__main__':
    parse()
