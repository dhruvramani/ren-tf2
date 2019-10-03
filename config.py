DIR = "/home/nevronas/Projects/Personal-Projects/Dhruv/NeuralDialog-CVAE/data/commonsense/"
GLOVE_PATH = '/home/nevronas/word_embeddings/glove_twitter'
BERT_BASE_DIR = "/home/nevronas/word_embeddings/uncased_L-12_H-768_A-12"
#DIR = "/home/devamanyu/ren-tf2/storycommonsense_data/"
#GLOVE_PATH = '/home/devamanyu/glove_twitter'

#BERT_BASE_DIR = "./preprocessor/bert/uncased_L-12_H-768_A-12"
GLOVE_EMB_DIM = 100
BERT_EMB_DIM = 768
MAX_WLEN = 18
VOCAB = -1

ADJACENCY_THRESHOLD = 0.2
ATTENTION = True
GAN = False



USE_BERT = True

EMB_DIM = BERT_EMB_DIM if USE_BERT else GLOVE_EMB_DIM
