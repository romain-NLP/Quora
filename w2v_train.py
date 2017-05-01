# import modules and set up logging
from gensim.models import word2vec
import logging
from os.path import join
path='w2v'


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# load up unzipped corpus from http://mattmahoney.net/dc/text8.zip
sentences = word2vec.Text8Corpus(join(path, 'text8'))
# train the skip-gram model; default window=5
model = word2vec.Word2Vec(sentences, size=100)
model.save(join(path, 'model_text8'))