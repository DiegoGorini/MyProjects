#!/bin/env python3
# coding: utf-8

import sys
import gensim
import logging
import zipfile
import json

# Simple toy script to get an idea of what one can do with word embedding models using Gensim
# Models can be found at http://vectors.nlpl.eu/explore/embeddings/models/
# or in the /cluster/shared/nlpl/data/vectors/latest/  directory on Saga

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    embeddings_file = sys.argv[1]  # File containing word embeddings

    logger.info('Loading the embedding model...')

    # Detect the model format by its extension:

    # Binary word2vec format:
    if embeddings_file.endswith('.bin.gz') or embeddings_file.endswith('.bin'):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=True, unicode_errors='replace')
    # Text word2vec format:
    elif embeddings_file.endswith('.txt.gz') or embeddings_file.endswith('.txt') \
            or embeddings_file.endswith('.vec.gz') or embeddings_file.endswith('.vec'):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=False, unicode_errors='replace')
    # ZIP archive from the NLPL vector repository:
    elif embeddings_file.endswith('.zip'):
        with zipfile.ZipFile(embeddings_file, "r") as archive:
            # Loading and showing the metadata of the model:
            metafile = archive.open('meta.json')
            metadata = json.loads(metafile.read())
            for key in metadata:
                print(key, metadata[key])
            print('============')

            # Loading the model itself:
            stream = archive.open("model.bin")  # or model.txt, if you want to look at the model
            emb_model = gensim.models.KeyedVectors.load_word2vec_format(
                stream, binary=True, unicode_errors='replace')
    else:  # Native Gensim format?
        emb_model = gensim.models.KeyedVectors.load(embeddings_file)

        #  If you intend to train the model further:
        # emb_model = gensim.models.Word2Vec.load(embeddings_file)

    emb_model.init_sims(replace=True)  # Unit-normalizing the vectors (if they aren't already)

    logger.info('Finished loading the embedding model...')

    logger.info('Model vocabulary size: %d' % len(emb_model.vocab))

    logger.info('Example of a word in the model: "%s"' % emb_model.index2word[3])

# ———————————————————————ASSIGNMENT————————————————————————

with open(sys.argv[2]) as file:
    d = file.readlines()
    d = [word[:-1] for word in d]

myList = []
for word in d:
    try:
        x = emb_model.most_similar(word, topn=15)[10:15]
        myList.append(x)
        print(x)
    except:
        myList.append(["Not found"])


# for (i, word) in enumerate(myList):
#    print("Word:", i+1)
#    for result in word:
#        print(result)

with open(sys.argv[3], "w+") as file:
    for (i, word) in enumerate(myList):
        file.write("Word " + str(i+1) + ": " +  d[i] + "\n")
        for result in word:
            file.write(str(result) + "\n")
