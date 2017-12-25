from __future__ import unicode_literals

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))


import argparse
import logging
import random
from pathlib import Path

import en_core_web_sm
import spacy
from spacy.util import minibatch, compounding

logger = logging.getLogger(__name__)
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def main(model, training_path, output_path, iterations, split):
    print(
        "Initialising spacy categorizer, training path: {}, output path: {}, iterations: {}".format(training_path,
                                                                                                    output_path,
                                                                                                    str(iterations)))
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    #nlp = en_core_web_sm.load()

    textcat = get_textcat_pipe(nlp)
    map(lambda x: textcat.add_label(x), labels)

    (train_texts,train_cats), (dev_texts, dev_cats) = load_data(training_path, split)
    print("Using ({} training, {} evaluation)"
          .format(len(train_texts), len(dev_texts)))

    train_data = list(zip(train_texts,
                    [{'cats': ele} for ele in train_cats]))
    
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        print("Training the model...")
        print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))
        for i in range(iterations):
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                #print texts, annotations
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
                           losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
            print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'  # print a simple table
                  .format(losses['textcat'], scores['textcat_p'],
                          scores['textcat_r'], scores['textcat_f']))

    # test the trained model
    test_text = "wanna fuck you"
    doc = nlp(test_text)
    print(test_text, doc.cats)

    if output_path is not None:
        output_dir = Path(output_path)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        print(test_text, doc2.cats)

# From Spacy example
def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 1e-8  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 1e-8  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)
    return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score}


def get_textcat_pipe(nlp):
    # Add text categorizer to the spacy pipeline
    if 'textcat' not in nlp.pipe_names:
        textcat = nlp.create_pipe('textcat')
        nlp.add_pipe(textcat, last='true')
    else:
        textcat = nlp.get_pipe('textcat')
    return textcat

def load_data(training_path, split):
    train = pd.read_csv(training_path, header = 0)

    X, y = [item.decode('utf-8') for item in train['comment_text']], train[labels]
   
    ## convert to proper label which spacy accept
    accepted_tags = []
    from collections import OrderedDict
    for index, row in y.iterrows():
        cat_input = OrderedDict()
        for label in labels:
            if row[label] == 1 :
                cat_input[label]=True
            else:
                cat_input[label]=False
        accepted_tags.append(dict(cat_input));

    from sklearn.model_selection import train_test_split
    X_train, X_test , y_train, y_test = train_test_split(X, accepted_tags, test_size = split, random_state = 42)

    return (X_train, y_train), (X_test, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--training_path')
    parser.add_argument('-o', '--output_path')
    parser.add_argument('-i', '--iterations')
    parser.add_argument('-s', '--split', default=0.7)
    args = parser.parse_args()
    main(model = "./output", training_path="./train.csv", output_path="./output", iterations=20, split=0.10)