from __future__ import unicode_literals

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import argparse, sys, os
from pathlib import Path

import spacy

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def main(model_dir, test_path, submission_path, limit):
    # test the saved model
    print("Loading from", model_dir)
    nlp = spacy.load(model_dir)

    raw_data = {'id': []}
    for label in labels:
        raw_data[label]= []

    for id, text in load_test_data(test_path, limit):
        doc = nlp(text)
        for key in raw_data.keys():
            if key is 'id':
                raw_data['id'].append(id)
            else:
                raw_data[key].append(doc.cats[key.decode('utf-8')])
    # write to csv
    df = pd.DataFrame(raw_data, columns = ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
    
    if os.path.exists(submission_path):
        os.remove(str(submission_path))
    else:
        submission_path = open(str(submission_path), "w")

    df.to_csv(submission_path, index=False)

def load_test_data(test_data_path, limit):
    test = pd.read_csv(test_data_path, header = 0)
    test_data = zip(test['id'], [str(item).decode('utf-8') for item in test['comment_text']])
    return test_data[:limit]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', dest="model_dir", default="./output")
    parser.add_argument('-t', '--test_file', dest="test_file", type= file, default="./test.csv")
    parser.add_argument('-o', '--output_file', dest="output_file", type= str, default= "./submission.csv")
    parser.add_argument('-l', '--limit', type=int, dest="limit", default= sys.maxint)
    args = parser.parse_args()
    main(args.model_dir, args.test_file, args.output_file, args.limit)