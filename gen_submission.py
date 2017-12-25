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
    nlp = spacy.load("./output")

    raw_data = {'id': []}
    for label in labels:
        raw_data[label]= []

    import csv
    with open(submission_path, 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(["id" ]+labels) # header
        for id, text in load_test_data(test_path, limit):
            output = [id]
            if  str(id) != str("206058417140"):
                doc = nlp(text)
            else:
                wr.writerow(output +[0.5]*6)
                continue

            for label in labels:
                output.append(doc.cats[label.decode('utf-8')])
            wr.writerow(output)


def load_test_data(test_data_path, limit):
    test = pd.read_csv(test_data_path, header = 0)
    test_data = zip(test['id'], [str(item).decode('utf-8') for item in test['comment_text']])
    return test_data[:limit]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', dest="model_dir", default=str("./output"))
    parser.add_argument('-t', '--test_file', dest="test_file", type= file, default=str("./test.csv"))
    parser.add_argument('-o', '--output_file', dest="output_file", type= str, default= str("./submission.csv"))
    parser.add_argument('-l', '--limit', type=int, dest="limit", default= sys.maxint)
    args = parser.parse_args()
    main(args.model_dir, args.test_file, args.output_file, args.limit)