#!/usr/bin/python

import sys
import numpy as np

from lib.classifier import build_classifier, classify_sequences
from lib.evaluation import evaluate

'''
My solution of the assignment builds upon
the sample solution for the Bioinformatics course homework submission draft
by Petr Ryšavý <petr.rysavy@fel.cvut.cz>.
'''


def read_sequences(path):
    with open(path, "r") as file_handle:
        return (file_handle.read().splitlines())


if __name__ == '__main__':
    # Train
    null_train = read_sequences("null_train.txt")
    cpg_train = read_sequences("cpg_train.txt")
    classifier = build_classifier(null_train, cpg_train)

    # Test
    test_sequences = read_sequences("seqs_test.txt")
    predictions = classify_sequences(test_sequences, classifier)
    with open("predictions.txt", "w") as file_handle:
        file_handle.writelines(str(x)+'\n' for x in predictions)

    # Evaluate
    test_classes = np.array([int(cl) for cl in read_sequences("classes_test.txt")])
    correct, wrong, accuracy, precision, recall = evaluate(predictions, test_classes)
    with open("accuracy.txt", "w") as file_handle:
        file_handle.writelines(
            str(x)+'\n' for x in [correct, wrong, accuracy, precision, recall])
