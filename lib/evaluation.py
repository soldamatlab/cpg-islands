from lib.classifier import NULL, CPG


def evaluate(prediction, truth):
    TP = sum((prediction == CPG) & (truth == CPG))
    TN = sum((prediction == NULL) & (truth == NULL))
    FP = sum((prediction == CPG) & (truth == NULL))
    FN = sum((prediction == NULL) & (truth == CPG))

    correct = TP + TN
    wrong = FP + FN
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return correct, wrong, accuracy, precision, recall
