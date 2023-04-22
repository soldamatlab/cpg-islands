import numpy as np

ALPHABET = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3,
    'end': 4
    }

NULL = 0
CPG = 1


def build_classifier(null_train, cpg_train):
    """
    :return
    """
    apriori = get_apriori(null_train, cpg_train)
    null_model = build_model(null_train)
    cpg_model = build_model(cpg_train)
    
    apriori_log_odds = log_odds(apriori[0], apriori[1])
    symbol_log_odds, transitions_log_odds = log_odds_models(cpg_model, null_model)

    return apriori_log_odds, symbol_log_odds, transitions_log_odds


def get_apriori(null_train, cpg_train):
    apriori = np.array([len(null_train), len(cpg_train)])
    apriori = apriori / sum(apriori)
    return apriori


def build_model(data):
    symbols = np.ones(len(ALPHABET), dtype=int)
    transitions = np.ones([len(ALPHABET), len(ALPHABET)])

    for sequence in data:
        for i in range(len(sequence)-1):
            symbols[ALPHABET[sequence[i]]] += 1
            transitions[ALPHABET[sequence[i]], ALPHABET[sequence[i+1]]] += 1
        symbols[ALPHABET[sequence[-1]]] += 1
        transitions[ALPHABET[sequence[-1]], ALPHABET['end']] += 1
        symbols[ALPHABET['end']] += 1

    symbols = symbols / np.sum(symbols) # ! numerical
    transitions /= transitions.sum(axis=1)[:,np.newaxis] # ! numerical

    return (symbols, transitions)


def log_odds(p1, p2): return np.log(p1/p2) # ! numerical


def log_odds_models(model1, model2):
    symbol_log_odds = log_odds_symbols(model1[0], model2[0])
    transitions_log_odds = log_odds_transitions(model1[1], model2[1])
    return symbol_log_odds, transitions_log_odds


def log_odds_symbols(symbols1, symbols2):
    symbol_log_odds = np.empty_like(symbols1)
    for s in range(len(symbols1)):
        symbol_log_odds[s] = log_odds(symbols1[s], symbols2[s])
    return symbol_log_odds


def log_odds_transitions(transitions1, transitions2):
    transitions_log_odds = np.empty_like(transitions1)
    for i in range(transitions1.shape[0]):
        for j in range(transitions1.shape[1]):
            transitions_log_odds[i,j] = log_odds(transitions1[i,j], transitions2[i,j])
    return transitions_log_odds


def classify_sequences(test_sequences, classifier):
    classification = np.empty(len(test_sequences), dtype=int)
    for i in range(len(test_sequences)):
        classification[i] = classify(test_sequences[i], classifier)
    return classification


def classify(sequence, classifier):
    apriori_log_odds = classifier[0]
    symbol_log_odds = classifier[1]
    transitions_log_odds = classifier[2]

    # ! numerical
    log_odds = apriori_log_odds + symbol_log_odds[ALPHABET[sequence[0]]]
    for i in range(1, len(sequence)):
        log_odds += transitions_log_odds[ALPHABET[sequence[i-1]], ALPHABET[sequence[i]]]
    log_odds += transitions_log_odds[ALPHABET[sequence[-1]], ALPHABET['end']]

    return CPG if log_odds > 0 else NULL
