import spacy
import pandas as pd


def tokenize(s: pd.Series) -> pd.Series:
    nlp = spacy.load('ja_ginza')

    tokens = []
    for doc in nlp.pipe(s.astype("unicode").values, batch_size=32):
        tokens.append(" ".join([token.text for token in doc]))

    return pd.Series(tokens, index=s.index)
