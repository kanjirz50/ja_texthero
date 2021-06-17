import spacy
import pandas as pd


def named_entities(s):
    entities = []

    nlp = spacy.load('ja_ginza')

    for doc in nlp.pipe(s.astype("unicode").values, batch_size=32):
        entities.append(
            [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
        )

    return pd.Series(entities, index=s.index)
