
from nltk.tokenize import sent_tokenize
from langchain_core.documents import Document
import pandas as pd
from nltk.corpus import stopwords
stop_words = stopwords.words("french")


class Preprocessing():
    def __init__(self, data):
        self.data = data

    def remove_newlines(self):
        self.data = self.data.replace("\n", '', regex=True)
        return self.data

    def remove_stopwords(self):
        self.data["texte"] = \
            self.data["texte"].apply(lambda x: ' '.join(
                [word for word in x.split() if word not in (stop_words)]))

        return self.data


class PreprocessDocuments():
    def __init__(self, df_or_series):

        self.df_or_series = df_or_series

    def preprocess_documents(self):
        pages_content = []

        if isinstance(self.df_or_series, pd.Series):
            df = self.df_or_series.to_frame().transpose()
        else:
            df = self.df_or_series

        for _, row in df.iterrows():
            texte = row["texte"]
            name = row["filename"]

            phrases = sent_tokenize(texte)

            for phrase in phrases:
                texte_doc = Document(page_content=phrase,
                                     metadata={'name': name})
                pages_content.append(texte_doc)

        return pages_content
