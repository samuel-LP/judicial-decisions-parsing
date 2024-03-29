from langchain_community.embeddings.openai import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from tqdm import tqdm
import pandas as pd
from rag_preprocess import PreprocessDocuments


class RetrievalAugmentedGenerator():
    def __init__(self, df, query, openai_model, embedding_model_name):
        self.df = df
        self.query = query
        self.openai_model = openai_model
        self.embedding_model_name = embedding_model_name

    def retriver(self):
        res_date = []

        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):

            texte = row["texte"]
            filename = row["filename"]

            # content = preprocess_documents(pd.Series({"texte": texte, "filename": filename}))
            content = PreprocessDocuments(pd.Series({"texte": texte,
                                                     "filename": filename}
                                                     )).preprocess_documents()
            if self.openai_model != None:
                model = ChatOpenAI(model=self.openai_model, temperature=0.2)

            else:
                model = ChatOpenAI(temperature=0.2)

            if self.embedding_model_name == None:
                embedding_model = OpenAIEmbeddings()
                faiss_index = FAISS.from_documents(content, embedding_model)

            else:
                embedding_model = \
                    HuggingFaceEmbeddings(model_name=self.embedding_model_name)
                faiss_index = \
                    FAISS.from_documents(content, embedding_model)
            
            retriever = faiss_index.as_retriever(search_kwargs={'k': 7},
                                                 search_type="mmr")

            if self.query == "what is the date of accident?":
                template = """You are a legal assistant.
                You will look for information in the following document: {context}.
                You will search for one of the following information :
                - the date of the accident : The date of the accident refers to the date on which the accident occurred
                You will only return the date with the following format : YYYY-MM-DD. I don't want text around it
                The user is going to ask you the following thing: {question}
                """

                prompt = ChatPromptTemplate.from_template(template)

                chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | model
                    | StrOutputParser()
                )

                val = chain.invoke(self.query)
                res_date.append(val)

            elif self.query == "what is the date of consolidation?":
                template = """You are a legal assistant. You will look for information in the following document::
                {context}.

                You will search for one of the following information :
                - the date of consolidation : The consolidation date is the date on which the victim's injuries
                became stable and were declared definitive by a doctor.
                The information should be present in most cases, but sometimes it is either missing (you will display the value "n.c.") or not applicable (you will display the value "n.a.") if the injury did not stabilize before the victim's death.

                You will return the date with the following format : YYYY-MM-DD. I don't want text around it
                If the date of consolidation is missing in the document, you will return the value n.c.
                If the injury did not stabilize before the victim's death, you will return the value n.a.

                The user is going to ask you the following thing: {question}
                """

                prompt = ChatPromptTemplate.from_template(template)

                chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | model
                    | StrOutputParser()
                )

                val = chain.invoke(self.query)
                res_date.append(val)
        return res_date
