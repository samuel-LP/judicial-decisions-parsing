# Juridical decisions parsing

## Welcome on our juridical decisions parser!

The objective of this repository is to create a parser for judicial decisions texts. 
The information targeted are the following :
- The sex of the victim : it can be male or female
- The date of accident : this is the date when the accident happened. It is quite always present in the document.
- The date of conoslidation of the injuries : This is the date when the injuries of the victim became stable and were declared final by a physician. It is not always present in the document.

The datas come from this challenge : [NLP applied to judicial decisions parsing](https://challengedata.ens.fr/participants/challenges/24/)

## Installation

To run our code, do the following steps : 

1. Clone the repo by executing the following command :

```bash
git clone https://github.com/samuel-LP/juridical-decisions-parser
cd juridical-decisions-parser
```

2. Create your virtual environment (venv) 

```bash
python -m venv venv
```

3. Activate your venv
- On Windows:

    ```bash
    .\venv\Scripts\Activate
    ```

- On Linux/Mac

    ```bash
    source venv/bin/activate
    ```

4. install the dependencies

```bash
pip install -r requirements.txt
```

## How do we parse the juridical text?

To predict the sex of the victim, we used a TF-IDF

For the dates, we used 2 different methods :
1. For each texts, we used a NER for isolating the dates and the context behind it. After that used a RAG to predict the dates.
2. We applied a RAG on all the text.

## Project structure

1. **src**

The project source code, this folder include:
- For the sex recognition: the data collator, the dataloader, the BERT model and the embeddings model scripts.
- For the date recognition: the data preprocessing, ,the metrics evaluation, the RAG code and the normalization of dates scripts.

2. **notebooks**

This folder contains Jupyter notebooks used for exploratory data analysis and all the models and metrics we used in this project.

## Authors

- [Axel Fritz](https://github.com/AxelFritz2)
- [Jynaldo Jeannot](https://github.com/jeannoj99)
- [Samuel Pariente Launay](https://github.com/samuel-LP)
