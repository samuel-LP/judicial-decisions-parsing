# Judicial decisions text parser

## Welcome on our Judicial decisions text parser!

The objective of this repository is to create a parser for judicial decisions texts. 
The information targeted are the following :
- The sex of the victim : it can be male or female
- The date of accident : this is the date when the accident happened. It is quite always present in the document.
- The date of conoslidation of the injuries : This is the date when the injuries of the victim became stable and were declared final by a physician. It is not always present in the document.

The datas come from this challenge : [NLP applied to judicial decisions parsing](https://challengedata.ens.fr/participants/challenges/24/)

## Installations
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

## Objectif

build an algorithm to automate the extraction of the relevant information :

- the sex of the victim : This information is always contained in the document and can only take two values : "homme" and "femme"

- the date of the accident : Except in very rare cases, this information is always domewhere in the document (usually at the beginning). It is the date when the accident happenned. We expect a date in the format dd/mm/yyyy.

- the date of the consolidation of the injuries : This is the date when the injuries of the victim became stable and were declared final by a physician. The information should be present in most cases but sometimes it is either missing (so we put "n.c." in the csv file) or not applicable (so we put "n.a." in the csv) if the injury did not stabilize before the death of the victim.

