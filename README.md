## Sujet choisit 

NLP applied to judicial decisions parsing

## Objectif

build an algorithm to automate the extraction of the relevant information :

- the sex of the victim : This information is always contained in the document and can only take two values : "homme" and "femme"

- the date of the accident : Except in very rare cases, this information is always domewhere in the document (usually at the beginning). It is the date when the accident happenned. We expect a date in the format dd/mm/yyyy.

- the date of the consolidation of the injuries : This is the date when the injuries of the victim became stable and were declared final by a physician. The information should be present in most cases but sometimes it is either missing (so we put "n.c." in the csv file) or not applicable (so we put "n.a." in the csv) if the injury did not stabilize before the death of the victim.

