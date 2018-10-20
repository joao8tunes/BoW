## BoW
Bag of Words (BoW) is the more traditional text representation technique based in vector space model. This technique assume that the frequency relationship between words is independent, considering the relancy of a word directly proporcional to the frequency with that word occurs in a text. Thus, this BoW based script generate an unique vector representation to each document, calculating the frequency of all words in all documents. The output is a matrix, where rows are the documents ids, and columns are the words frequencies to each document.
> Generating a BoW based text representation matrix:
```
python3 BoW.py --language EN --n_gram 1 --input in/db/ --output out/BoW/txt/
```
> Converting a Doc-Term matrix to Arff file (Weka):
```
python3 Bag2Arff.py --token - --input out/Bag/txt/ --output out/Bag/arff/
```


### Related scripts
* [BoW.py](https://github.com/joao8tunes/BoW/blob/master/BoW.py)
* [Bag2Arff.py](https://github.com/joao8tunes/Bag2Arff/blob/master/Bag2Arff.py)


### Assumptions
These scripts expect a database folder following an specific hierarchy like shown below:
```
in/db/                 (main directory)
---> class_1/          (class_1's directory)
---------> file_1      (text file)
---------> file_2      (text file)
---------> ...
---> class_2/          (class_2's directory)
---------> file_1      (text file)
---------> file_2      (text file)
---------> ...
---> ...
```


### Observations
All generated files use *TAB* character as a separator.


### Requirements installation (Linux)
> Python 3 + PIP installation as super user:
```
apt-get install python3 python3-pip
```
> NLTK installation as normal user:
```
pip3 install -U nltk
```


### See more
Project page on LABIC website: http://sites.labic.icmc.usp.br/MSc-Thesis_Antunes_2018
