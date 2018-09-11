#!/usr/bin/python3.4
# -*- coding: utf-8 -*-
################################################################################
##              Laboratory of Computational Intelligence (LABIC)              ##
##             --------------------------------------------------             ##
##       Originally developed by: João Antunes  (joao8tunes@gmail.com)        ##
##       Laboratory: labic.icmc.usp.br    Personal: joaoantunes.esy.es        ##
##                                                                            ##
##   "Não há nada mais trabalhoso do que viver sem trabalhar". Seu Madruga    ##
################################################################################

import filecmp
import datetime
import argparse
import codecs
import logging
import nltk
import os
import sys
import time
import math
import re


################################################################################
### FUNCTIONS                                                                ###
################################################################################

# Print iterations progress: https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
def print_progress(iteration, total, estimation, prefix='   ', decimals=1, bar_length=100, final=False):
    columns = 32    #columns = os.popen('stty size', 'r').read().split()[1]    #Doesn't work with nohup.
    eta = str( datetime.timedelta(seconds=max(0, int( math.ceil(estimation) ))) )
    bar_length = int(columns)
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write('\r%s %s%s |%s| %s' % (prefix, percents, '%', bar, eta))

    if final == True:    #iteration == total
        sys.stdout.write('\n')

    sys.stdout.flush()


#Format a value in seconds to "day, HH:mm:ss".
def format_time(seconds):
    return str( datetime.timedelta(seconds=max(0, int( math.ceil(seconds) ))) )


#Convert a string value to boolean:
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("invalid boolean value: " + "'" + v + "'")


#Verify if a value correspond to a natural number (it's an integer and bigger than 0):
def natural(v):
    try:
        v = int(v)

        if v > 0:
            return v
        else:
            raise argparse.ArgumentTypeError("invalid natural number value: " + "'" + v + "'")
    except ValueError:
        raise argparse.ArgumentTypeError("invalid natural number value: " + "'" + v + "'")


#Verify if a string correspond to a common word (has just digits, letters (accented or not), hyphens and underlines):
def isword(word):
    if not any( l.isalpha() for l in word ):
        return False

    return all( l.isalpha() or bool(re.search("[A-Za-z0-9-_\']+", l)) for l in word )

################################################################################


################################################################################

#URL: https://github.com/joao8tunes/BoW

#Example usage: python3 BoW.py --language EN --input in/db/ --output out/BoW/txt/

#Defining script arguments:
parser = argparse.ArgumentParser(description="BoW based text representation generator\n=======================================")
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
optional.add_argument("--log", metavar='BOOL', type=str2bool, action="store", dest="log", nargs="?", const=True, default=False, required=False, help='display log during the process: y, [N]')
optional.add_argument("--tokenize", metavar='BOOL', type=str2bool, action="store", dest="tokenize", nargs="?", const=True, default=False, required=False, help='specify if texts need to be tokenized: y, [N]')
optional.add_argument("--ignore_case", metavar='BOOL', type=str2bool, action="store", dest="ignore_case", nargs="?", const=True, default=True, required=False, help='ignore case: [Y], n')
optional.add_argument("--stemm", metavar='BOOL', type=str2bool, action="store", dest="stemm", nargs="?", const=True, default=True, required=False, help='enable stemming (case insensitive): [Y], n')
optional.add_argument("--validate_words", metavar='BOOL', type=str2bool, action="store", dest="validate_words", nargs="?", const=True, default=True, required=False, help='validate vocabulary ([A-Za-z0-9-_\']+): [Y], n')
optional.add_argument("--stoplist", metavar='FILE_PATH', type=str, action="store", dest="stoplist", default=None, required=False, nargs="?", const=True, help='specify stoplist file')
optional.add_argument("--n_gram", metavar='INT', type=natural, action="store", dest="n_gram", default=1, nargs="?", const=True, required=False, help='specify max. (>= 1) N-gram: [1]')
optional.add_argument("--doc_freq", metavar='INT', type=natural, action="store", dest="doc_freq", default=2, nargs="?", const=True, required=False, help='min. frequency of documents (>= 1): [2]')
optional.add_argument("--metric", metavar='STR', type=str, action="store", dest="metric", default="TF-IDF", nargs="?", const=True, required=False, help='metric of term relevance: tf, idf, [TF-IDF]')
optional.add_argument("--print_features", metavar='BOOL', type=str2bool, action="store", dest="print_features", nargs="?", const=True, default=True, required=False, help='print features in BoW header: [Y], n')
required.add_argument("--language", metavar='STR', type=str, action="store", dest="language", nargs="?", const=True, required=True, help='language of database: EN, ES, FR, DE, IT, PT')
required.add_argument("--input", "-i", metavar='DIR_PATH', type=str, action="store", dest="input", required=True, nargs="?", const=True, help='input directory of database')
required.add_argument("--output", "-o", metavar='DIR_PATH', type=str, action="store", dest="output", required=True, nargs="?", const=True, help='output directory to save the BoWs')
args = parser.parse_args()    #Verifying arguments.

################################################################################


################################################################################

#Setup logging:
if args.log:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if args.language == "ES":      #Spanish.
    nltk_language = "spanish"
    stemmer = nltk.stem.snowball.SpanishStemmer()
elif args.language == "FR":    #French.
    nltk_language = "french"
    stemmer = nltk.stem.snowball.FrenchStemmer()
elif args.language == "DE":    #Deutsch.
    nltk_language = "german"
    stemmer = nltk.stem.snowball.GermanStemmer()
elif args.language == "IT":    #Italian.
    nltk_language = "italian"
    stemmer = nltk.stem.snowball.ItalianStemmer()
elif args.language == "PT":    #Portuguese.
    nltk_language = "portuguese"
    stemmer = nltk.stem.snowball.PortugueseStemmer()
else:                          #English.
    args.language = "EN"
    nltk_language = "english"
    stemmer = nltk.stem.snowball.EnglishStemmer()

total_start = time.time()

################################################################################


################################################################################
### INPUT (LOAD DATABASE AND STOPLIST)                                       ###
################################################################################

log = codecs.open("BoW-log_" + time.strftime("%Y-%m-%d") + "_" + time.strftime("%H-%M-%S") + "_" + str(uuid.uuid4().hex) + ".txt", "w", "utf-8")
print("\nBoW based text representation generator\n=======================================\n\n\n")
log.write("BoW based text representation generator\n=======================================\n\n\n")
log.write("> Parameters:\n")

if args.tokenize:
    log.write("\t- Tokenize:\t\tyes\n")
else:
    log.write("\t- Tokenize:\t\tno\n")

if args.ignore_case:
    log.write("\t- Ignore case:\t\tyes\n")
else:
    log.write("\t- Ignore case:\t\tno\n")

if args.stemm:
    log.write("\t- Stemming:\t\tyes\n")
else:
    log.write("\t- Stemming:\t\tno\n")
    
if args.validate_words:
    log.write("\t- Validate words:\tyes\n")
else:
    log.write("\t- Validate words:\tno\n")    

if args.stoplist is not None:
    log.write("\t- Stoplist:\t\t" + args.stoplist + "\n")

log.write("\t- N-grams:\t\t<= " + str(args.n_gram) + "\n")
log.write("\t- Doc. frequency:\t>= " + str(args.doc_freq) + "\n")
args.metric = args.metric.lower()

if args.metric == "tf":
    log.write("\t- Metric:\t\tTF\n")
elif args.metric == "idf":
    log.write("\t- Metric:\t\tIDF\n")
else:
    args.metric = "tf-idf"
    log.write("\t- Metric:\t\tTF-IDF\n")

if args.print_features:
    log.write("\t- Print features:\tyes\n")
else:
    log.write("\t- Print features:\tno\n")

log.write("\t- Language:\t\t" + args.language + "\n")
log.write("\t- Input:\t\t" + args.input + "\n")
log.write("\t- Output:\t\t" + args.output + "\n\n\n")

if not os.path.exists(args.input):
    print("ERROR: input directory does not exists!\n\t!Directory: " + args.input)
    log.write("ERROR: input directory does not exists!\n\t!Directory: " + args.input)
    log.close()
    sys.exit()

print("> Loading input filepaths...\n\n\n")
files_list = []

#Loading all filepaths from all root directories:
for directory in os.listdir(args.input):
    for file_item in os.listdir(args.input + "/" + directory):
        files_list.append(args.input + directory + "/" + file_item)

files_list.sort()
total_num_examples = len(files_list)
stoplist = []
log.write("> Database: " + args.input + "\n")
log.write("\t# Files: " + str(total_num_examples) + "\n\n")

for filepath in files_list:
    log.write("\t" + filepath + "\n")

if args.stoplist is not None:
    print("> Loading stoplist...\n\n\n")
    stoplist_file = codecs.open(args.stoplist, "r", encoding='utf-8')

    for line in stoplist_file.readlines():
        stoplist.append(line.strip())

    if args.ignore_case:
        stoplist = [w.lower() for w in stoplist]

    stoplist.sort()
    stoplist_file.close()

################################################################################


################################################################################
### TASK 1 - N-GRAM VARIATION                                                ###
################################################################################

out_string = args.output + "BoW_1st-2nd_ng"
print("> TASK 1 - N-GRAM VARIATION / TASK 2 - TEXT REPRESENTATION:")
print("..................................................")
total_operations = args.n_gram*total_num_examples
total_num_paragraphs = 0
total_num_sentences = 0
filepath_i = 0
eta = 0
print_progress(filepath_i, total_operations, eta)
operation_start = time.time()

for n in range(1, args.n_gram+1):
    document_words = []
    vectors = ""
    
    for filepath in files_list:
        file_item = codecs.open(filepath, "r", "UTF-8")
        paragraphs = [p.strip() for p in file_item.readlines()]    #Removing extra spaces.
        file_item.close()
        words = []
        
        if n == 1:
            total_num_paragraphs += len(paragraphs)

        for paragraph in paragraphs:            
            sentences = nltk.sent_tokenize(paragraph, nltk_language)    #Identifying sentences.
            
            if n == 1:
                total_num_sentences += len(sentences)
            
            for sentence in sentences:                
                if args.tokenize:
                    tokens = nltk.tokenize.word_tokenize(sentence)    #Works well for many European languages.
                else:
                    tokens = sentence.split()
                    
                if args.ignore_case:
                    tokens = [t.lower() for t in tokens]
                
                if args.validate_words:
                    allowed_tokens = [t for t in tokens if isword(t) and t not in stoplist]    #Filter allowed tokens.
                else:
                    allowed_tokens = [t for t in tokens if t not in stoplist]    #Filter allowed tokens.
                    
                n_grams = nltk.everygrams(allowed_tokens, max_len=n)    #Call n-grams combinations.
        
                for n_gram in n_grams:
                    if args.stemm:
                        new_word = "_".join([stemmer.stem(t) for t in n_gram])
                    else:
                        new_word = "_".join(n_gram)
        
                    if new_word not in stoplist:
                        words.append(new_word)

        document_words.append({"length": len(words), "words": sorted(words), "class": filepath.split('/')[-2].strip()})    #The words is sorted.

    ############################################################################
    ### TASK 2 - TEXT REPRESENTATION                                         ###
    ############################################################################

    bag = []
    features = []

    for words in document_words:
        start = time.time()
        bag.append([])    #New document line of frequencies.

        #Reading all words:
        for word in words["words"]:
            new_word = True

            #Checking if the word has already been inserted in features list (BoW column):
            for feature_i, feature in enumerate(features):
                if word == feature:
                    new_word = False
                    new_cell = True

                    #Searching where (if has) is it the correspondent column of current 'feature' of this line (bag[-1]):
                    for item_i, item in enumerate(bag[-1]):
                        if item['index'] == feature_i:
                            new_cell = False
                            bag[-1][item_i]['freq'] += 1
                            break

                    #Adding new correspondent item in this line:
                    if new_cell:
                        bag[-1].append({'index': feature_i, 'freq': 1})

                    break

            #Adding new word in features list (BoW column):
            if new_word:
                features.append(word)
                bag[-1].append({'index': len(features)-1, 'freq': 1})

        filepath_i += 1
        end = time.time()
        eta = (total_operations-filepath_i)*(end-start)
        print_progress(filepath_i, total_operations, eta)

    ############################################################################


    ############################################################################
    ### CALCULATING WORDS RELEVANCE:                                         ###
    ############################################################################

    bow = []
    len_features = len(features)
    output_file = codecs.open(out_string + str(n), "w", encoding='utf-8')
    doc_occurences = [0]*len_features

    for doc in bag:    #Frequency.
        bow.append([0]*len_features)

        for cell in doc:
            bow[-1][ cell['index'] ] = cell['freq']
            doc_occurences[ cell['index'] ] += 1

    if args.metric == "tf":    #Term Frequency.
        for doc_i, doc in enumerate(bow):
            for freq_j, freq in enumerate(doc):
                bow[doc_i][freq_j] /= document_words[doc_i]["length"]
    elif args.metric == "idf":    #Inverse Document Frequency.
        for doc_i, doc in enumerate(bow):
            for freq_j, freq in enumerate(doc):
                bow[doc_i][freq_j] = math.log(total_num_examples / doc_occurences[freq_j])
    else:    #Term Frequency - Inverse Document Frequency:
        for doc_i, doc in enumerate(bow):
            for freq_j, freq in enumerate(doc):
                bow[doc_i][freq_j] = (bow[doc_i][freq_j] / document_words[doc_i]["length"]) * math.log(total_num_examples / doc_occurences[freq_j])

    #Removing features with frequencies less than specified document frequency:
    del_indexes = []

    for feature_i, doc_occurence in enumerate(doc_occurences):
        if doc_occurence < args.doc_freq:
            del_indexes.append(feature_i)

    for index in sorted(del_indexes, reverse=True):
        del features[index]

        for doc in bow:
            del doc[index]

    len_features = len(features)

    ############################################################################


    ############################################################################
    ### OUTPUT (WRITING TEXT REPRESENTATION MATRIX)                          ###
    ############################################################################

    output_file.write(str(total_num_examples) + " " + str(len_features) + "\n")

    if args.print_features:
        for feature in features:
            output_file.write(str(feature.replace("'", "\\'")) + "\t")
    else:
        for feature_i in range(1, len_features+1):
            output_file.write("w" + str(feature_i) + "\t")

    output_file.write("class_atr\n")

    for doc_i, doc in enumerate(bow):
        for freq in doc:
            output_file.write(str(freq) + "\t")

        output_file.write(str(document_words[doc_i]["class"]) + "\n")

    output_file.close()

    ############################################################################

operation_end = time.time()
eta = operation_end-operation_start
print_progress(total_operations, total_operations, eta, final=True)
print("..................................................\n\n\n")

################################################################################

#Comparing output files:
if args.n_gram > 1:
    print("> Removing duplicated files:")
    print("..................................................")
    any_file_removed = False
    
    for n in reversed(range(2, args.n_gram+1)):
        if (filecmp.cmp(out_string + str(n), out_string + str(n-1), shallow=False)):
            any_file_removed = True
            os.remove(out_string + str(n))
            print(out_string + str(n) + " \t\t\t--> REMOVED")
    
    if not any_file_removed:
        print("- All files are different!")
    
    print("..................................................\n\n\n")

################################################################################


################################################################################

total_end = time.time()
time = format_time(total_end-total_start)
files = str(total_num_examples)
paragraphs = str(total_num_paragraphs)
sentences = str(total_num_sentences)
print("> Log:")
print("..................................................")
print("- Time: " + time)
print("- Input files: " + files)
print("- Input paragraphs: " + paragraphs)
print("- Input sentences: " + sentences)
print("..................................................\n")
log.write("\n\n> Log:\n")
log.write("\t- Time:\t\t\t" + time + "\n")
log.write("\t- Input files:\t\t" + files + "\n")
log.write("\t- Input paragraphs:\t\t" + paragraphs + "\n")
log.write("\t- Input sentences:\t\t" + sentences + "\n")
log.close()