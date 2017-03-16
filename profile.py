""" Profile generator.
Each character will answer questions about themselves.
"""
__author__ = 'Alice Wong and Cecille Yang'

import json
from nltk import *
import os.path
import parseCorpus
import os
import codecs

def sample_from_dist(d):
    """Function by Sravana Reddy. Given a dictionary representing a discrete probability distribution
    (keys are atomic outcomes, values are probabilities)
    sample a key according to the distribution.
    Example: if d is {'H': 0.7, 'T': 0.3}, 'H' should be returned about 0.7 of time.
    """
    roll = numpy.random.random()
    cumul = 0
    for k in d:
        cumul += d[k]
        if roll < cumul:
            return k


def answer_question(dictionary, movie, character, file):
    """ Function to answer the fill in the blank questions
    """
    
    # Note: Some characters don't have all the dictionaries so be
    # careful in which parts of speech we choose from. Could do something like this:
    # question += sample_from_dist(dictionary[movie][char].get('NNS', dictionary[movie][char]['NN']))
    with codecs.open(file, 'a', 'utf8') as o:
        count = 1
        o.write("QUESTIONS ABOUT ME: \n")
        o.write(str(count)+". What are you passionate about? \n")
        question = "I am passionate about "
        question += sample_from_dist(dictionary[movie][character]['NN'])
        question += "."
        o.write(question+'\n\n')
        count += 1

        o.write(str(count)+". What is something or someone you couldn't live without?\n")
        question_two = "I couldn't live without my "
        question_two += sample_from_dist(dictionary[movie][character]['NN'])
        question_two += '.'
        o.write(question_two+'\n\n')
        count += 1

        o.write(str(count)+". Three words I'd describe myself with: \n")
        o.write(sample_from_dist(dictionary[movie][character]['JJ'])+", " 
            +sample_from_dist(dictionary[movie][character]['JJ']) + ", "
            +sample_from_dist(dictionary[movie][character]['JJ'])+ "\n")

        count += 1



def main():
    # If the json file does not exist yet then create it. This should only need to
    # happen once since our corpus will not be changing
    if not os.path.isfile('data/character_pos_single.json'):
        movies = parseCorpus.parseFiles()
        print "DONE MOVIE PARSING"
        parseCorpus.partsOfSpeech(movies)
        print "DONE PARTSOFSPEECH"

    # Open the json file and convert it back to a dictionary to be used
    # in answering dating profile questions
    with open('data/character_pos_single.json') as json_pos:
        pos_dict = json.load(json_pos)
        json_pos.close()

    answer_question(pos_dict)


if __name__ == '__main__':
    main()
