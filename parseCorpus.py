""" Parsing Corpus
File is meant to parse through the corpus once so that when we run our program
we can just read in a file instead of recalculating everything each time.
"""

__author__ = 'Alice Wong and Cecille Yang'

import codecs
import json
from nltk import *
import hmm
import os
import profile
from random import randint


def normalize(countdict):
    """Function by Sravana Reddy, but modified slightly so that we cast all numbers to be floats
    Given a dictionary mapping items to counts,
    return a dictionary mapping items to their normalized (relative) counts
    Example: normalize({'a': 2, 'b': 1, 'c': 1}) -> {'a': 0.5, 'b': 0.25, 'c': 0.25}
    """
    total = sum(countdict.values())
    return {item: float(val) / float(total) for item, val in countdict.items()}

def writeEmitFile(character, lines, movie):
    """Write the emissions file given a character name, their tokenized lines in the form of a
    nessted list, and a movie title. Smooth before normalizing."""

    emitfolder = os.getcwd()+'/data/emit/'
    if not os.path.exists(emitfolder):
	   os.makedirs(emitfolder)
	
	
    freqDict = dict()
    cfile = file(emitfolder+character+".emit", "w")
    #for every line and every word, get its part of speech and make a dictionary if it doesn't exist
    for line in lines:
        newline = pos_tag(line)
        for word in newline:
            freqDict[word[1]] = freqDict.get(word[1], dict()) #make dict if it doesn't exist already
    #second iteration: add all words to every POS dict
    for line in lines:
        newline = pos_tag(line)
        for word in newline:
            for pos in freqDict:
                freqDict[pos][word[0]] = freqDict[pos].get(word[0], 0) 
            freqDict[word[1]][word[0]] = freqDict[word[1]].get(word[0], 0) + 1
    #smooth
    for pos in freqDict:
        for key in freqDict[pos]:
            freqDict[pos][key] = freqDict[pos][key]+.1 #add smoothing
        freqDict[pos] = normalize(freqDict[pos])
    #write to file
    for pos in freqDict:
        for word in freqDict[pos]:
            cfile.write(pos+" "+word+" "+str(freqDict[pos][word])+"\n")
    return freqDict

def writeTransFile(character, lines, movie, freqDict):
    """Given the character, a nested list of tokenized lines, a movie title, and
    a dictionary of part of speech frequencies, write a transition file"""

    transfolder = os.getcwd()+'/data/trans/'
    if not os.path.exists(transfolder):
       os.makedirs(transfolder)

    cfile = file(transfolder+character+".trans", "w")


    transDict = dict()
    transDict['#'] = dict()
    #Need all of the parts of speech, not just the ones that have nonzero transitions
    for pos in freqDict:
        transDict[pos] = dict()
        for pos2 in freqDict:
            transDict[pos][pos2] = 0
        transDict['#'][pos] = 0

    #Go through lines and words, and add their transition counts
    for line in lines:
        newline = pos_tag(line)
        for wi, word in enumerate(newline):
            if wi == 0:
                transDict['#'] = transDict.get('#', dict())
                transDict['#'][word[1]] = transDict['#'].get(word[1], 0) + 1 
            elif wi < len(newline)-1:
                transDict[newline[wi][1]] = transDict.get(newline[wi][1], dict())
                transDict[newline[wi][1]][newline[wi+1][1]] = transDict[newline[wi][1]].get(newline[wi+1][1], 0) + 1
            #otherwise we are at the last word and don't do anything

    #smooth
    for line in lines:
        for pos in transDict:
            for key in transDict[pos]:
                transDict[pos][key] = transDict[pos][key]+.1 #add smoothing
            transDict[pos] = normalize(transDict[pos])

    #write to file
    for pos in transDict:
        for pos2 in transDict[pos]:
            cfile.write(pos+" "+pos2+" "+str(transDict[pos][pos2])+"\n")

def buildHMM(name, lineList, numRestarts, convergence):
    """Build the HMM like we did in class. Write to a separate profile prof_folder
    the modified transition and emit files"""
    path = os.getcwd()+"/data/"
    model = hmm.HMM(path+"trans/"+name+'.trans', path+"emit/"+name+'.emit', False, False)

    # keep track of the parameters that maximize the log likelihood over different initializations
    best_log_likelihood = model.expectation_maximization(lineList, convergence)
    best_transitions = model.transitions
    best_emissions = model.emissions

    #restart and run EM with random initializations
    for restart in range(numRestarts):
        print "RESTART", restart+1, 'of', numRestarts
        if not model.translock:
            model.init_transitions_random()
            print 'Re-initializing transition probabilities'
        if not model.emitlock:
            model.init_emissions_random()
            print 'Re-initializing emission probabilities'

        log_likelihood = model.expectation_maximization(lineList, convergence)
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_transitions = model.transitions
            best_emissions = model.emissions

    #write the winning models
    print 'The best log likelihood was', best_log_likelihood
    outputfile = name+'.trained'
    hmm.write_arcs(best_transitions, "tenthings/trans/"+name+'.trans')
    hmm.write_arcs(best_emissions, "tenthings/emit/"+name+'.emit')


def parseFiles():
    """ Parse the files of Cornell's movie corpus to create a dictionary of movie titles
    which is a dictionary of characters, which is a list of lines
    :return: dictionary
    :param: path to the files - each cluster of files should be denoted by their movie title name

    """
    print "STARTING METHOD PARSEFILES"
    movies = dict()
    wordCount = dict()
    movieTitles = codecs.open("data/movie_titles_metadata_single.txt", 'r', 'latin-1').readlines()
    print "PARSING MOVIE TITLES"
    for movie in movieTitles:
        movieArray = movie.split("+++$+++")
        movies[movieArray[0].strip()] = dict()
    movieChars = codecs.open("data/movie_characters_metadata_single.txt", 'r', 'latin-1').readlines()
    print "PARSING CHARACTERS"
    for char in movieChars:
        charArray = char.split(u"+++$+++")
        movies[charArray[2].strip()][charArray[1].strip()] = []
    movieLines = codecs.open("data/movie_lines_single.txt", 'r', 'latin-1').readlines()
    print "PARSING MOVIE LINES"
    for line in movieLines:
        lineArray = line.split("+++$+++")
        lineString = lineArray[4].lower()
        sentenceList = sent_tokenize(lineString)
        sList = []
        for i in sentenceList:
            sList.append(word_tokenize(i))
        movies[lineArray[2].strip()][lineArray[3].strip()] += sList

    for movie in movies:
        for character in movies[movie]:
            freqDict = writeEmitFile(character+movie, movies[movie][character], movie)
            writeTransFile(character+movie, movies[movie][character], movie, freqDict)
            print character, "EMIT AND TRANS DONE"
    
    return movies

def partsOfSpeech(dictionary):
    """ Given a dictionary of movie title, character, and lines (nested), produce a dictionary of
    movie title, character, and normalized part of speech counts"""
    print "STARTING PARTS OF SPEECH METHOD"
    charPOS = dict()
    movies = dictionary.keys()
    for movie in movies:
        print movie
        charPOS[movie] = dict()
        characters = dictionary[movie].keys()
        for char in characters:
            print char
            charPOS[movie][char] = dict()
            for line in dictionary[movie][char]:  # go line by line
                for l in line:  # lines may be multiple sentences
                    ps = pos_tag(l)
                    for p in ps:  # categorize parts of speech in new dictionary
                        charPOS[movie][char][p[1]] = charPOS[movie][char].get(p[1], dict())
                        charPOS[movie][char][p[1]][p[0]] = charPOS[movie][char][p[1]].get(p[0], 0) + 1
            pos = charPOS[movie][char].keys()
            for p in pos:
                charPOS[movie][char][p] = normalize(charPOS[movie][char][p])

    with open('data/character_pos.json', 'w') as fp:
        json.dump(charPOS, fp)


def main():
    # parse the files to create a dictionary of movie titles which is a dictionary of
    # # characters, which is a list of lines
    prof_folder = os.getcwd()+'/data/datingprofiles/'
    if not os.path.exists(prof_folder):
        os.makedirs(prof_folder)

    movies = parseFiles()
    print "DONE MOVIE PARSING"

    for movie in movies:
        for character in movies[movie]:
            if (not (os.path.isfile('data/emit/'+character+movie+".emit") and os.path.isfile('data/trans/'+character+movie+".trans"))):
                freqDict = writeEmitFile(character, movies[movie][character], movie)
                writeTransFile(character, movies[movie][character], movie, freqDict)
                buildHMM(character+movie, movies[movie][character], 1, 0.001)

            name = character+movie
            p = os.getcwd()+"/data/"
            model = hmm.HMM(os.getcwd()+"/data/trans/"+name+'.trans', os.getcwd()+"/data/emit/"+name+'.emit', False, False)
            with codecs.open(p+'/datingprofiles/'+name+"_datingprofile.txt", 'w', 'utf8') as o:
                o.write("Name: "+character+"\n")
                o.write("\nINTRODUCTION: \n")
                for _ in range(20):
                    o.write(' '.join(model.generate(randint(3,7)))+'\n')
                o.write("\n")
            with open('data/character_pos_single.json') as json_pos:
                pos_dict = json.load(json_pos)
                json_pos.close()
            profile.answer_question(pos_dict, movie, character, p+'/datingprofiles/'+name+"_datingprofile.txt")



if __name__ == '__main__':
    main()
