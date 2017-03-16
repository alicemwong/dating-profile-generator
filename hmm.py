"""Hidden Markov Model Toolkit
with the fundamental tasks --
randomly generating data,
finding the best state sequence for observation,
computing the probability of observations,
and Baum-Welch EM algorithm for parameter learning.
"""

__author__='Alice Wong'

import math
import numpy   # install this with pip or Canopy's package manager. You're probably all set if you have NLTK/matplotlib.
import numpy.random
import argparse
import codecs
import os
from collections import defaultdict

def normalize(countdict):
    """given a dictionary mapping items to counts,
    return a dictionary mapping items to their normalized (relative) counts
    Example: normalize({'a': 2, 'b': 1, 'c': 1}) -> {'a': 0.5, 'b': 0.25, 'c': 0.25}
    """
    # Do not modify this function
    total = sum(countdict.values())
    return {item: val/total for item, val in countdict.items()}

def read_arcs(filename):
    """Load parameters from file and store in nested dictionary.
    Assume probabilities are already normalized.
    Return dictionary and boolean indicating whether probabilities were provided.
    """
    # Do not modify this function
    arcs = {}
    provided = True
    for line in map(lambda line: line.split(), codecs.open(filename, 'r', 'utf8')):
        if len(line)<2:
            continue
        from_s = line[0]
        to_s = line[1]
        if len(line)==3:
            prob = float(line[2])
        else:
            prob = None
            provided = False
        if from_s in arcs:
            arcs[from_s][to_s] = prob
        else:
            arcs[from_s] = {to_s: prob}
    return arcs, provided

def write_arcs(arcdict, filename):
    """write dictionary of conditional probabilities to file
    """
    # Do not modify this function
    o = codecs.open(filename, 'w', 'utf8')
    for from_s in arcdict:
        for to_s in arcdict[from_s]:
            o.write(from_s+' '+to_s+' '+str(arcdict[from_s][to_s])+'\n')
    o.close()

def read_corpus(filename):
    # Do not modify this function
    return filter(lambda line: len(line)>0, map(lambda line: line.split(),
               codecs.open(filename, 'r', 'utf8').readlines()))

def sample_from_dist(d):
    """given a dictionary representing a discrete probability distribution
    (keys are atomic outcomes, values are probabilities)
    sample a key according to the distribution.
    Example: if d is {'H': 0.7, 'T': 0.3}, 'H' should be returned about 0.7 of time.
    """
    # Do not modify this function
    roll = numpy.random.random()
    cumul = 0
    for k in d:
        cumul += d[k]
        if roll < cumul:
            return k

class HMM:
    def __init__(self, transfile, emitfile, translock=False, emitlock=False):
        """reads HMM structure from transition and emission files, and probabilities if given.
        no error checking: assumes the files are in the correct format."""
        # Do not modify this function
        self.transitions, tprovided = read_arcs(transfile)
        self.emissions, eprovided = read_arcs(emitfile)
        self.states = self.emissions.keys()

        self.translock = translock
        self.emitlock = emitlock

        # initialize with random parameters if probs were not specified
        if not tprovided:   # at least one transition probability not given in file
            print 'Transition probabilities not given: initializing randomly.'
            self.init_transitions_random()
        if not eprovided:   # at least one emission probability not given in file
            print 'Emission probabilities not given: initializing randomly.'
            self.init_emissions_random()

    def init_transitions_random(self):
        """assign random probability values to the HMM transition parameters
        """
        # Do not modify this function
        for from_state in self.transitions:
            random_probs = numpy.random.random(len(self.transitions[from_state]))
            total = sum(random_probs)
            for to_index, to_state in enumerate(self.transitions[from_state]):
                self.transitions[from_state][to_state] = random_probs[to_index]/total

    def init_emissions_random(self):
        for state in self.emissions:
            random_probs = numpy.random.random(len(self.emissions[state]))
            total = sum(random_probs)
            for symi, sym in enumerate(self.emissions[state]):
                self.emissions[state][sym] = random_probs[symi]/total

    def generate(self, n):
        """return a list of n symbols by randomly sampling from this HMM.
        """
        # TODO: fill in
        # due at checkpoint date (ungraded)
        # Use the sample_from_dist helper function in this file

        # Initialize the first state based on the '#' state
        current = sample_from_dist(self.transitions['#'])

        # Initialize the list and generate up to n number of symbols
        # Each time using the current state to get the next state based on probability
        symbolsSample = []
        for x in range(0, n):
            symbolsSample.append(sample_from_dist(self.emissions[current]))
            # Use the dictionary of the current transition state to get the next one
            current = sample_from_dist(self.transitions[current])
        return symbolsSample

    def viterbi(self, observation):
        """given an observation as a list of symbols,
        compute and return the most likely state sequence that generated it
        using the Viterbi algorithm
        """
        # TODO: fill in
        # due at checkpoint date (ungraded)
        # Will look into a more efficient way later

        # Initialize the viterbi matrix to be tuples so that we can keep track of the
        # backpointer and the probability in the viterbi matrix
        viterbi_matrix = numpy.empty((len(self.states), len(observation)), dtype=tuple)

        # For each cell, calculate the probability the current cell based on previous cells
        for oi, obs in enumerate(observation):
            for si, state in enumerate(self.states):
                if oi == 0:
                    viterbi_matrix[si, oi] = ('#', self.transitions['#'][state] * self.emissions[state][obs])
                else:
                    maxVal = -1
                    maxPrevState = '#'

                    # Calculate all possible probabilities based on the last column probabilities
                    # But only keep track of the highest probability which will be the only one
                    # inserted into the matrix
                    for pi, prevstate in enumerate(self.states):
                        current = viterbi_matrix[pi, oi-1][1] * self.transitions[prevstate][state] \
                                  * self.emissions[state][obs]
                        if current > maxVal:
                            maxVal = current
                            maxPrevState = prevstate
                    viterbi_matrix[si, oi] = (maxPrevState, maxVal)

        # An array to keep track of all the correct transitions
        final_result = []

        # Using the last column of the matrix, find the highest probability
        # The cell of the highest probability is the last transmission of the sentence
        # Append that onto our list and then use the

        maxprob = -1
        backpointer = ''
        last_transmission = ''
        for ni, nstate in enumerate(self.states):
            prob = viterbi_matrix[ni][len(observation)-1][1]
            if maxprob < prob:
                maxprob = prob
                backpointer = viterbi_matrix[ni][len(observation)-1][0]
                last_transmission = nstate

        final_result.append(last_transmission)

        # Not sure what the efficient way of doing this is...
        # But I'm making a dictionary of the states and their respective order
        # so that I can find which index to access easily based on the backpointer
        state_dict = {}
        for n, st in enumerate(self.states):
            state_dict[st] = n

        # Start from the second to the last obs and decrement until 0.
        # Append each backpointer
        for x in range(len(observation)-2, -1, -1):
            final_result.append(backpointer)
            backpointer = viterbi_matrix[state_dict[backpointer]][x][0]

        return list(reversed(final_result))


    def forward(self, observation):
        """given an observation as a list of T symbols,
        compute and return the forward algorithm dynamic programming matrix
        of size m x T where m is the number of states.
        """
        # Do not modify this function
        forward_matrix = numpy.zeros((len(self.states), len(observation)))

        for oi, obs in enumerate(observation):
            for si, state in enumerate(self.states):
                if oi==0:
                    forward_matrix[si, oi] = self.transitions['#'][state] * self.emissions[state][obs]
                else:
                    for pi, prevstate in enumerate(self.states):
                        forward_matrix[si, oi] += forward_matrix[pi, oi-1] * self.transitions[prevstate][state]

                    forward_matrix[si, oi] *= self.emissions[state][obs]  # factor out common emission prob

        return forward_matrix

    def forward_probability(self, observation):
        """return probability of observation, computed with forward algorithm.
        """
        # Do not modify this function
        forward_matrix = self.forward(observation)
        # sum of forward probabilities in last time step, over all states
        return sum(forward_matrix[:, len(observation)-1])

    def backward(self, observation):
        """given an observation as a list of T symbols,
        compute and return the backward algorithm dynamic programming matrix
        of size m x T where m is the number of states.
        """
        # TODO: fill in
        # due at checkpoint date (ungraded)
        # Follow the example in the slides from Mar 07 as a guideline.

        # Create the backward matrix
        if(len(observation)==0):
            print "observation length is 0"
            return
        backward_matrix = numpy.zeros((len(self.states), len(observation)))

        # For each observation going backwards (i.e. go from the last obs to the first)
        # Calculate the prob of each cell:
        # Initialize the column of the last obs to be 1 (i.e. the last column of the matrix)
        # Otherwise, calculate the current cell probability by summing up all probabilities
        # based on the next observation (i.e. the next column):
        # Sum up: (P( Next transmission | Current transmission) * P ( Next Observation|Next transmission) * Next Cell
        # Note when I say next, I mean the cell to the right.
        for oi, obs in reversed(list(enumerate(observation))):
            for si, state in enumerate(self.states):
                if oi == len(observation)-1:
                    backward_matrix[si][oi] = 1
                else:
                    total = 0.0
                    for pi, prevstate in enumerate(self.states):
                        t_state = self.transitions.get(state, dict())
                        t = t_state.get(prevstate, 0.0)
                        total += backward_matrix[pi][oi+1] * t *self.emissions[prevstate][observation[oi+1]]
                    backward_matrix[si][oi] = total

        return backward_matrix


    def backward_probability(self, observation):
        """return probability of observation, computed with backward algorithm.
        """
        # Do not modify this function
        backward_matrix = self.backward(observation)
        backprob = 0.0  # total probability
        for si, state in enumerate(self.states):
            # print self.transitions['#'][state]
            # print self.emissions[state][observation[0]]
            # print backward_matrix[si, 0]

            # prob of transitioning from # to state and giving out observation[0]
            transit = self.transitions.get("#", dict())
            t = transit.get(state, 0.0)

            backprob += t\
            * self.emissions[state][observation[0]]\
             * backward_matrix[si, 0]

        return backprob

    def maximization(self, emitcounts, transcounts):
        """M-Step: set self.emissions and self.transitions
        conditional probability parameters to be the normalized
        counts from emitcounts and transcounts respectively.
        Do not update if self.emissions if the self.emitlock flag is True,
        or self.transitions if self.translock is True.
        """
        # TODO: fill in
        # note that this is a short and simple function
        if self.emitlock == False:
            for k in emitcounts.keys():
                self.emissions[k] = normalize(emitcounts[k])

        if self.translock == False:
            for k in transcounts.keys():
                self.transitions[k] = normalize(transcounts[k])


    def expectation(self, corpus):
        """E-Step: given a corpus, which is a list of observations,
        calculate the expected number of each transition and emission,
        as well as the log likelihood of the observations under the current parameters.
        return a list containing the log likelihood (float),
        expected emission counts, and expected transition counts (dictionaries that can be passed to maximization).
        """
        # TODO: fill in
        # follow the Excel sheet example and slides from Mar 10 to fully understand the E step

        # OVERALL - Variables to be returned in a tuple at the end
        log_likelihood = 0.0
        emitcounts = {}
        transcounts = {}

        # FOR EACH OBSERVATION: Calculate the log_likelihood, emitcounts and transounts
        for li, l in enumerate(corpus):
            # LOG LIKELIHOOD - Since forward and backward probability are the same we only need to calculate
            # one of them
            #if we have an empty list, we don't want to do all this work
            if (len(l)>0):
                backward = self.backward_probability(l)
                log_likelihood += math.log(backward,2)

                # Alpha and Beta Matrix - The forward and backward matrices respectively
                alpha = self.forward(l)
                beta = self.backward(l)

                # EMISSION COUNTS - For each state, calculate the emission counts and add it to the
                # dictionary under the correct key
                for si, s in enumerate(self.states):
                    emitcounts[s] = emitcounts.get(s, {})
                    for ei, element in enumerate(l):
                        emitcounts[s][element] = emitcounts[s].get(element, 0.0)+((alpha[si][ei]*beta[si][ei])/backward)

                # TRANSITION COUNTS - For each possible transition, calculate the counts
                keys_without = [x for x in self.transitions.keys() if x != '#']

                # Case: all keys except #
                for ki, k in enumerate(keys_without):
                    transcounts[k] = transcounts.get(k, {})
                    for vki, v_key in enumerate(self.transitions[k].keys()):
                        state_one = k
                        state_two = v_key
                        for oi, obs in enumerate(corpus[li][1:]):
                            t = (alpha[ki][oi]*self.transitions[state_one][state_two]
                                 *self.emissions[state_two][corpus[li][oi+1]]*beta[vki][oi+1])/backward
                            transcounts[state_one][state_two] = transcounts[state_one].get(state_two, 0.0) + t

                # Case: # key
                transcounts['#'] = transcounts.get('#', {})
                for v_ki, v_k in enumerate(self.transitions['#']):
                    num = (alpha[v_ki][0]*beta[v_ki][0]) / backward
                    transcounts['#'][v_k] = transcounts['#'].get(v_k, 0) + num

        return (log_likelihood, emitcounts, transcounts)


    def expectation_maximization(self, corpus, convergence):
        """given a corpus,
        apply EM to learn the HMM parameters that maximize the corpus likelihood.
        stop when the log likelihood changes less than the convergence threhshold,
        and return the final log likelihood.
        """
        # Do not modify this function
        old_ll = -10**210    # approximation to negative infinity
        log_likelihood = -10**209   # higher than old_ll

        while log_likelihood-old_ll > convergence:
            old_ll = log_likelihood
            log_likelihood, emitcounts, transcounts = self.expectation(corpus) # E Step
            self.maximization(emitcounts, transcounts)  # M Step
            print 'LOG LIKELIHOOD:', log_likelihood,
            print 'DIFFERENCE:', log_likelihood-old_ll
        print 'CONVERGED'

        return log_likelihood

def main():
    # Do not modify this function
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('corpusfile', type=str, help='corpus of observations')
    parser.add_argument('paramfile', type=str, help='basename of the HMM parameter file')
    parser.add_argument('function',
                        type=str,
                        choices = ['v', 'p', 'em', 'g'],
                        help='best state sequence (v), total probability of observations (p), parameter learning (em), or random generation (g)?')
    # optional arguments for EM
    parser.add_argument('--convergence', type=float, default=0.1, help='convergence threshold for EM')
    parser.add_argument('--restarts', type=int, default=0, help='number of random restarts for EM')
    parser.add_argument('--translock', type=bool, default=False, help='should the transition parameters be frozen during EM training?')
    parser.add_argument('--emitlock', type=bool, default=False, help='should the emission parameters be frozen during EM training?')

    args = parser.parse_args()

    # initialize model and read data
    model = HMM(args.paramfile+'.trans', args.paramfile+'.emit', args.translock, args.emitlock)

    if args.function == 'v':
        corpus = read_corpus(args.corpusfile)
        outputfile = os.path.splitext(args.corpusfile)[0]+'.tagged'
        with codecs.open(outputfile, 'w', 'utf8') as o:
            for observation in corpus:
                viterbi_path = model.viterbi(observation)
                o.write(' '.join(viterbi_path)+'\n')

    elif args.function == 'p':
        corpus = read_corpus(args.corpusfile)
        outputfile = os.path.splitext(args.corpusfile)[0]+'.dataprob'
        with open(outputfile, 'w') as o:
            for observation in corpus:
                o.write(str(model.backward_probability(observation))+'\n')

    elif args.function == 'em':
        corpus = read_corpus(args.corpusfile)

        # keep track of the parameters that maximize the log likelihood over different initializations
        best_log_likelihood = model.expectation_maximization(corpus, args.convergence)
        best_transitions = model.transitions
        best_emissions = model.emissions

        #restart and run EM with random initializations
        for restart in range(args.restarts):
            print "RESTART", restart+1, 'of', args.restarts
            if not model.translock:
                model.init_transitions_random()
                print 'Re-initializing transition probabilities'
            if not model.emitlock:
                model.init_emissions_random()
                print 'Re-initializing emission probabilities'

            log_likelihood = model.expectation_maximization(corpus, args.convergence)
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_transitions = model.transitions
                best_emissions = model.emissions

        #write the winning models
        print 'The best log likelihood was', best_log_likelihood
        outputfile = args.paramfile+'.trained'
        write_arcs(best_transitions, outputfile+'.trans')
        write_arcs(best_emissions, outputfile+'.emit')

    elif args.function == 'g':
        # write randomly generated sentences
        with codecs.open(args.corpusfile, 'w', 'utf8') as o:
            for _ in range(20):
                o.write(' '.join(model.generate(10))+'\n')  # sentences with 10 words

    print "DONE"

if __name__=='__main__':
    main()
