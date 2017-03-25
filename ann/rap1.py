import numpy as np
import re
import random


def prepare_data():
    """This method prepares input positive and negative datasets as bitvectors for the Rap1 binding problem. Output: three lists of bitvectors, one containing positive samples, negative samples that are similar to positive samples, and negative examples that are randomly chosen from the fasta sequences. All bitvectors are 17 bp (34 bits) long"""
    # read in all positive data, convert to bitvectors
    pos_str = read_positives()
    pos_vec = str_to_vec(pos_str)

    # read in all negative data. then, remove false negatives from the negative fa sequences and their reverse complements. Call this new set of sequences and their reverse complements "neg_str".
    neg_str = read_negatives()
    neg_str = remove_falseneg(neg_str, pos_str)
    rc_neg_str = reverse_complement(neg_str)
    rc_neg_str = remove_falseneg(rc_neg_str, pos_str)
    neg_str = reverse_complement(rc_neg_str)
    neg_str = neg_str + rc_neg_str

    # cache interesting cases as "neg_simiar". interesting cases are those that look similar to the positive sequences (in that they contain cysteines at positions 5, 6, and 10) but are considered negative. also cache randomly chosen sequences, so that the neural net can be trained on sequences that are not similar to positive examples.
    neg_sim, neg_rand = cache_cases(neg_str)
    neg_sim_vec = str_to_vec(neg_sim)
    neg_rand_vec = str_to_vec(neg_rand)

    return pos_vec, neg_sim_vec, neg_rand_vec


def read_positives():
    "reads in positive samples as strings"
    seqs = []
    file = '/Users/cjmathy/Documents/courses/bmi203/Final-Project/ann_bmi203/rap1-lieb-positives.txt'
    with open(file, 'rb') as f:
        for seq in f:
            seqs.append(seq.strip())
    return seqs


def read_negatives():
    "reads in negative samples as strings"
    seqs = []
    file = '/Users/cjmathy/Documents/courses/bmi203/Final-Project/ann_bmi203/yeast-upstream-1k-negative.fa'
    with open(file, 'rb') as f:
        sequence = ''
        for line in f:
            if line[0] is not '>':
                sequence += line.strip()
            else:
                if sequence:
                    seqs.append(sequence)
                    sequence = ""
    return seqs


def str_to_vec(sequences):
    """converts nucleotide strings into vectors using a 2-bit encoding scheme."""
    vecs = []
    nuc2bit = {"A": (0, 0),
               "C": (0, 1),
               "T": (1, 0),
               "G": (1, 1)}
    for seq in sequences:
        vec = []
        for nuc in seq:
            vec.append(nuc2bit[nuc][0])
            vec.append(nuc2bit[nuc][1])
        vecs.append(vec)
    return vecs


def remove_falseneg(negatives, positives):
    """this method removes any negative fasta sequences that contain one of the positive sample sequences (essentially making them false negatives."""
    seqs = []
    for n in negatives:
        if not any(p in n for p in positives):
            seqs.append(n)
    return seqs


def reverse_complement(sequences):
    """returns a list of reverse complemented sequences"""
    rc = []
    complement = {'A': 'T',
                  'C': 'G',
                  'G': 'C',
                  'T': 'A'}
    for seq in sequences:
        seq = list(seq)
        seq = reversed([complement.get(nuc) for nuc in seq])
        seq = ''.join(seq)
        rc.append(seq)
    return rc


def cache_cases(sequences):
    """this method separates the negative data into two sets: those that contain the Rap1 binding signature sequence, and a set that is randomly chosen from the negative data."""

    # 1) cache negative cases that are similar to positives
    sim_cache = []
    for seq in sequences:
        matches = re.findall(r'....CC...C.......', seq)
        for match in matches:
            sim_cache.append(match)
    sim_cache = list(set(sim_cache))

    # 2) cache randomly chosen 17 bp negatives. 5 from each fa sequence (including reverse complements). there are about 30000 neg_sim samples, so this will create about 30000 neg_rand samples from the 3000 sequences and their 3000 reverse complements.
    bp = 17
    rand_cache = []
    for seq in sequences:
        for _ in xrange(5):
            i = random.randint(0, len(seq)-bp)
            substr = seq[i:i+bp]
            rand_cache.append(substr)

    return sim_cache, rand_cache


def build_training_set(pos, neg_sim, neg_rand):
    """Builds a training set using 50% positive data, and 50% negative data. Negative data consists equally of similar-to-positve and random negative sequences"""

    # we have 137 positive examples, 30000 special negative examples, and 30000 random negative examples, all 34 bits long. take 69 special negative examples and 68 random negative examples. add them to the positive examples to make our training set.

    neg = []
    for _ in xrange(69):
        i = np.random.randint(0, len(neg_sim))
        neg.append(neg_sim[i])
    for _ in xrange(68):
        i = np.random.randint(0, len(neg_rand))
        neg.append(neg_rand[i])

    Xp = np.array(pos)
    Xn = np.array(neg)
    X = np.concatenate((Xp, Xn), axis=0) # nd array, 274 x 34
    yp = np.ones((Xp.shape[0],))
    yn = np.zeros((Xn.shape[0],))
    y = np.concatenate((yp, yn), axis=0) # nd array, 34 x 1

    return X, y


def build_training_set_100(pos, neg_sim, neg_rand):
    """same as above, but allowing for some positive and negative samples to be held out as a test set"""
    neg = []
    for _ in xrange(50):
        i = np.random.randint(0, len(neg_sim))
        neg.append(neg_sim[i])
    for _ in xrange(50):
        i = np.random.randint(0, len(neg_rand))
        neg.append(neg_rand[i])

    Xp = np.array(pos)
    Xn = np.array(neg)
    X = np.concatenate((Xp, Xn), axis=0)
    yp = np.ones((Xp.shape[0],))
    yn = np.zeros((Xn.shape[0],))
    y = np.concatenate((yp, yn), axis=0)

    return X, y


def build_test_set(pos, neg_sim, neg_rand):
    """same as above, but allowing for some positive and negative samples to be held out as a test set"""
    neg = []
    for _ in xrange(19):
        i = np.random.randint(0, len(neg_sim))
        neg.append(neg_sim[i])
    for _ in xrange(18):
        i = np.random.randint(0, len(neg_rand))
        neg.append(neg_rand[i])

    Xp = np.array(pos)
    Xn = np.array(neg)
    X = np.concatenate((Xp, Xn), axis=0)
    yp = np.ones((Xp.shape[0],))
    yn = np.zeros((Xn.shape[0],))
    y = np.concatenate((yp, yn), axis=0)

    return X, y
