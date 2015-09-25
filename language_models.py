import os
import random
import math
import time
from collections import defaultdict
from utils import *

TRIGRAM = "count_trigram"
BIGRAM = "count_bigram"
UNIGRAM = "count_unigram"
TRIGRAM_A = "trigram_A"
BIGRAM_A = "bigram_A"
WORD_SET = "word_set"

START_SYMBOL = "START"
STOP_SYMBOL = "STOP"
DISCOUNTED = 0.5


class LanguageModels:
    count_trigram = {}
    count_bigram = {}
    count_unigram = {}
    trigram_A = defaultdict(set)
    bigram_A = defaultdict(set)
    word_set = set()
    total_words = 1

    def __init__(self, training_corpus_dir, dict_dir):
        self.training_corpus_dir = training_corpus_dir
        self.dict_dir = dict_dir

        self.word_set_dir = self.dict_dir + WORD_SET
        self.trigram_dir = self.dict_dir + TRIGRAM
        self.bigram_dir = self.dict_dir + BIGRAM
        self.unigram_dir = self.dict_dir + UNIGRAM
        self.trigram_A_dir = self.dict_dir + TRIGRAM_A
        self.bigram_A_dir = self.dict_dir + BIGRAM_A

        print("initializing BrownCorpus object for '%s'..." % training_corpus_dir)
        if os.path.isdir(self.dict_dir):
            print('loading trained data...')
            start_reading_from_files = time.time()

            self.word_set = get_trained_data(self.word_set_dir)
            self.count_trigram = get_trained_data(self.trigram_dir)
            self.count_bigram = get_trained_data(self.bigram_dir)
            self.count_unigram = get_trained_data(self.unigram_dir)
            self.trigram_A = get_trained_data(self.trigram_A_dir)
            self.bigram_A = get_trained_data(self.bigram_A_dir)

            end_reading_from_files = time.time()
            print('===> time for reading from files: ~%ss' % round(end_reading_from_files - start_reading_from_files))
        else:
            print('training...')
            start_training = time.time()

            os.makedirs(self.dict_dir)
            for filename in os.listdir(self.training_corpus_dir):
                with open(self.training_corpus_dir + filename, 'r+') as file:
                    lines = file.readlines()
                    for line in lines:
                        if line.strip():
                            line = line.lower() + " " + STOP_SYMBOL
                            penult_word = START_SYMBOL
                            last_word = START_SYMBOL

                            self.count_unigram[START_SYMBOL] = self.count_unigram.get(START_SYMBOL, 0) + 1
                            self.count_bigram[START_SYMBOL, START_SYMBOL] = self.count_bigram.get((START_SYMBOL,
                                                                                                   START_SYMBOL), 0) + 1

                            for word in line.split():
                                self.word_set.add(word)
                                self.count_trigram[penult_word, last_word, word] = self.count_trigram.get((penult_word,
                                                                                                           last_word,
                                                                                                           word), 0) + 1
                                self.count_bigram[last_word, word] = self.count_bigram.get((last_word, word), 0) + 1
                                self.count_unigram[word] = self.count_unigram.get(word, 0) + 1

                                self.trigram_A[penult_word, last_word].add(word)
                                self.bigram_A[last_word].add(word)

                                penult_word = last_word
                                last_word = word

                    file.close()

            end_training = time.time()
            print('===> time for training: ~%ss' % round(end_training - start_training))

            print('saving trained data to files...')
            start_saving_to_files = time.time()

            save_to_file(self.word_set, self.word_set_dir)
            save_to_file(self.count_trigram, self.trigram_dir)
            save_to_file(self.count_bigram, self.bigram_dir)
            save_to_file(self.count_unigram, self.unigram_dir)
            save_to_file(self.trigram_A, self.trigram_A_dir)
            save_to_file(self.bigram_A, self.bigram_A_dir)

            end_saving_to_files = time.time()
            print('===> time for saving to files: ~%ss' % round(end_saving_to_files - start_saving_to_files))

        self.total_words = sum(self.count_unigram.values()) - self.count_unigram[START_SYMBOL]

    def bigram_discounted_count(self, last_word, word):
        return self.count_bigram[last_word, word] - DISCOUNTED

    def bigram_alpha(self, last_word):
        return 1 - sum(self.bigram_discounted_model(last_word, w) for w in self.bigram_A[last_word])

    def bigram_discounted_model(self, last_word, word):
        return self.bigram_discounted_count(last_word, word) / self.count_unigram.get(last_word, 1)

    def unigram_mle(self, word):
        unique_count = len(self.count_unigram) - 1
        return self.count_unigram.get(word, unique_count) / self.total_words  # TODO unknown word ???

    def bigram_back_off_model(self, last_word, word):
        if word in self.bigram_A[last_word]:
            return self.bigram_discounted_model(last_word, word)
        else:
            return self.unigram_back_off_model(last_word, word)

    def unigram_back_off_model(self, last_word, word):
        unigram_mle_others = 1 - sum(self.unigram_mle(w) for w in self.bigram_A[last_word])
        return self.bigram_alpha(last_word) * self.unigram_mle(word) / unigram_mle_others

    def trigram_discounted_count(self, penult_word, last_word, word):
        return self.count_trigram[penult_word, last_word, word] - DISCOUNTED

    def trigram_alpha(self, penult_word, last_word):
        return 1 - sum(self.trigram_discounted_model(penult_word, last_word, w)
                       for w in self.trigram_A[penult_word, last_word])

    def trigram_discounted_model(self, penult_word, last_word, word):
        return self.trigram_discounted_count(penult_word, last_word, word) / self.count_bigram[penult_word, last_word]

    def trigram_back_off_model(self, penult_word, last_word, word):
        if word in self.trigram_A[penult_word, last_word]:
            return self.trigram_discounted_model(penult_word, last_word, word)
        else:
            bigram_mle_others = 1 - sum(self.bigram_back_off_model(last_word, w)
                                        for w in self.trigram_A[penult_word, last_word])

            return self.trigram_alpha(penult_word, last_word) * self.bigram_back_off_model(last_word, word) / \
                bigram_mle_others

    def get_sentence_prob(self, sentence):
        sentence = sentence.lower() + " STOP"
        prob = 1.0
        penult = 'START'
        last = 'START'
        for word in sentence.split():
            prob *= self.trigram_back_off_model(penult, last, word)
            penult = last
            last = word
        return prob

    def evaluate(self, test_dir):
        print("computing the perplexity for '%s'..." % test_dir)
        start_time = time.time()
        sum_log = 0.0
        total_words = 0
        for filename in os.listdir(test_dir):
            with open(test_dir + filename, 'r+') as file:
                lines = file.readlines()
                for line in lines:
                    if line.strip():
                        total_words += len(line.rstrip('\n').split()) + 1
                        sent_prob = self.get_sentence_prob(line.rstrip('\n'))
                        if sent_prob != 0:
                            sum_log += math.log2(sent_prob)

                file.close()

        l = sum_log / total_words

        perplexity = 2**(-l)
        end_time = time.time()

        print('elapsed time: ~%ss' % round(end_time-start_time))

        return perplexity

    def generate_random_sentence(self, size=26):
        words = list(self.bigram_A['START'])

        seed_word = words[random.randint(0, len(words) - 1)]
        next_word = random.choice(list(self.bigram_A[seed_word]))

        w1, w2 = seed_word, next_word
        gen_words = []
        for i in range(size):
            gen_words.append(w1)
            next_candidates = list(self.trigram_A[w1, w2])
            if len(next_candidates) == 0:
                break
            prob, w2, w1 = max([(self.trigram_back_off_model(w1, w2, w), w, w2) for w in next_candidates])

            if w2 == '.':
                break
        gen_words.append(w2)

        print('===> random sentence: ', ' '.join(gen_words))

