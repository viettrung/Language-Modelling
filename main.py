from language_models import LanguageModels


def init_on_brown_corpus():
    return LanguageModels('corpus/brown/training/', 'dictionary/brown/')


def init_on_reuters_corpus():
    return LanguageModels('corpus/reuters/training/', 'dictionary/reuters/')


def get_sentence_prob():
    sentence = input('Please enter your sentence: ')
    prob = lm.get_sentence_prob(sentence)
    print('===> result = %s' % prob)
    input('press any key to continues...')


def evaluate(test_dir):
    ppl = lm.evaluate(test_dir)
    print('===> perplexity = %s' % ppl)
    input('press any key to continues...')

default_training = input('Please select your corpus (1: brown, 2: reuters): ')
if default_training == '1':
    lm = init_on_brown_corpus()
else:
    lm = init_on_reuters_corpus()

while True:
    print("\n============= TRAINED SET: '%s' =============" % lm.training_corpus_dir)
    print('1. Get probability for a sentence')
    print('2. Compute the perplexity on the BROWN corpus')
    print('3. Compute the perplexity on the REUTERS corpus')
    print('4. Generate random sentence')
    print('5. Switch to the BROWN corpus')
    print('6. Switch to the REUTERS corpus')

    user_choice = input('Your choice (0 to exit): ')
    print()

    if user_choice == '1':
        get_sentence_prob()
    elif user_choice == '2':
        evaluate('corpus/brown/test/')
    elif user_choice == '3':
        evaluate('corpus/reuters/test/')
    elif user_choice == '4':
        lm.generate_random_sentence()
        input('press any key to continues...')
    elif user_choice == '5':
        lm = init_on_brown_corpus()
    elif user_choice == '6':
        lm = init_on_reuters_corpus()
    elif user_choice == '0':
        break
