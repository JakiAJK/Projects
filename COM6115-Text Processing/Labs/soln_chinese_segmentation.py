"""
USE: python <PROGNAME> (options) WORDLIST-FILE INPUT-FILE OUTPUT-FILE
OPTIONS:
    -h : print this help message and exit
"""
################################################################

import sys

################################################################

MAXWORDLEN = 5

################################################################
# Command line options handling, and help

def print_help():
    progname = sys.argv[0]
    progname = progname.split('/')[-1] # strip out extended path
    help = __doc__.replace('<PROGNAME>', progname, 1)
    print('-' * 60, help, '-' * 60, file=sys.stderr)
    sys.exit(0)
    
if '-h' in sys.argv or len(sys.argv) != 4:
    print_help()

word_list_file = sys.argv[1]
input_file = sys.argv[2]
output_file = sys.argv[3]

################################################################
# READ CHINESE WORD LIST

word_set = set()

with open(word_list_file, encoding = "utf8") as words_in:
    for line in words_in:
        word_set.add(line.strip())

################################################################
# FUNCTION TO PROCESS ONE SENTENCE
# Sentence provided as a string. Result returned as a list of strings 

# def segment(sent, wordset):   # dummy definition provided, so can
#     return list(sent)         # write code for main loop first.

# Two alternative definitions provided

# Definition 1: 

def segment1(sentence, wordset):
    words = []
    sentlen = len(sentence)
    current = 0
    while current < sentlen:
        maxlen = min(sentlen - current, MAXWORDLEN)
        for i in range(maxlen, 0, -1):
            candidate = sentence[current:current+i]
            if i == 1 or candidate in wordset:
                words.append(candidate)
                current += i
                break
    return words

# Definition 2: 

def segment2(sentence, wordset):
    words = []
    while sentence:  # Empty string treated as False; non-empty string as True
        maxlen = min(len(sentence), MAXWORDLEN)
        for wlen in range(maxlen, 0, -1):
            candidate = sentence[:wlen]
            if wlen == 1 or candidate in wordset:
                words.append(candidate)
                sentence = sentence[wlen:]
                break
    return words

################################################################
# MAIN LOOP
# Read each line from input file, segment, and print to output file

with open(input_file, encoding = "utf8") as text_in, \
     open(output_file, "w", encoding = "utf8") as text_out:
    for line in text_in:
        words = segment1(line.strip(), word_set)
        print(' '.join(words), file = text_out)    

################################################################

