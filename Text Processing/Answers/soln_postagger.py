"""
USE: python <PROGNAME> (options) 
OPTIONS:
    -h : print this help message and exit
    -d FILE : use FILE as data to create a new lexicon file
    -t FILE : apply lexicon to test data in FILE
"""
################################################################

import sys, re, getopt

################################################################
# Command line options handling, and help

opts, args = getopt.getopt(sys.argv[1:], 'hd:t:')
opts = dict(opts)

def print_help():
    progname = sys.argv[0]
    progname = progname.split('/')[-1] # strip out extended path
    help = __doc__.replace('<PROGNAME>', progname, 1)
    print('-' * 60, help, '-' * 60, file=sys.stderr)
    sys.exit()
    
if '-h' in opts:
    print_help()

if len(args) > 0:
    print("\n** ERROR: no arg files - only options! **", file=sys.stderr)
    printHelp()

if '-d' not in opts:
    print("\n** ERROR: must specify training data file (opt: -d FILE) **", file=sys.stderr)
    printHelp()

################################################################

# Function to split up a line of "Brill format" data into a list
# of word/tag pairs. 

def parse_line(line):
    wdtags = line.split()
    wdtagpairs = []
    for wdtag in wdtags:
        parts = wdtag.split('/')
        wdtagpairs.append((parts[0], parts[1]))
    return wdtagpairs

####################

word_tag_counts = {}
    # t_his is main data structure of lexicon - a two-level
    # dictionary, mapping {words->{tags->counts}}

print('<reading data for new lexicon ....>', file=sys.stderr)
with open(opts['-d']) as data_in:
    for line in data_in:
        for (wd, tag) in parse_line(line):
            if wd not in word_tag_counts:
                word_tag_counts[wd] = {}
            if tag in word_tag_counts[wd]:
                word_tag_counts[wd][tag] += 1
            else:
                word_tag_counts[wd][tag] = 1
print('<done>', file=sys.stderr)

################################################
# ANALYSE word-tag-count dictionary, to compute:
# -- proportion of types that have more than one tag
# -- accuracy naive tagger would have on the training data
# -- most common tags globally

tag_counts = {}
ambiguous_types = 0
ambiguous_tokens = 0
all_types = len(word_tag_counts)
all_tokens = 0
correct_tokens = 0

for wd in word_tag_counts:
    values = word_tag_counts[wd].values()
    if len(values) > 1:
        ambiguous_types += 1
        ambiguous_tokens += sum(values)
    correct_tokens += max(values)
    all_tokens += sum(values)
    for t, c in word_tag_counts[wd].items():
        if t in tag_counts:
            tag_counts[t] += c
        else:
            tag_counts[t] = c

print('Proportion of word types that are ambiguous: %5.1f%% (%d / %d)' % \
        ((100.0 * ambiguous_types) / all_types, ambiguous_types, all_types), file=sys.stderr)

print('Proportion of tokens that are ambiguous in data: %5.1f%% (%d / %d)' % \
        ((100.0 * ambiguous_tokens) / all_tokens, ambiguous_tokens, all_tokens), file=sys.stderr)

print('Accuracy of naive tagger on training data: %5.1f%% (%d / %d)' % \
        ((100.0 * correct_tokens) / all_tokens, correct_tokens, all_tokens), file=sys.stderr)

tags = sorted(tag_counts, key=lambda x:tag_counts[x], reverse=True)

print('Top Ten Tags by count:', file=sys.stderr)
for tag in tags[:10]:
    count = tag_counts[tag]
    print('   %9s %6.2f%% (%5d / %d)' % \
          (tag, (100.0 * count) / all_tokens, count, all_tokens), file=sys.stderr)

################################################
# Function to 'guess' tag for unknown words

digitRE = re.compile('\d')
jj_ends_RE = re.compile('(ed|us|ic|ble|ive|ary|ful|ical|less)$')
        
# NOTE: if you uncomment the 'return' at the start of the following 
# definition, the score achieved will be that where all unknown words 
# are tagged *incorrectly* (as UNK). Uncommenting instead the third 
# 'return', will yield the score where the default tag for unknown
# words is NNP. Otherwise, the definition attempts to guess the 
# correct tags for unknown words based on their suffix or other 
# characteristics. 

def tag_unknown(wd):
    # return 'UNK'
    # return 'NN'
    # return 'NNP'
    if wd[0:1].isupper():
        return 'NNP'
    if '-' in wd:
        return 'JJ'
    if digitRE.search(wd):
        return 'CD'
    if jj_ends_RE.search(wd):
        return 'JJ'
    if wd.endswith('s'):
        return 'NNS'
    if wd.endswith('ly'):
        return 'RB'
    if wd.endswith('ing'):
        return 'VBG'

################################################
# Apply naive tagging method to test data, and score performance

if '-t' in opts:
    
    # Compute 'most common' tag for each known word - store in maxtag dictionary
    maxtag = {}
    for wd in word_tag_counts:
        tags = sorted(word_tag_counts[wd], key=lambda x:word_tag_counts[wd][x], reverse=True)
        maxtag[wd] = tags[0]

    print('<tagging test data ....>', file=sys.stderr)

    # Tag each word of test data, and score
    test = open(opts['-t'], 'r')
    alltest = 0
    correct = 0
    for line in test:
        for wd, truetag in parse_line(line):
            if wd in maxtag:
                newtag = maxtag[wd]
            else:
                newtag = tag_unknown(wd)
            alltest += 1
            if newtag == truetag:
                correct += 1

    print('<done>', file=sys.stderr)
            
    print("Score on test data: %5.1f%% (%5d / %5d)" % \
          ((100.0*correct)/alltest, correct, alltest), file=sys.stderr)
    
################################################

