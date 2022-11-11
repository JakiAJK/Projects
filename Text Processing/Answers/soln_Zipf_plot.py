
import sys
import re
import pylab as p
import nltk

nltk.download('brown')

# Import data

from nltk.corpus import brown

# Get a list (strictly, interator-over) all the tokens of the Brown Corpus

brown_words = brown.words()

################################################################
# Count items (words/bigrams) in data

non_word = re.compile('^\W+$') # has only non-alphabetic chars

print('<counting ...>', file=sys.stderr)
sys.stderr.flush()

counts = {}

items = 'words'
for wd in brown_words:
    if non_word.search(wd): # skip over punctuation tokens
        continue
    wd = wd.lower()
    if wd not in counts:
        counts[wd] = 0
    counts[wd] += 1

# items = 'bigrams'
# last_wd = 'DUMMY'
# for wd in brown_words:
#     if non_word.search(wd):
#         continue
#     wd = wd.lower()
#     bigram = (last_wd, wd)
#     if bigram not in counts:
#         counts[bigram] = 0
#     counts[bigram] += 1
#     last_wd = wd

print('<done>', file=sys.stderr)
sys.stderr.flush()

################################################################
# Sort items (words/bigrams) / print top N

sorted_items = sorted(counts, reverse = True, key = lambda v:counts[v])

print()
print('TYPES: ', len(sorted_items))
print('TOKENS:', sum(counts.values()))
print()

top_N = 20
c = 0
for item in sorted_items[:top_N]:
    c += 1
    print('[%2d]' % c, item, '=', counts[item])

################################################################
# Get list of item frequencies, sorted into descending order

freqs = sorted(counts.values(), reverse = True)

################################################################
# Plot freq vs. rank

ranks = range(1, len(freqs)+1)

p.figure()
p.plot(ranks, freqs)
p.title('freq vs rank (%s)' % items)

################################################################
# Plot cumulative freq vs. rank

cumulative = list(freqs) # makes copy of freqs list

for i in range(len(cumulative) - 1):
    cumulative[i + 1] += cumulative[i]

p.figure()
p.plot(ranks, cumulative)
p.title('cumulative freq vs rank (%s)' % items)

################################################################
# Plot log-freq vs. log-rank

logfreqs = [p.log(freq) for freq in freqs]
logranks = [p.log(rank) for rank in ranks]

p.figure()
p.plot(logranks, logfreqs)
p.title('log-freq vs log-rank (%s)' % items)

################################################################
# Display figures

p.show()

################################################################

