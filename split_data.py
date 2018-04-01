import random
import sys

in_file = open('{}/full.txt.lower'.format(sys.argv[1]),'r')
out_file = {}
for subset in ['train','valid','test']:
    out_file[subset] = open('{}/{}.{}'.format(sys.argv[1],subset,sys.argv[2]),'w')

def out_write(s):
    subset_ind = random.randint(0,99)
    if subset_ind<89:
        subset='train'
    elif subset_ind<94:
        subset='valid'
    else: subset='test'
    out_file[subset].write(s)

for line in in_file:
    out_write(line)
