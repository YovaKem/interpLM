import random
import sys

in_file = open(sys.argv[1],'r')
out_file = {}
for subset in ['train','valid','test']:
    out_file[subset] = open('data/{}.{}'.format(subset,sys.argv[2]),'w')

def out_write(s):
    subset_ind = random.randint(0,9)
    if subset_ind<8:
        subset='train'
    elif subset_ind<9:
        subset='valid'
    else: subset='test'
    out_file[subset].write(s)

for line in in_file:
    out_write(line)
