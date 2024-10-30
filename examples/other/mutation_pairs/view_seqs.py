from Bio import SeqIO
from sys import argv

fname = argv[1]

with open(fname, 'r') as fopen:
    records = [r for r in SeqIO.parse(fopen, 'fasta')]

wt = records[0]
records = records[1:]

wtA, wtB, wtC = wt.seq.split('/')

for rec in records:
    print('=' * 50)
    print(rec.name)
    A, B, C = rec.seq.split('/')
    for ref, des, label in zip((wtA, wtB, wtC), (A, B, C), ("A", "B", "C")):
        for refAA, desAA, n in zip(ref, des, range(len(des))):
            if refAA != desAA:
                changelog = label + str(n + 1) + ':' + refAA + '-to-' + desAA
                print(changelog)
