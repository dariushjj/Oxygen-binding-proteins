'''
generate PSSM from Oxygen proteins
eg: args
  1.fa
  2. out dir
'''
import sys
import os
import time
import subprocess
from Bio.Blast.Applications import NcbipsiblastCommandline
from Bio import SeqIO

#

workdrir = "/home/wanglei/data/Oxygen"
OUT = sys.argv[2]

# check dir exist
if not os.path.exists(OUT):
    os.mkdir(OUT)


def main():
    for record in SeqIO.parse(sys.argv[1], format="fasta"):
        SeqIO.write(record, os.path.join(
            OUT, record.id.split("_")[1] + ".fa"), "fasta")
        commmand = "qsub -N {0} -d {1} -v outfile={2},query={3},output={4} psiblast.sh".format(record.id.split("_")[1], workdrir, os.path.join(
            OUT, record.id.split("_")[1]+".pssm"), os.path.join(OUT, record.id.split("_")[1] + ".fa"),os.path.join(OUT, record.id.split("_")[1] + ".txt"))
        process = subprocess.Popen(commmand, shell=True)
        # process.communicate(input=record.format("fasta"))


if __name__ == "__main__":
    main()
