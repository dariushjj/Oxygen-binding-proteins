'''
generate PSSM from Oxygen proteins
'''
import sys
import subprocess
from Bio.Blast.Applications import NcbipsiblastCommandline
from Bio import SeqIO

DB = "/home/wanglei/data/nr/nr"


def main():
    for record in SeqIO.parse(sys.argv[1], format="fasta"):
        # command = "psiblast -query {0} -db {1} -num_iterations 3 -evalue 0.001 -out ./output/{2}.txt -out_ascii_pssm ./output/{3}.pssm".format(
            # str(record.seq), DB, record.id.split("|")[1], record.id.split("|")[1])
        psiblastline = NcbipsiblastCommandline(cmd="psiblast", db=DB, num_iterations=3, evalue=0.001, out="./output/{}.txt".format(
            record.id.split("|")[1]), out_ascii_pssm="./output/{}.pssm".format(record.id.split("|")[1]))
        psiblastline(record.format("fasta"))


if __name__ == "__main__":
    main()
