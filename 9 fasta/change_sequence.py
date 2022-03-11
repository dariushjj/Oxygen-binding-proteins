import os
import sys
import numpy as np

fR='C:\\Users\\黄家俊\\Desktop\\9 fasta\\uniprot-name_cytoglobin3,297.fasta'
fR1='C:\\Users\\黄家俊\\Desktop\\9 fasta\\uniprot-name_erythrocruorin1,14.fasta'
fR2='C:\\Users\\黄家俊\\Desktop\\9 fasta\\uniprot-name_flavohemoprotein3,1872.fasta'
fR3='C:\\Users\\黄家俊\\Desktop\\9 fasta\\uniprot-name_hemerythrin2,59.fasta'
fR4='C:\\Users\\黄家俊\\Desktop\\9 fasta\\uniprot-name_hemocyanin2,56.fasta'
fR5='C:\\Users\\黄家俊\\Desktop\\9 fasta\\uniprot-name_hemoglobin3,2676.fasta'
fR6='C:\\Users\\黄家俊\\Desktop\\9 fasta\\uniprot-name_leghemoglobin3,45.fasta'
fR7='C:\\Users\\黄家俊\\Desktop\\9 fasta\\uniprot-name_myoglobin3,410.fasta'
fR8='C:\\Users\\黄家俊\\Desktop\\9 fasta\\uniprot-name_neuroglobin3,214.fasta'
def get_sequence(file):
    file_data=''
    with open(file,'r') as f:
        for line in f:
            if '>' not in line:
                line = line.replace('X', '')
                line = line.replace('B', '')
                line = line.replace('J', '')
                line = line.replace('O', '')
                line = line.replace('U', '')
                line = line.replace('Z', '')
            file_data+=line
    with open(file, "w") as f:
        f.write(file_data)


#get_sequence(fR)
#get_sequence(fR1)
#get_sequence(fR2)
#get_sequence(fR3)
#get_sequence(fR4)
#get_sequence(fR5)
#get_sequence(fR6)
get_sequence(fR7)
#get_sequence(fR8)
'''
my_str = 'aXbXc'
my_str = my_str.replace('X', '')
print(my_str)

'''