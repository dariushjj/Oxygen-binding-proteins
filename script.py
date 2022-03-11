# import click
import glob
import os


import sys
import numpy as np
import math
import re
# import fileinput 
import time
GROUP=20


def pssm_ac_cal(PSSM, g, l):
    """
    输入PSSM、GROUP、蛋白质序列的长度，返回GROUP×20的PSSM_AC矩阵
    """
    PSSM_AC = np.array([ [0.0] * 20 ] * g)
 
    for pg in range(g):
        l_g = l - pg - 1
        for pj in range(20):
            sum_jl = 0.0
            for i in range(l):
                sum_jl += PSSM[i][pj]
            sum_jl /= l
         
            pssm_acjg = 0.0
            for i in range(l_g):
                pssm_acjg += (PSSM[i][pj]-sum_jl) * (PSSM[i+pg+1][pj]-sum_jl)
        pssm_acjg /= l_g
        PSSM_AC[pg][pj] = pssm_acjg
    return PSSM_AC
def scale(x, max, min):
    """
    把矩阵中最大的数和最小的数变为1和-1,其他数按以下比例缩放
    """
    try:
        return ((x-min)/(max-min))
    except Exception:
        print("ZeroDivisionError, max-min:\s", max-min)
        
def pssm(fi,output):    #pssm(fi, 't4pssm.txt')     fi==./t4pssm/t4_1.pssm
    Amino_vec = "ARNDCQEGHILKMFPSTWYV"    #20个基本氨基酸
     
    PSSM = []            #存储PSSM的特征向量矩阵
    ACC = [ [0.0] * 20 ] * 20
    seq_cn = 0
    for line, strin in enumerate(open(fi,mode='r').readlines()):
        if line > 2:
            str_vec = strin.split()[1:22]    #切片截取该行的第１个至第21个字符存入向量str_vec中
        #print str_vec   #['M', '-2', '-3', '-4', '-5', '-3', '-2', '-4', '-4', '-3', '0', '3', '-3', '8', '-1', '-4', '-3', '-2', '-3', '-2', '0']等525个向量
            if len(str_vec) == 0:
                break
            PSSM.append(list(map(int, str_vec[1:])))    #将每个向量中的氨基酸符号去掉，其他字符转换为int类型，并全部存入PSSM特征向量矩阵    
            seq_cn += 1      
            ACC[Amino_vec.find(str_vec[0])] = list(map(sum, zip(list(map(int, str_vec[1:])), ACC[Amino_vec.find(str_vec[0])])))    #这句最长，从赋值号(=)右边开始逐步分析
 
    """
    1. Amino_vec.index(str_vec[0])　返回本行氨基酸符号在Amino_vec中的位置
    2. ACC[Amino_vec.index(str_vec[0])] 初始值均为20维向量[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    3. map(int, str_vec[1:])　将每个向量中的氨基酸符号去掉，其他字符转换为int类型
    4. zip(map(int, str_vec[1:]), ACC[Amino_vec.index(str_vec[0])])　返回一个tuple列表，形如[(-2,0.0),(-3,0.0),……,(-2,0.0),(0,0.0)]
    5. map(sum, zip(map(int, str_vec[1:]), ACC[Amino_vec.index(str_vec[0])]))　将tuple列表中各元素转换为sum类型,即求和
    6. 循环迭代，最后ACC(20维)为：
    [[85.0, -27.0, -42.0, -58.0, -59.0, -14.0, -15.0, -25.0, -46.0, -53.0, -55.0, -20.0, -32.0, -75.0, -32.0, 10.0, -17.0, -88.0, -57.0, -28.0], [-23.0, 90.0, -3.0, -15.0, -80.0, 24.0, -3.0, -52.0, -10.0, -45.0, -42.0, 19.0, -31.0, -63.0, -44.0, -17.0, -10.0, -68.0, -32.0, -48.0], ……　, [-5.0, -33.0, -36.0, -34.0, -36.0, -30.0, -26.0, -58.0, -49.0, 23.0, 11.0, -19.0, -12.0, -36.0, -36.0, -31.0, -5.0, -72.0, -35.0, 42.0]] 
    """  
    ACC_np = np.array(ACC)    #转换成数组
    ACC_np = np.divide(ACC_np, seq_cn)    #将每一个元素除以525
    amcnt_max = np.amax(ACC_np)    #找出最大的元素0.28380952381
    amcnt_min = np.amin(ACC_np)    ##找出最小的元素-0.274285714286
     
    vfunc = np.vectorize(scale) #矢量化scale函数
    """
    numpy中的vectorize函数可以将scale函数转换成可以接受向量参数的函数
    """
    ACC_np = vfunc(ACC_np, amcnt_max,amcnt_min)    #调用scale函数，将矩阵范围控制在[-1,1]之间
    ACC_shp = np.shape(ACC_np)    #(20,20)
    PSSM = np.array(PSSM)
    PSSM_AC = pssm_ac_cal(PSSM, GROUP, seq_cn)      #调用函数pssm_ac_cal(PSSM, g, l)   
    """
    pssm_ac_cal(PSSM, g, l)函数，三个参数：1.PSSM矩阵；2.20;3.蛋白质序列长度
    返回10×20的矩阵PSSM_AC
    """               
    PSSM_shp = np.shape(PSSM_AC)    #(10, 20)
    # file_out = open(output,'a')    #打开文件t4pssm.txt，如果没有则创建一个；ａ表示add,以追加的方式打开
    np.savetxt(output, [np.concatenate((np.reshape(ACC_np, (ACC_shp[0] * ACC_shp[1], )), np.reshape(PSSM_AC, (PSSM_shp[0] * PSSM_shp[1], ))))], delimiter=",")    #这句虽然比较长，但是逻辑比较简单清晰
    """
    1. reshape函数将矩阵的行、列、维重新调整为200×1的矩阵.
    2. concatenate函数将两个矩阵连接起来成为一个新矩阵.
    3. savetxt函数将矩阵保存为txt格式文件，分隔符为",".
    """     

def main(dir, out):
    if not os.path.exists(out):
        os.makedirs(out)
    for line in glob.glob(dir + '/*.pssm'):
        pssm(line, os.path.join(out, os.path.split(line)[1].strip(".pssm") + ".feature"))


if __name__ == "__main__":
    '''
    eg: python script.py xxx(absolute path) yyy(随意)
    '''
    main(sys.argv[1], sys.argv[2])
