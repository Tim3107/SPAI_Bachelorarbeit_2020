# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 15:56:29 2020
@author: tim31
"""
import numpy as np
import threading
from SPAI_Database import Database_SPAI
from SPAI_parallel_Thread import myThread
import time
from load_data import load_data
from scipy import sparse
import sys

def load_Matrix_A(level):
    Matrix_A = load_data(level)
    Dimension = Matrix_A.shape[0]
    M_default = sparse.eye(Dimension).tocsc()       #sparse.lil_matrix((self.Dimension,self.Dimension))
    M = sparse.lil_matrix((Dimension,Dimension))
        
    return [Matrix_A,M_default,M,Dimension]

def save_data(A,file):
    A = A.tocsc()
    indices = A.indices
    indptr  = A.indptr
    data    = A.data
    np.savez_compressed(file, data=data, indices=indices, indptr= indptr)
    print(file)
    print('Die Matrix M wurde im aktuellen Ordner unter',file,'.npz abgespeichert')
    
if __name__ == "__main__":
    
    epsilon = float(sys.argv[1])                     #obere Schranke Abbruchkriterium Residuum ||r-Am||
    max_fillin = int(sys.argv[2])                   #obere Schranke an Einträgen in den Spalten aus M
    case = int(sys.argv[3])                                 #Die Art des Abbruchkriteriums... Kombiniert?
    new_Indices = int(sys.argv[4])      #Anzahl an neuen Einträgen, die pro Iterationsschritt zu J hinzugefügt werden 
    level = int(sys.argv[5])
    file = sys.argv[6]
    
    print("Der Algorithmus wird gestartet mit den Parametern \n " "epsilon = ",epsilon,"\n max_fillin = ",max_fillin,"\n case = ",case,"\n level = ",level,"\n new_Indices = ",new_Indices)
    
    [Matrix_A,M_Default,M,Dimension] = load_Matrix_A(level)
    Matrix_A = Matrix_A.tocsc()
    Matrix_A.eliminate_zeros()
    Database = Database_SPAI(Matrix_A,M,M_Default,Dimension)
    Database.positions_nonzero()
    Database.getcol_boost()
    Threads = []
    for i in range(0,Dimension):
        Threads.append(threading.Thread(target=myThread().Start, args=(epsilon,max_fillin,case,new_Indices,M,M_Default.getcol(i),Dimension,i,Database)))
    l = len(Threads)
    time_start = time.time()
    for j in range(0,l):
        (Threads[j].start())
    
    for j in range(0,l):
        (Threads[j].join())
        
    time_end = time.time()
    time_duration = time_end - time_start
    time.sleep(2)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
    print("Es wurde ",time_duration,"Sekunden gerechnet")
    print("Es wurden ",Database.get_iterations(),"Schritte gebraucht")
    save_data(M,file)