# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 15:56:29 2020

@author: tim31
"""
import numpy as np
import threading
from Database_SPAI import Database_SPAI
from Thread_SPAI_September import myThread
import time
from load_data import load_data
from scipy import sparse

def load_Matrix_A(level):
    Matrix_A = load_data(level)
    Dimension = Matrix_A.shape[0]
    M_default = sparse.eye(Dimension).tocsc()       #sparse.lil_matrix((self.Dimension,self.Dimension))
    M = sparse.lil_matrix((Dimension,Dimension))
        
    return [Matrix_A,M_default,M,Dimension]

if __name__ == "__main__":
    
    epsilon = 0.1                                   #obere Schranke Abbruchkriterium Residuum ||r-Am||
    max_fillin = 4                                  #obere Schranke an Einträgen in den Spalten aus M
    case_Abbruchkriterium = 0                       #Die Art des Abbruchkriteriums... Kombiniert?
    Anzahl_neue_Indizes_J_pro_Iteration = 2         #Anzahl an neuen Einträgen, die pro Iterationsschritt zu J hinzugefügt werden 

    level = 1
    [Matrix_A,M_Default,M,Dimension] = load_Matrix_A(level)
    Matrix_A = Matrix_A.tocsc()
    Matrix_A.eliminate_zeros()
    Database = Database_SPAI(Matrix_A,M,M_Default,Dimension)
    Database.positions_nonzero()
    Database.getcol_boost()
    Threads = []
    for i in range(0,Dimension):
        Threads.append(threading.Thread(target=myThread().Start, args=(epsilon,max_fillin,case_Abbruchkriterium,Anzahl_neue_Indizes_J_pro_Iteration,M,M_Default.getcol(i),Dimension,i,Database)))
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
    np.savez_compressed('Matrix_M123', a=Database.M, allow_pickle=True)
    
    
    