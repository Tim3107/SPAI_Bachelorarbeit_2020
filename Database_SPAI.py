# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 15:17:22 2020

@author: tim31
"""
import numpy as np
import sys
import threading

class Database_SPAI:            #Objekt, welches die Threading-Daten speichert und den Zugriff regelt
    
    def __init__(self,pMatrix_A,pM,pM_default,pDimension):
        
        self.Matrix_A = pMatrix_A
        self.Dimension = pDimension
        self.M = pM
        self.M_default = pM_default
        # Die einzelnen Semaphores werden erstellt. Für jeden Prozess ein einzelnes
        self.sem_M = threading.Semaphore()          
        self.sem_Matrix_A = threading.Semaphore()
        self.sem_iterations = threading.Semaphore()
        self.sem_duration = threading.Semaphore()
        self.sem_nonzero = threading.Semaphore()
        self.sem_cols = threading.Semaphore()
        self.sem_finish = threading.Semaphore()
        self.sem_col = threading.Semaphore()
        # Die Berechnungsdaten 
        self.iterations = 0
        self.duration = 0
        self.positions_nonzeros = None
        self.norm_squared_cols = None
        self.finished = 0
        
    
    def set_M(self,column,index_column,J):          #setter Methode für die Approximation an A^-1
        self.sem_M.acquire()
        self.M[J,index_column]= column
        self.sem_M.release()
        
    def get_M(self,column,index_column):            #getter Methode für die Approximation an A^-1
        self.sem.acquire()
        temp = self.M[:,index_column]
        self.sem.release()
        return temp
    
    def get_Matrix_A(self):                         #getter Methode für die Matrix_A
        self.sem_Matrix_A.acquire()
        temp = self.Matrix_A.copy()
        self.sem_Matrix_A.release()
        return temp
    
    def set_Matrix_A(self,Matrix_A):
        self.Matrix_A = Matrix_A
    
    def set_M_default(self,column,index_column):    #setter Methode für die Default Matrix M
        self.sem.acquire()
        self.M_deafult[:,index_column] = column
        self.sem.release()
        
    def get_M_default(self,column,index_column):    #getter Methode für die Default Matrix M
        self.sem.acquire()
        temp = self.M_default[:,index_column]
        self.sem.release()
        return temp
        
    def set_iterations(self):                       #setter Methode für die Anzahl der Iterationen
        self.sem_iterations.acquire()
        self.iterations = self.iterations + 1
        self.sem_iterations.release()
    
    def set_iteration(self,a):                       #setter Methode für die Anzahl der Iterationen
        self.sem_iterations.acquire()
        self.iterations = self.iterations + a
        self.sem_iterations.release()
        
    def get_iterations(self):                       #getter Methode für die Anzahl der Iterationen
        return self.iterations
    
    def set_duration(self,duration):                #setter Methode für die Dauer
        self.sem_duration.acquire()
        self.duration = duration
        self.sem_duration.release()
        
    def get_duration(self):                         #getter Methode für die Dauer
        return self.duration

    def get_positions_nonzeros(self):
        self.sem_nonzero.acquire()
        temp = self.positions_nonzeros
        self.sem_nonzero.release()
        return temp
    
    def get_norm_squared_cols(self):
        self.sem_cols.acquire()
        temp = self.norm_squared_cols
        self.sem_cols.release()
        return temp
    
    def getcolumn(self,i):
        self.sem_col.acquire()
        temp = self.M_default.getcol(i)
        self.sem_col.release()
        return temp
        
    def set_finished(self):
        self.sem_finish.acquire()
        self.finished = self.finished + 1
        self.sem_finish.release()
        
    def positions_nonzero(self):
        self.Matrix_A = self.Matrix_A.tocsr()
        Nonzero_positions = []
        for i in range(0,self.Dimension):
            index_begin = self.Matrix_A.indptr[i]
            index_end = self.Matrix_A.indptr[i+1]
            nonzeros = self.Matrix_A.indices[index_begin:index_end]
            Nonzero_positions.append(nonzeros.tolist())
        
        
        self.positions_nonzeros = Nonzero_positions
        self.Matrix_A = self.Matrix_A.tocsc()
    
    def getcol_boost(self):
        temp = self.Matrix_A@self.Matrix_A.T
        self.norm_squared_cols = temp.diagonal()
    

        