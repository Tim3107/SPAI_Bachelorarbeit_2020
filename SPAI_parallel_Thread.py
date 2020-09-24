# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 15:13:15 2020
@author: tim31
"""
import threading
import numpy as np
from scipy import linalg as sp






class myThread (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def Start(self,epsilon,max_fillin,case_Abbruchkriterium,new_Indices,M,Default_m,Dimension,i,Database):
        if(case_Abbruchkriterium == 0):                #case 0: nur Epsilon
            self.start_case_0(epsilon,new_Indices,M,Default_m,Dimension,i,Database)   
        elif(case_Abbruchkriterium == 1):                                         #case 1: nur max fill-in
            self.start_case_1(max_fillin,new_Indices,M,Default_m,Dimension,i,Database)
        elif((case_Abbruchkriterium == 2)):                                             #case 2: Epsilon und max fill-in
            self.start_case_2(epsilon,max_fillin,new_Indices,M,Default_m,Dimension,i,Database)               

    def start_case_0(self,epsilon,new_Indices,M,m,Dimension,i,Database):
        Simulation = True
        J = self.Initialize_J(m)                                                 #Menge J wird erstellt. 
        I = self.Initialize_I(J,i,Dimension,Database)                                                     #Auf Basis der Menge J kann nun auch die Menge I initialisiert werden

        e_i_tilde = self.unit_vector_tilde(i,I)
        [Q,R] = self.compute_QR_decomposition(I,J,Database)               #Einmal wird die große QR-Zerlegung berechnet
        
        m_tilde = self.R_backwards_solver(i,Q,R,len(J),I)
        residual = (Database.get_Matrix_A()[I,:])[:,J]@(m_tilde) - (e_i_tilde)         #Das Residuum kann auch mit A[I,J] berechnet werden, da die restlichen Einträge nur auf Nulllen treffen
        current_residual = self.retransform_residual(residual,I,Dimension)
        current_norm_residual = np.linalg.norm(current_residual)
        Database.set_iterations()
        while((current_norm_residual > epsilon) and Simulation):                                   
            J_tilde = self.new_Indizes_J_Minimization_problem(new_Indices,current_residual,I,J,residual,Database)                           #beste Kandidaten werden zu J_tilde hinzugefügt
            I_tilde = self.set_I_tilde(J_tilde,I,J,Database)
            
            if(self.control_lists(I_tilde,J_tilde)):
                
                [Q,R] = self.compose_QR_decomposition(I,J,I_tilde,J_tilde,Q,R,Database)                              #Die neuen Bestandteile Q und R der QR-Zerlegung werden geupdatet (keine komplett neue QR-Zerlegung)
                [I,J] = self.unite_sets(I,J,I_tilde,J_tilde)
                e_i_tilde = self.unit_vector_tilde(i,I)
                m_tilde  = self.R_backwards_solver(i,Q,R,len(J),I)
                residual = (Database.get_Matrix_A()[I,:])[:,J]@(m_tilde) - (e_i_tilde)         #Das Residuum kann auch mit A[I,J] berechnet werden, da die restlichen Einträge nur auf Nulllen treffen
                current_residual = self.retransform_residual(residual,I,Dimension)
                current_norm_residual = np.linalg.norm(current_residual)
            else:
                Simulation = False
            Database.set_iterations()
        self.set_m_to_M(m_tilde,i,J,Database)                                                            #Die aktuelle Näherung an die i-te Spalte der inversen Matrix wird in M abgespeichert

        Database.set_finished()

    
    def start_case_1(self,max_fillin,new_Indices,M,m,Dimension,i,Database):
        Simulation = True
        J = self.Initialize_J(m)                                                 #Menge J wird erstellt. 
        I = self.Initialize_I(J,i,Dimension,Database)                                                     #Auf Basis der Menge J kann nun auch die Menge I initialisiert werden
        e_i_tilde = self.unit_vector_tilde(i,I)
        [Q,R] = self.compute_QR_decomposition(I,J,Database)               #Einmal wird die große QR-Zerlegung berechnet
        
        m_tilde = self.R_backwards_solver(i,Q,R,len(J),I)
        residual = (Database.get_Matrix_A()[I,:])[:,J]@(m_tilde) - (e_i_tilde)         #Das Residuum kann auch mit A[I,J] berechnet werden, da die restlichen Einträge nur auf Nulllen treffen
        current_residual = self.retransform_residual(residual,I,Dimension)
        Database.set_iterations()
        while((len(J) + new_Indices) <= max_fillin and Simulation):
            J_tilde = self.new_Indizes_J_Minimization_problem(new_Indices,current_residual,I,J,residual,Database)                           #beste Kandidaten werden zu J_tilde hinzugefügt
            I_tilde = self.set_I_tilde(J_tilde,I,J,Database)
            
            if(self.control_lists(I_tilde,J_tilde)):
                
                [Q,R] = self.compose_QR_decomposition(I,J,I_tilde,J_tilde,Q,R,Database)                              #Die neuen Bestandteile Q und R der QR-Zerlegung werden geupdatet (keine komplett neue QR-Zerlegung)
                [I,J] = self.unite_sets(I,J,I_tilde,J_tilde)
                e_i_tilde = self.unit_vector_tilde(i,I)
                m_tilde  = self.R_backwards_solver(i,Q,R,len(J),I)
                residual = (Database.get_Matrix_A()[I,:])[:,J]@(m_tilde) - (e_i_tilde)         #Das Residuum kann auch mit A[I,J] berechnet werden, da die restlichen Einträge nur auf Nulllen treffen
                current_residual = self.retransform_residual(residual,I,Dimension)
            else:
                Simulation = False
            Database.set_iterations()
        self.set_m_to_M(m_tilde,i,J,Database)                                                            #Die aktuelle Näherung an die i-te Spalte der inversen Matrix wird in M abgespeichert

        Database.set_finished()
        
    def start_case_2(self,epsilon,max_fillin,new_Indices,M,m,Dimension,i,Database):
        Simulation = True
        J = self.Initialize_J(m)                                                 #Menge J wird erstellt. 
        I = self.Initialize_I(J,i,Dimension,Database)                                                     #Auf Basis der Menge J kann nun auch die Menge I initialisiert werden
        e_i_tilde = self.unit_vector_tilde(i,I)
        [Q,R] = self.compute_QR_decomposition(I,J,Database)               #Einmal wird die große QR-Zerlegung berechnet
        
        m_tilde = self.R_backwards_solver(i,Q,R,len(J),I)
        residual = (Database.get_Matrix_A()[I,:])[:,J]@(m_tilde) - (e_i_tilde)         #Das Residuum kann auch mit A[I,J] berechnet werden, da die restlichen Einträge nur auf Nulllen treffen
        current_residual = self.retransform_residual(residual,I,Dimension)
        current_norm_residual = np.linalg.norm(current_residual)
        Database.set_iterations()
        while(((len(J) + new_Indices) <= max_fillin) and (current_norm_residual > epsilon) and Simulation):
            J_tilde = self.new_Indizes_J_Minimization_problem(new_Indices,current_residual,I,J,residual,Database)                           #beste Kandidaten werden zu J_tilde hinzugefügt
            I_tilde = self.set_I_tilde(J_tilde,I,J,Database)
            
            if(self.control_lists(I_tilde,J_tilde)):
                
                [Q,R] = self.compose_QR_decomposition(I,J,I_tilde,J_tilde,Q,R,Database)                              #Die neuen Bestandteile Q und R der QR-Zerlegung werden geupdatet (keine komplett neue QR-Zerlegung)
                [I,J] = self.unite_sets(I,J,I_tilde,J_tilde)
                e_i_tilde = self.unit_vector_tilde(i,I)
                m_tilde  = self.R_backwards_solver(i,Q,R,len(J),I)
                residual = (Database.get_Matrix_A()[I,:])[:,J]@(m_tilde) - (e_i_tilde)         #Das Residuum kann auch mit A[I,J] berechnet werden, da die restlichen Einträge nur auf Nulllen treffen
                current_residual = self.retransform_residual(residual,I,Dimension)
                current_norm_residual = np.linalg.norm(current_residual)
            else:
                Simulation = False
            Database.set_iterations()
        self.set_m_to_M(m_tilde,i,J,Database)                                                            #Die aktuelle Näherung an die i-te Spalte der inversen Matrix wird in M abgespeichert

        Database.set_finished()
        
    def unit_vector_tilde(self,index,I):
        e_i_tilde = np.zeros(len(I))    
        index_in_I = I.index(index)                        #Position der index-Zeile in der Menge I
        e_i_tilde[index_in_I] = 1
        return e_i_tilde
    
    def Initialize_J(self,m):                      #Indizes der Nichtnullelemente der Spalte m sollen zu J hinzugefügt werden

        J = m.indices.tolist()
        return J
            
    def Initialize_I(self,J,index,Dimension,Database):
        sliced = Database.get_Matrix_A()[:,J]
        temp = np.unique(sliced.indices).tolist()
        if index not in temp:
            temp.append(index)
        return temp
    
    def retransform_residual(self,residual,I,Dimension):
        return_vec = np.zeros((Dimension,1))
        (return_vec[I,0]) = residual 
        return return_vec
            
    def set_J_tilde(self,current_residual,Database,J):                                          #Diese Methode setzt die Kandidaten für eine Expansion der Menge J
        J_tilde = []
        
        vector_non_zero_residual = current_residual.nonzero()[0]
        temp = Database.get_positions_nonzeros()
        for i in vector_non_zero_residual:
            J_tilde.extend(temp[i])
            
        J_tilde = np.unique(J_tilde)
        return [np.setdiff1d(J_tilde,J)]
        

    def unite_sets(self,I,J,I_tilde,J_tilde):                                    #Die Mengen I und J werden jeweils mit den Mengen I_tilde und J_tilde vereinigt
            
        I.extend(I_tilde)
        J.extend(J_tilde)
        return [I,J]
    
    def array_of_candidates(self,J_tilde,current_residual,l,I,J,residual,Database):        #Array aus Index-Werte Paaren wird erstellt
    
    
        array_kandidaten_und_Werte = np.zeros((l,2))               #Matrix, die sowohl Kandidat als auch zugehörigen Wert speichert
        
        
        i = 0         
        rAej_vector = residual.T@(Database.get_Matrix_A()[I,:])[:,J_tilde]
        temp = Database.get_norm_squared_cols()
        
        for j in J_tilde:
                                         
        
            array_kandidaten_und_Werte[i][1] = j                                                          #Kandidat wird auf Position 1 geschrieben (Python---> 0)
            rAej = rAej_vector[i]
    
            inner_Aej = temp[j]
            array_kandidaten_und_Werte[i][0] = - (rAej*rAej)/inner_Aej   #Wert des Kandidaten wird auf Position 0 gespeichert(Python ---> 1)
        
            i = i+1
        return [array_kandidaten_und_Werte,i]
    
    
    def new_Indizes_J_Minimization_problem(self,new_Indices,current_residual,I,J,residual,Database): #neue Indizes werden über 1-dim Minproblem bestimmt
    
    
        [J_tilde] = self.set_J_tilde(current_residual,Database,J)
    
        l = len(J_tilde)
        
    
        [array_kandidaten_und_Werte,i] = self.array_of_candidates(J_tilde,current_residual,l,I,J,residual,Database)
    
        int_kandidaten = []
    
        array_kandidaten_und_Werte = array_kandidaten_und_Werte[array_kandidaten_und_Werte[:,0].argsort()]                            #Es wird nach dem kleinsten Residuum sortiert
        if (new_Indices >= i):                    #weil es weniger Kandidaten geben könnte als maximal möglich wären, würden alle hinzugefügt werden. Im Fall großer Matrizen eher unwahrscheinlich
    
            for k in range(0,i):
                int_kandidaten.append(int(array_kandidaten_und_Werte[k][1]))       
        
        else:
            for k in range(0,new_Indices):
                int_kandidaten.append(int(array_kandidaten_und_Werte[k][1]))
    
        J_tilde = np.sort(int_kandidaten,0)                 #Die Indizes werden noch sortiert
    
        return J_tilde.tolist() 
    
    def set_I_tilde(self,J_tilde,I,J,Database):             #ausgehend von J_tilde wird I_tilde erstellt

        J_new = J + J_tilde

        sliced = Database.get_Matrix_A()[:,J_new]

        I_tilde = np.unique(sliced.indices)
        I_tilde = np.setdiff1d(I_tilde,I).tolist()
        return I_tilde
    
    def set_m_to_M(self,m,index_column,J,Database):   
                                                                #Der Vektor m, der die Approximation an eine Spalte von M darstellt wird wieder auf die "normale" Dimension gebracht und dann in M implementiert
    
        Database.set_M(m,index_column,J)         #Die Approximation an die index_column Spalte wird zu M hinzugefügt

        
                    
    def R_backwards_solver(self,index_k,Q,R,l,I):                                   #Problem: index_k ist Position der 1 des Einheitsvektors. Diese muss jedoch noch durch die Menge I umgewandelt werden
        index_k = I.index(index_k)
        m = Q[index_k,0:l]                                  #Multiplikation Q^T e_k nicht nötig, weil das der k-ten Zeile von Q entspricht
        m = sp.solve_triangular(R,m)
        return m
    
    def compute_QR_decomposition(self,I,J,Database):

        n_2 = len(J)

        [Q,R_ganz] = np.linalg.qr((Database.get_Matrix_A()[I,:])[:,J].toarray(),"complete")
        R = R_ganz[0:n_2,0:n_2]
        
        return [Q,R] 
    
        
    def compose_QR_decomposition(self,I,J,I_tilde,J_tilde,Q,R,Database):        
            
        n_1 = len(I)
        n_2 = len(J)
        n_1_tilde = len(I_tilde)
        n_2_tilde = len(J_tilde)
           
        Test = Q.T@(Database.get_Matrix_A()[I,:])[:,J_tilde]
        
        Q1 = np.zeros((n_1 + n_1_tilde,n_1 + n_1_tilde))
        Q2 = np.zeros((n_1 + n_1_tilde,n_1 + n_1_tilde))
            
        Q1[0:n_1,0:n_1] = Q                                                 #Die alte Q-Matrix wird wiederbenutzt
        Q1[n_1:n_1 + n_1_tilde,n_1:n_1 + n_1_tilde] = np.eye(n_1_tilde)
            
        Q2[0:n_2,0:n_2] = np.eye(n_2)
            
    
        R_neu = np.zeros((n_2_tilde + n_2,n_2_tilde + n_2))
            
        R_neu[0:n_2,0:n_2] = R                              #der linke obere R-Block ist noch der der alten QR-Zerlegung
            
        B_1 = Test[0:n_2,:]
            
    
        B_2 = np.zeros((n_1 + n_1_tilde - n_2,n_2_tilde))
            
        B_2[0:(n_1-n_2),0:n_2_tilde] = Test[n_2:n_1,:]
        B_2[(n_1-n_2):(n_1-n_2+n_1_tilde),0:n_2_tilde] = (Database.get_Matrix_A()[I_tilde,:])[:,J_tilde].toarray()

        [Q_1,R_1] = np.linalg.qr(B_2,"complete")            #Die neue kleine QR-Zerlegung
            
        Q2[n_2:n_1 + n_1_tilde,n_2:n_1 + n_1_tilde] = Q_1
        R_neu[n_2:n_2 + n_2_tilde,n_2:n_2 + n_2_tilde] = R_1[0:n_2_tilde,0:n_2_tilde]
        
        R_neu[0:n_2,n_2:n_2 + n_2_tilde] = B_1

        Q = Q1@Q2

        return [Q,R_neu]
        
    def control_lists(self,I,J):                     #kontrolliert, ob eine Menge leer ist und damit die weitere Berechnung unnötig ist
        if ((len(I) == 0) and (len(J) == 0)):
            return False
        return True