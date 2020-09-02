# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:53:07 2020

@author: tim31
"""
        
import numpy as np
from scipy.sparse import linalg as sp
from scipy import sparse
from scipy import linalg as sclin
import time
from load_data import load_data
import matplotlib.pyplot as plt




def Start(Matrix_A,epsilon,max_fillin,case_Abbruchkriterium,Anzahl_neue_Indizes_J_pro_Iteration,M_default,M):
    start_time = time.time()

    if(case_Abbruchkriterium == 0):                #case 0: nur Epsilon
        [iterations,M] = start_case_0(Matrix_A,epsilon,Anzahl_neue_Indizes_J_pro_Iteration,M_default,M)   
    elif(case_Abbruchkriterium == 1):                                         #case 1: nur max fill-in
        [iterations,M] = start_case_1(Matrix_A,max_fillin,Anzahl_neue_Indizes_J_pro_Iteration,M_default,M)
    elif((case_Abbruchkriterium == 2)):                                             #case 2: Epsilon und max fill-in
        [iterations,M] = start_case_2(Matrix_A,epsilon,max_fillin,Anzahl_neue_Indizes_J_pro_Iteration,M_default,M)
    end_time = time.time()
    time_duration = end_time-start_time
        
    return [iterations,time_duration,M]
        
def start_case_0(Matrix_A,epsilon,Anzahl_neue_Indizes_J_pro_Iteration,M_default,M):
    iterations = 0
    Dimension = Matrix_A.shape[0]
    [Positions_nonzero,Matrix_A] = positions_nonzeros(Matrix_A,Dimension)
    norm_squared_cols = getcol_boost(Matrix_A,Dimension)
    for i in range(0,Dimension):
        print(i,"---------------- --------- --------- --------- --------- --------- ---------")
        Simulation = True
        m = M_default.getcol(i)                         #m Defaultspalte
        
        J = Initialize_J(m)                                                 #Menge J wird erstellt. In jedem i-Schritt anders, weil default-M unterschiedliche Spalten hat
        I = Initialize_I(Matrix_A,J,i)                                                     #Auf Basis der Menge J kann nun auch die Menge I initialisiert werden
        e_i_tilde = unit_vector_tilde(i,I)                                             #muss noch in die richtige Form gebracht werden, nur Indizes aus Menge I wichtig 
        [Q,R] = compute_QR_decomposition(I,J,Matrix_A)               #Einmal wird die große QR-Zerlegung berechnet
        
        m_tilde = R_backwards_solver(i,Q,R,len(J),I)
        
        residual = (Matrix_A[I,:])[:,J]@m_tilde - e_i_tilde        #evtl Matrix_A[I,J]@m und dann vom i-ten eintrag noch 1 abziehen für Effizienz
        current_residual = retransform_residual(residual,I,Dimension)
        
        current_norm_residual = np.linalg.norm(current_residual)
        
        iterations = iterations +1
        print(iterations)
        while((current_norm_residual > epsilon) and Simulation):                                   #neue Kandidaten für eine Expansion werden geschaffen
            J_tilde = new_Indizes_J_Minimization_problem(Matrix_A,Anzahl_neue_Indizes_J_pro_Iteration,current_residual,J,Positions_nonzero,norm_squared_cols,I,residual)                           #beste Kandidaten werden zu J_tilde hinzugefügt
            I_tilde = set_I_tilde(Matrix_A,J_tilde,I,J)
            if(control_lists(I_tilde,J_tilde)):
                [Q,R] = compose_QR_decomposition(I,J,I_tilde,J_tilde,Matrix_A,Q,R)                              #Die neuen Bestandteile Q und R der QR-Zerlegung werden geupdatet (keine komplett neue QR-Zerlegung)
                [I,J] = unite_sets(I,J,I_tilde,J_tilde)
                e_i_tilde = unit_vector_tilde(i,I)
                m_tilde  = R_backwards_solver(i,Q,R,len(J),I)
                residual = (Matrix_A[I,:])[:,J]@m_tilde - e_i_tilde          #evtl Matrix_A[I,J]@m und dann vom i-ten eintrag noch 1 abziehen für Effizienz
                current_residual = retransform_residual(residual,I,Dimension)
                current_norm_residual = np.linalg.norm(residual)
            else:
                Simulation = False
            iterations = iterations +1
        print(iterations,"Iterationen")
        M = set_m_to_M(m_tilde,i,M,J)                                                            #Die aktuelle Näherung an die i-te Spalte der inversen Matrix wird in M abgespeichert
    return [iterations,M]
    
def start_case_1(Matrix_A,max_fillin,Anzahl_neue_Indizes_J_pro_Iteration,M_default,M):
    iterations = 0
    Dimension = Matrix_A.shape[0]
    [Positions_nonzero,Matrix_A] = positions_nonzeros(Matrix_A,Dimension)
    norm_squared_cols = getcol_boost(Matrix_A,Dimension)
    for i in range(0,Dimension):
        print(i,"---------------- --------- --------- --------- --------- --------- ---------")
        Simulation = True
        m = M_default.getcol(i)                         #m Defaultspalte
        
        J = Initialize_J(m)                                                 #Menge J wird erstellt. In jedem i-Schritt anders, weil default-M unterschiedliche Spalten hat
        I = Initialize_I(Matrix_A,J,i)                                                     #Auf Basis der Menge J kann nun auch die Menge I initialisiert werden
        print
        e_i_tilde = unit_vector_tilde(i,I)
                                                     #muss noch in die richtige Form gebracht werden, nur Indizes aus Menge I wichtig 
        [Q,R] = compute_QR_decomposition(I,J,Matrix_A)               #Einmal wird die große QR-Zerlegung berechnet
        
        m_tilde = R_backwards_solver(i,Q,R,len(J),I)
        
        
        residual = (Matrix_A[I,:])[:,J]@m_tilde - e_i_tilde        #evtl Matrix_A[I,J]@m und dann vom i-ten eintrag noch 1 abziehen für Effizienz
        current_residual = retransform_residual(residual,I,Dimension)
        iterations = iterations +1
        print(iterations)
        while((len(J) + Anzahl_neue_Indizes_J_pro_Iteration) <= max_fillin and Simulation):
            J_tilde = new_Indizes_J_Minimization_problem(Matrix_A,Anzahl_neue_Indizes_J_pro_Iteration,current_residual,J,Positions_nonzero,norm_squared_cols,I,residual)                           #beste Kandidaten werden zu J_tilde hinzugefügt
            I_tilde = set_I_tilde(Matrix_A,J_tilde,I,J)
            if(control_lists(I_tilde,J_tilde)):
                [Q,R] = compose_QR_decomposition(I,J,I_tilde,J_tilde,Matrix_A,Q,R)                              #Die neuen Bestandteile Q und R der QR-Zerlegung werden geupdatet (keine komplett neue QR-Zerlegung)
                [I,J] = unite_sets(I,J,I_tilde,J_tilde)
                e_i_tilde = unit_vector_tilde(i,I)
                m_tilde  = R_backwards_solver(i,Q,R,len(J),I)
                residual = (Matrix_A[I,:])[:,J]@m_tilde - e_i_tilde          #evtl Matrix_A[I,J]@m und dann vom i-ten eintrag noch 1 abziehen für Effizienz
                current_residual = retransform_residual(residual,I,Dimension)
            else:
                Simulation = False
            iterations = iterations +1
        print(iterations,"Iterationen")
        M = set_m_to_M(m_tilde,i,M,J)                                                            #Die aktuelle Näherung an die i-te Spalte der inversen Matrix wird in M abgespeichert
    return [iterations,M]

def start_case_2(Matrix_A,epsilon,max_fillin,Anzahl_neue_Indizes_J_pro_Iteration,M_default,M):
    iterations = 0
    Dimension = Matrix_A.shape[0]
    [Positions_nonzero,Matrix_A] = positions_nonzeros(Matrix_A,Dimension)
    norm_squared_cols = getcol_boost(Matrix_A,Dimension)
    for i in range(0,Dimension):
        print(i,"---------------- --------- --------- --------- --------- --------- ---------")
        Simulation = True
        m = M_default.getcol(i)                         #m Defaultspalte
        
        J = Initialize_J(m)                                                 #Menge J wird erstellt. In jedem i-Schritt anders, weil default-M unterschiedliche Spalten hat
        I = Initialize_I(Matrix_A,J,i)                                                     #Auf Basis der Menge J kann nun auch die Menge I initialisiert werden
        e_i_tilde = unit_vector_tilde(i,I)
                                                     #muss noch in die richtige Form gebracht werden, nur Indizes aus Menge I wichtig 
        [Q,R] = compute_QR_decomposition(I,J,Matrix_A)               #Einmal wird die große QR-Zerlegung berechnet
        
        m_tilde = R_backwards_solver(i,Q,R,len(J),I)
        
        
        residual = (Matrix_A[I,:])[:,J]@m_tilde - e_i_tilde        #evtl Matrix_A[I,J]@m und dann vom i-ten eintrag noch 1 abziehen für Effizienz
        current_residual = retransform_residual(residual,I,Dimension)
        
        current_norm_residual = np.linalg.norm(current_residual)
        
        iterations = iterations +1
        print(iterations)
        while(((len(J) + Anzahl_neue_Indizes_J_pro_Iteration) <= max_fillin) and (current_norm_residual > epsilon) and Simulation):
            J_tilde = new_Indizes_J_Minimization_problem(Matrix_A,Anzahl_neue_Indizes_J_pro_Iteration,current_residual,J,Positions_nonzero,norm_squared_cols,I,residual)                           #beste Kandidaten werden zu J_tilde hinzugefügt
            I_tilde = set_I_tilde(Matrix_A,J_tilde,I,J)
            if(control_lists(I_tilde,J_tilde)):
                [Q,R] = compose_QR_decomposition(I,J,I_tilde,J_tilde,Matrix_A,Q,R)                              #Die neuen Bestandteile Q und R der QR-Zerlegung werden geupdatet (keine komplett neue QR-Zerlegung)
                [I,J] = unite_sets(I,J,I_tilde,J_tilde)
                e_i_tilde = unit_vector_tilde(i,I)
                m_tilde  = R_backwards_solver(i,Q,R,len(J),I)
                residual = (Matrix_A[I,:])[:,J]@m_tilde - e_i_tilde          #evtl Matrix_A[I,J]@m und dann vom i-ten eintrag noch 1 abziehen für Effizienz
                current_residual = retransform_residual(residual,I,Dimension)
                current_norm_residual = np.linalg.norm(residual)
            else:
                Simulation = False
            iterations = iterations +1
        print(iterations,"Iterationen")
        M = set_m_to_M(m_tilde,i,M,J)                                                            #Die aktuelle Näherung an die i-te Spalte der inversen Matrix wird in M abgespeichert
    return [iterations,M]


def positions_nonzeros(Matrix_A,Dimension):         #Einmal wird die Liste der Positionen der Nichtnuullelemente erstellt
    Matrix_A = Matrix_A.tocsr()
    Nonzero_positions = []
    for i in range(0,Dimension):
        index_begin = Matrix_A.indptr[i]                        #erster Nichtnull-Eintrag der i-ten Zeile 
        index_end = Matrix_A.indptr[i+1]                            #erster Nichtnull-Eintrag der i+1-ten Zeile
        nonzeros = Matrix_A.indices[index_begin:index_end]
        Nonzero_positions.append(nonzeros.tolist())
    Matrix_A = Matrix_A.tocsc()                 
    return [Nonzero_positions,Matrix_A]

def getcol_boost(Matrix_A,Dimension):
    temp = Matrix_A.dot(Matrix_A.T)
    return temp.diagonal()

def Initialize_J(m):                      #Indizes der Nichtnullelemente der Spalte m sollen zu J hinzugefügt werden
    J = m.indices.tolist()
    return J

def Initialize_I(Matrix_A,J,index):
    sliced = Matrix_A[:,J]
    temp = np.unique(sliced.indices).tolist()
    #print(J,"J")
    if index not in temp:
        temp.append(index)
    return temp

                
def unit_vector_tilde(index,I):                 #reduzierter Einheitsvektor 
    e_i_tilde = np.zeros(len(I))    
    index_in_I = I.index(index)                        #Position der index-Zeile in der Menge I
    e_i_tilde[index_in_I] = 1
    return e_i_tilde

def retransform_residual(residual,I,Dimension):             #Residuum wird wieder auf die Ausgangsdimension transformiert
    return_vec = np.zeros((Dimension,1))
    (return_vec[I,0]) = residual 
    return return_vec
     

def unite_sets(I,J,I_tilde,J_tilde):                                    #Die Mengen I und J werden jeweils mit den Mengen I_tilde und J_tilde vereinigt
        
    I.extend(I_tilde)
    J.extend(J_tilde)
    return [I,J]


def set_J_tilde(Matrix_A,current_residual,J,Positions_nonzero):                                          #Diese Methode setzt die Kandidaten für eine Expansion der Menge J

    J_tilde = []
    
    vector_non_zero_residual = current_residual.nonzero()[0]
    
    for i in vector_non_zero_residual:
        J_tilde.extend(Positions_nonzero[i])
    #J_tilde = np.unique(J_tilde)
    return [np.setdiff1d(J_tilde,J)]



def array_of_candidates(J_tilde,Matrix_A,current_residual,l,norm_squared_cols,I,J,residual):        #Array aus Index-Werte Paaren wird erstellt
    
    
    array_kandidaten_und_Werte = np.zeros((l,2))               #Matrix, die sowohl Kandidat als auch zugehörigen Wert speichert
    
    
    i = 0         
    rAej_vector = residual.T@(Matrix_A[I,:])[:,J_tilde]

    
    for j in J_tilde:
                                     
    
        array_kandidaten_und_Werte[i][1] = j                                                          #Kandidat wird auf Position 1 geschrieben (Python---> 0)
        rAej = rAej_vector[i]

        inner_Aej = norm_squared_cols[j]
        array_kandidaten_und_Werte[i][0] = - (rAej*rAej)/inner_Aej   #Wert des Kandidaten wird auf Position 0 gespeichert(Python ---> 1)
    
        i = i+1
    return [array_kandidaten_und_Werte,i]

def new_Indizes_J_Minimization_problem(Matrix_A,Anzahl_neue_Indizes_J_pro_Iteration,current_residual,J,Positions_nonzero,norm_squared_cols,I,residual): #neue Indizes werden über 1-dim Minproblem bestimmt
    
    
    [J_tilde] = set_J_tilde(Matrix_A,current_residual,J,Positions_nonzero)

    l = len(J_tilde)
    

    [array_kandidaten_und_Werte,i] = array_of_candidates(J_tilde,Matrix_A,current_residual,l,norm_squared_cols,I,J,residual)

    int_kandidaten = []

    array_kandidaten_und_Werte = array_kandidaten_und_Werte[array_kandidaten_und_Werte[:,0].argsort()]                            #Es wird nach dem kleinsten Residuum sortiert
    if (Anzahl_neue_Indizes_J_pro_Iteration >= i):                    #weil es weniger Kandidaten geben könnte als maximal möglich wären, würden alle hinzugefügt werden. Im Fall großer Matrizen eher unwahrscheinlich

        for k in range(0,i):
            int_kandidaten.append(int(array_kandidaten_und_Werte[k][1]))       
    
    else:
        for k in range(0,Anzahl_neue_Indizes_J_pro_Iteration):
            int_kandidaten.append(int(array_kandidaten_und_Werte[k][1]))

    J_tilde = np.sort(int_kandidaten,0)                 #Die Indizes werden noch sortiert

    return J_tilde.tolist()  

def set_I_tilde(Matrix_A,J_tilde,I,J):             #ausgehend von J_tilde wird I_tilde erstellt

    J_new = J + J_tilde

    sliced = Matrix_A[:,J_new]

    I_tilde = np.unique(sliced.indices)
    I_tilde = np.setdiff1d(I_tilde,I).tolist()


    return I_tilde

def set_m_to_M(m,index_column,M,J):         #Der Vektor m, der die Approximation an eine Spalte von M darstellt wird in M implementiert

    M[J,index_column] = m

    return M
    

def R_backwards_solver(index_k,Q,R,l,I):            #R ist obere Dreicksmatrix
    index_k = I.index(index_k)

    m = Q[index_k,0:l]                                           #Multiplikation Q^T e_k nicht nötig, weil das der k-ten Zeile von Q entspricht
    m = sclin.solve_triangular(R,m)

    return m


def compute_QR_decomposition(I,J,Matrix_A):
        
    n_2 = len(J)

    [Q,R_ganz] = np.linalg.qr((Matrix_A[I,:])[:,J].toarray(),"complete")
    R = R_ganz[0:n_2,0:n_2]
        
    return [Q,R] 


def compose_QR_decomposition(I,J,I_tilde,J_tilde,Matrix_A,Q,R):           
    
    n_1 = len(I)
    n_2 = len(J)
    n_1_tilde = len(I_tilde)
    n_2_tilde = len(J_tilde)
    temp = Q.T@(Matrix_A[I,:])[:,J_tilde]
    
    Q1 = np.zeros((n_1 + n_1_tilde,n_1 + n_1_tilde))
    Q2 = np.zeros((n_1 + n_1_tilde,n_1 + n_1_tilde))
        
    Q1[0:n_1,0:n_1] = Q                                                 #Die alte Q-Matrix wird wiederbenutzt
    Q1[n_1:n_1 + n_1_tilde,n_1:n_1 + n_1_tilde] = np.eye(n_1_tilde)
        
    Q2[0:n_2,0:n_2] = np.eye(n_2)
        

    R_neu = np.zeros((n_2_tilde + n_2,n_2_tilde + n_2))
        
    R_neu[0:n_2,0:n_2] = R                              #der linke obere R-Block ist noch der der alten QR-Zerlegung
        
    B_1 = temp[0:n_2,:]
        

    B_2 = np.zeros((n_1 + n_1_tilde - n_2,n_2_tilde))
        
    B_2[0:(n_1-n_2),0:n_2_tilde] = temp[n_2:n_1,:]
    B_2[(n_1-n_2):(n_1-n_2+n_1_tilde),0:n_2_tilde] = (Matrix_A[I_tilde,:])[:,J_tilde].toarray()

    [Q_1,R_1] = np.linalg.qr(B_2,"complete")            #Die neue kleine QR-Zerlegung
        
    Q2[n_2:n_1 + n_1_tilde,n_2:n_1 + n_1_tilde] = Q_1
    R_neu[n_2:n_2 + n_2_tilde,n_2:n_2 + n_2_tilde] = R_1[0:n_2_tilde,0:n_2_tilde]
    
    R_neu[0:n_2,n_2:n_2 + n_2_tilde] = B_1
    

    Q = Q1@Q2
    return [Q,R_neu]
    
def control_lists(I,J):                     #kontrolliert, ob eine Menge leer ist und damit die weitere Berechnung unnötig ist
    #print(I,J,"I und J")
    if (len(I) == 0) and (len(J) == 0):
        print("AUA")
        #time.sleep(5)
        return False
    return True
     
def load_Matrix_A(level):
    Matrix_A = load_data(level)
    Dimension = Matrix_A.shape[0]
    M_default = sparse.eye(Dimension).tocsc()       #sparse.lil_matrix((self.Dimension,self.Dimension))
    M = sparse.lil_matrix((Dimension,Dimension))
        
    return [Matrix_A,M_default,M]
    


if __name__ == "__main__":
    
    epsilon = 0.8                               #obere Schranke Abbruchkriterium Residuum ||r-Am||
    max_fillin = 2                           #obere Schranke an Einträgen in den Spalten aus M
    case = 2                                    #Die Art des Abbruchkriteriums... Kombiniert?
    Anzahl_neue_Indizes_J_pro_Iteration = 4     #Anzahl an neuen Einträgen, die pro Iterationsschritt zu J hinzugefügt werden 
    
    
    Dauer_SPAI = []
    Dauer_SPAI_Prec = []
    Dauer_ohne_Prec = []
    Dauer_ILU_Prec = []
    iterationen = []
    dimensions = []
    
    
    for i in range(1,6):                                                                             
        [Matrix_A,M_default,M] = load_Matrix_A(i)
        Matrix_A = Matrix_A.tocsc()
        Matrix_A.eliminate_zeros()
        np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
    
        
        [iterations,duration,M] = Start(Matrix_A,epsilon,max_fillin,case,Anzahl_neue_Indizes_J_pro_Iteration,M_default,M)
        print(iterations, " Iterationen")
        print(duration, " Dauer")
        
        iterationen.append(iterations)
        Dauer_SPAI.append(duration)
    
        Dimension = Matrix_A.shape[0]
        dimensions.append(Dimension)
        
        
        time_1 = time.time()
        [x,info] = sp.gmres(Matrix_A,np.ones(Dimension))
        time_2 = time.time()
        diff = time_2-time_1
        print(diff,"GMRES ohne Preconditioning")
        print(info)   
        Dauer_ohne_Prec.append(diff)
        
        time_1 = time.time()
        [x,info] = sp.gmres(Matrix_A,np.ones(Dimension),M = M.tocsr())
        time_2 = time.time()
        diff = time_2-time_1
        print(diff,"GMRES mit SPAI-Preconditioning")
        print(info)
        Dauer_SPAI_Prec.append(diff)
        
        
        Matrix_A.tocsc()
        spilu = sp.spilu(Matrix_A)
        solver = lambda x : spilu.solve(x) 
        M = sp.LinearOperator((Dimension,Dimension), solver)
        time_1 = time.time()
        [x,info] = sp.gmres(Matrix_A,np.ones(Dimension),M = M)
        time_2 = time.time()
        diff = time_2-time_1
        print(diff,"GMRES mit ILU-Preconditioning")
        print(info)
        Dauer_ILU_Prec.append(diff)
        
        
        
    
    
    print(Dauer_SPAI_Prec,"Dauer_SPAI_Prec")
    print(Dauer_SPAI,"Dauer_SPAI")
    print(Dauer_ohne_Prec,"Dauer_ohne_Prec")
    print(Dauer_ILU_Prec,"Dauer_ILU_Prec")
    print(iterationen,"iterationen")
    
    print(dimensions)
    print(Dauer_SPAI)
    
    plt.clf()
    plt.figure(1)
    plt.plot(dimensions,Dauer_SPAI,"ro")
    plt.show()
    
    #plt.clf()
    plt.figure(2)
    plt.plot(dimensions,Dauer_ohne_Prec,"bo")#,label="Dauer_ohne_Prec")
    plt.plot(dimensions,Dauer_SPAI_Prec,"ro")#,label="SPAI_Dauer_Prec")
    plt.plot(dimensions,Dauer_ILU_Prec,"yo")#,label="ILU_Dauer_Prec")
    plt.show()
    
    
    
    
    
    
    
    
    