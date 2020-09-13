# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:53:07 2020

@author: tim31
"""
        
import numpy as np
from scipy import sparse
from scipy import linalg as sclin
import time
from load_data import load_data
import cupy



def positions_nonzeros(Matrix_A,Dimension):
    Nonzero_positions = []
    for i in range(0,Dimension):
        index_begin = Matrix_A.indptr[i]
        index_end = Matrix_A.indptr[i+1]
        nonzeros = Matrix_A.indices[index_begin:index_end]
        Nonzero_positions.append(nonzeros.tolist())
    Matrix_A = Matrix_A.tocsc()
    return [Nonzero_positions,Matrix_A]

def getcol_boost(Matrix_A,Dimension):
    temp = Matrix_A@(Matrix_A.T)
    return temp.diagonal()
        
        
def Initialize_J(m):                      #Indizes der Nichtnullelemente der Spalte m sollen zu J hinzugefügt werden
    #J = np.zeros((m.shape[0],1))
    
    J = m.nonzero()[0]
    #print(type(J),"dyttttttttttttttttttttttttttttttttttttttttttttttt")
    #print(m,"m")
    #print(J,"J")
    return J
   
def Initialize_I(Matrix_A,J,index,Dimension):
    #time_1= time.time()
    sliced = Matrix_A[:,J]

    return np.unique(sliced.indices)
    
def Start(Matrix_A,epsilon,max_fillin,case_Abbruchkriterium,Anzahl_neue_Indizes_J_pro_Iteration,Dateiname_Matrix,M_default,M):
    start_time = time.time()

    if(case_Abbruchkriterium == 0):                #case 0: nur Epsilon
        [iterations,M] = start_case_0(Matrix_A,epsilon,Anzahl_neue_Indizes_J_pro_Iteration,Dateiname_Matrix,M_default,M)   
    elif(case_Abbruchkriterium == 1):                                         #case 1: nur max fill-in
        [iterations,M] = start_case_1(Matrix_A,max_fillin,Anzahl_neue_Indizes_J_pro_Iteration,Dateiname_Matrix,M_default,M)
    elif((case_Abbruchkriterium == 2)):                                             #case 2: Epsilon und max fill-in
        [iterations,M] = start_case_2(Matrix_A,epsilon,max_fillin,Anzahl_neue_Indizes_J_pro_Iteration,Dateiname_Matrix,M_default,M)
    end_time = time.time()
    time_duration = end_time-start_time
        
    return [iterations,time_duration,M]
        
def start_case_0(Matrix_A,epsilon,Anzahl_neue_Indizes_J_pro_Iteration,Dateiname_Matrix,M_default,M):
        
    iterations = 0
    Dimension = Matrix_A.shape[0]
    [Positions_nonzero,Matrix_A] = positions_nonzeros(Matrix_A,Dimension)
    norm_squared_cols = getcol_boost(Matrix_A,Dimension)
    for i in range(0,Dimension):
        print(i,"---------------------------------------------------------------------")
        Simulation = True
        #print("Schritt", i)
        m = M_default.getcol(i)                         #m Defaultspalte
        
        J = Initialize_J(m)                                                 #Menge J wird erstellt. In jedem i-Schritt anders, weil default-M unterschiedliche Spalten hat
        I = Initialize_I(Matrix_A,J,i,Dimension)                                                     #Auf Basis der Menge J kann nun auch die Menge I initialisiert werden
        e_i_tilde = unit_vector_tilde(i,I,Dimension)
       # print(I,J)
                                                     #muss noch in die richtige Form gebracht werden, nur Indizes aus Menge I wichtig 
        [Q,R] = compute_QR_decomposition(I,J,Matrix_A)               #Einmal wird die große QR-Zerlegung berechnet
        
        m_tilde = R_backwards_solver(i,Q,R,J.shape[0],I)
        
        #print(m_tilde,"wq")
        residual = (Matrix_A[np.ix_(I,J)])@(m_tilde) - e_i_tilde        #evtl Matrix_A[I,J]@m und dann vom i-ten eintrag noch 1 abziehen für Effizienz
        #print(residual,i,"Hhhhhhhhhhhhhhh")
        current_residual = retransform_residual(residual,I,Dimension)
        current_norm_residual = np.linalg.norm(current_residual)
        
        iterations = iterations +1
        print(iterations)
        #print(current_norm_residual,"norm")
        while((current_norm_residual > epsilon) and Simulation):                                   #neue Kandidaten für eine Expansion werden geschaffen
            #print(I,J)
            J_tilde = new_Indizes_J_Minimization_problem(Matrix_A,Anzahl_neue_Indizes_J_pro_Iteration,current_residual,J,Positions_nonzero,norm_squared_cols,I,residual)                           #beste Kandidaten werden zu J_tilde hinzugefügt
            I_tilde = set_I_tilde(Matrix_A,J_tilde,I,J,Dimension)
            #print(np.intersect1d(J,J_tilde),"Schnitt")
            #print(I_tilde,"I_tilde")
            #print(J_tilde,"J_tilde")
            if(control_lists(I_tilde,J_tilde)):
                [Q,R] = compose_QR_decomposition(I,J,I_tilde,J_tilde,Matrix_A,Q,R)                              #Die neuen Bestandteile Q und R der QR-Zerlegung werden geupdatet (keine komplett neue QR-Zerlegung)
                [I,J] = unite_I_and_J(I,J,I_tilde,J_tilde)
                #print(I,"I")
                #print(J,"J")
                e_i_tilde = unit_vector_tilde(i,I,Dimension)
                m_tilde  = R_backwards_solver(i,Q,R,len(J),I)
                residual = (Matrix_A[np.ix_(I,J)])@(m_tilde) - e_i_tilde          #evtl Matrix_A[I,J]@m und dann vom i-ten eintrag noch 1 abziehen für Effizienz
                #print(residual[0:6],i,"Hierddd")
                current_residual = retransform_residual(residual,I,Dimension)
                current_norm_residual = np.linalg.norm(residual)
                #print(current_norm_residual,"norm")
            else:
                Simulation = False
                print("Eine Menge ist leer")
            iterations = iterations +1
        print(iterations,"Iterationen")
        M = set_m_to_M(m_tilde,i,M,J)                                                            #Die aktuelle Näherung an die i-te Spalte der inversen Matrix wird in M abgespeichert
    return [iterations,M]
    
def start_case_1(Matrix_A,max_fillin,Anzahl_neue_Indizes_J_pro_Iteration,Dateiname_Matrix,M_default,M):
    iterations = 0
    Dimension = Matrix_A.shape[0]
    for i in range(0,Dimension):
        #print("---------------- --------- --------- --------- --------- --------- ---------")
        Simulation = True
        #print("Schritt", i)
        m = M_default.getcol(i)                        #m Defaultspalte
        J = Initialize_J(m)                                                 #Menge J wird erstellt. In jedem i-Schritt anders, weil default-M unterschiedliche Spalten hat
        I = Initialize_I(Matrix_A,J,i)                                                     #Auf Basis der Menge J kann nun auch die Menge I initialisiert werden
        e_i_tilde = unit_vector_tilde(i,I,Dimension)
                                                     #muss noch in die richtige Form gebracht werden, nur Indizes aus Menge I wichtig 
        [Q,R] = compute_QR_decomposition(I,J,Matrix_A)               #Einmal wird die große QR-Zerlegung berechnet
        
        m_tilde = R_backwards_solver(i,Q,R,len(J),I)
        residual = (Matrix_A[I,:])[:,J]@(m_tilde) - (e_i_tilde)          #evtl Matrix_A[I,J]@m und dann vom i-ten eintrag noch 1 abziehen für Effizienz
        current_residual = retransform_residual(residual,I,Dimension)
        current_norm_residual = np.linalg.norm(current_residual)
        
        iterations = iterations +1
        #print(current_norm_residual,"norm")
        while((len(J) + Anzahl_neue_Indizes_J_pro_Iteration) <= max_fillin and Simulation):
            J_tilde = new_Indizes_J_Minimization_problem(Matrix_A,Anzahl_neue_Indizes_J_pro_Iteration,current_residual,J)                           #beste Kandidaten werden zu J_tilde hinzugefügt
            I_tilde = set_I_tilde(Matrix_A,J_tilde,I,J)
            print(I_tilde,"I_tilde")
            print(J_tilde,"J_tilde")
            if(control_lists(I_tilde,J_tilde)):
                [Q,R] = compose_QR_decomposition(I,J,I_tilde,J_tilde,Matrix_A,Q,R)                              #Die neuen Bestandteile Q und R der QR-Zerlegung werden geupdatet (keine komplett neue QR-Zerlegung)
                [I,J] = unite_I_and_J(I,J,I_tilde,J_tilde)
                e_i_tilde = unit_vector_tilde(i,I,Dimension)
                m_tilde  = R_backwards_solver(i,Q,R,len(J),I)
                residual = (Matrix_A[I,:])[:,J]@(m_tilde) - np.array(e_i_tilde)          #evtl Matrix_A[I,J]@m und dann vom i-ten eintrag noch 1 abziehen für Effizienz
                current_residual = retransform_residual(residual,I,Dimension)
                current_norm_residual = np.linalg.norm(current_residual)
                #print(current_norm_residual,"norm")
            else:
                Simulation = False
                #print("Eine Menge ist leer")
            iterations = iterations +1  
        M = set_m_to_M(m_tilde ,i,M,J)
    return [iterations,M]

def start_case_2(Matrix_A,epsilon,max_fillin,Anzahl_neue_Indizes_J_pro_Iteration,Dateiname_Matrix,M_default,M):
    iterations = 0
    Dimension = Matrix_A.shape[0]
    for i in range(0,Dimension):
        #print("---------------- --------- --------- --------- --------- --------- ---------")
        Simulation = True
        #print("Schritt", i)
        m = M_default[:,i]                        #m Defaultspalte
        J = Initialize_J(m)                                                 #Menge J wird erstellt. In jedem i-Schritt anders, weil default-M unterschiedliche Spalten hat
        I = Initialize_I(Matrix_A,J,i)                                                     #Auf Basis der Menge J kann nun auch die Menge I initialisiert werden
        e_i_tilde = unit_vector_tilde(i,I,Dimension)
                                                     #muss noch in die richtige Form gebracht werden, nur Indizes aus Menge I wichtig 
        [Q,R] = compute_QR_decomposition(I,J,Matrix_A)               #Einmal wird die große QR-Zerlegung berechnet
        
        m_tilde = R_backwards_solver(i,Q,R,len(J),I)
        residual = (Matrix_A[I,:])[:,J]@(m_tilde) - (e_i_tilde)          #evtl Matrix_A[I,J]@m und dann vom i-ten eintrag noch 1 abziehen für Effizienz
        current_residual = retransform_residual(residual,I,Dimension)
        current_norm_residual = np.linalg.norm(current_residual)
        
        iterations = iterations +1
        #print(current_norm_residual,"norm")
        while(((len(J) + Anzahl_neue_Indizes_J_pro_Iteration) <= max_fillin) and (current_norm_residual > epsilon) and Simulation):
            J_tilde = new_Indizes_J_Minimization_problem(Matrix_A,Anzahl_neue_Indizes_J_pro_Iteration,current_residual,J)                           #beste Kandidaten werden zu J_tilde hinzugefügt
            I_tilde = set_I_tilde(Matrix_A,J_tilde,I,J)
            if(control_lists(I_tilde,J_tilde)):
                [Q,R] = compose_QR_decomposition(I,J,I_tilde,J_tilde,Matrix_A,Q,R)                              #Die neuen Bestandteile Q und R der QR-Zerlegung werden geupdatet (keine komplett neue QR-Zerlegung)
                [I,J] = unite_I_and_J(I,J,I_tilde,J_tilde)
                e_i_tilde = unit_vector_tilde(i,I,Dimension)
                m_tilde  = R_backwards_solver(i,Q,R,len(J),I)
                residual = (Matrix_A[I,:])[:,J]@(m_tilde) - np.array(e_i_tilde)          #evtl Matrix_A[I,J]@m und dann vom i-ten eintrag noch 1 abziehen für Effizienz
                current_residual = retransform_residual(residual,I,Dimension)
                current_norm_residual = np.linalg.norm(current_residual)
                #print(current_norm_residual,"norm")
            else:
                Simulation = False
                #print("Eine Menge ist leer")
            iterations = iterations +1     
        M = set_m_to_M(m_tilde ,i,M,J)                                                          #Die aktuelle Näherung an die i-te Spalte der inversen Matrix wird in M abgespeichert
    return [iterations,M]
                    
def unit_vector_tilde(index,I,Dimension):
    e_i_tilde = np.zeros((I.shape[0],1))
    if index in I:    
        index_in_I = np.where(I == index)[0]                      #Position der index-Zeile in der Menge I
        e_i_tilde[index_in_I,0] = 1
    return e_i_tilde

def retransform_residual(residual,I,Dimension):
    
    return_vec = np.zeros((Dimension,1))
    (return_vec[I,:]) = residual 
    return return_vec
        
def set_J_tilde(Matrix_A,current_residual,J,Positions_nonzero):                                          #Diese Methode setzt die Kandidaten für eine Expansion der Menge J
    #Matrix_A = Matrix_A.tocsr()
    #print(Positions_nonzero[5],"Hier")
    #J_tilde = np.array([],int)
    J_tilde = []
    
    vector_non_zero_residual = current_residual.nonzero()[0]
    
    for i in vector_non_zero_residual:
        #nonzeros = (Matrix_A1.getrow(i).nonzero())[1]
        #print(nonzeros,J_tilde,"hier",type(nonzeros),type(J_tilde))
        #J_tilde = (np.append(J_tilde,nonzeros))
        #J_tilde = np.append(J_tilde,Positions_nonzero[i])
        
        J_tilde.extend(Positions_nonzero[i])
        
        #print(Positions_nonzero[i],"LAJHBLAFLAJHBLAFLAJHBLAFLAJHBLAF")
    #print(type(J_tilde),"J_tilde")
    J_tilde = np.array(J_tilde,int)
    #for j in J_tilde:
    #    if j in J:
    #        J_tilde.remove(j)
    #        print("Einer wurde entfernt!Einer wurde entfernt!Einer wurde entfernt!Einer wurde entfernt!Einer wurde entfernt!Einer wurde entfernt!")
    
    #Matrix_A = Matrix_A.tocsc()
    #print(J_tilde)
    return [(np.setdiff1d(J_tilde,J)),vector_non_zero_residual]

def unite_I_and_J(I,J,I_tilde,J_tilde):                                    #Die Mengen I und J werden jeweils mit den Mengen I_tilde und J_tilde vereinigt
        
    I = np.append(I,I_tilde)
    J = np.append(J,J_tilde)
    return [I,J]

def inner(x):                   #Methode zur Bestimmung des Skalarprodukts mit sich selbst
    
    inner = 0
    for i in x.data:
        inner = inner + i*i
    return inner

def fuelle_Array_mit_Kandidaten(J_tilde,Matrix_A,current_residual,l,norm_squared_cols,I,J,residual):
    
    
    array_kandidaten_und_Werte = np.zeros((l,2))               #Matrix, die sowohl Kandidat als auch zugehörigen Wert speichert
    
    
    i = 0         #Laufindex für die neue 2D Liste
    #time_1 = time.time()
    #vector = residual.T@Matrix_A[:,J_tilde]
    
    #vector = residual.T@Matrix_A[I,J_tilde]
    
    rAej_vector = residual.T@Matrix_A[np.ix_(I,J_tilde)]
    
    for j in J_tilde:
    #for j in range(0,J_tilde.shape[0]):                                     
    
        array_kandidaten_und_Werte[i][1] = j                                                          #Kandidat wird auf Position 1 geschrieben (Python---> 0)
        #Aj = Matrix_A.getcol(j)
        rAej = rAej_vector[0,i]
        #rAej = vector[0,j]
        inner_Aej = norm_squared_cols[j]
        array_kandidaten_und_Werte[i][0] = - (rAej*rAej)/inner_Aej   #Wert des Kandidaten wird auf Position 0 gespeichert(Python ---> 1)
    
        i = i+1
    return [array_kandidaten_und_Werte,i]

def new_Indizes_J_Minimization_problem(Matrix_A,Anzahl_neue_Indizes_J_pro_Iteration,current_residual,J,Positions_nonzero,norm_squared_cols,I,residual):
    
    #time_1 = time.time()
    
    [J_tilde,vector_non_zero_residual] = set_J_tilde(Matrix_A,current_residual,J,Positions_nonzero)
    #print(J_tilde,"J_tilde")
    #time_2 = time.time()
    #diff = 1000*time_2 - 1000*time_1
    #print(diff,"J_tilde")
    l = J_tilde.shape[0]
    
    #array_kandidaten_und_Werte = np.zeros((l,2))               #Matrix, die sowohl Kandidat als auch zugehörigen Wert speichert
    #int_kandidaten = np.array([],int)
    [array_kandidaten_und_Werte,i] = fuelle_Array_mit_Kandidaten(J_tilde,Matrix_A,current_residual,l,norm_squared_cols,I,J,residual)
    #int_kandidaten = np.array([],int)
    int_kandidaten = []
    #time_2 = time.time()
    #diff = 1000*time_2 - 1000*time_1
    #print(diff,"Schleife")
    #time_1 = time.time()    
    #print(array_kandidaten_und_Werte, "vor" )
    array_kandidaten_und_Werte = array_kandidaten_und_Werte[array_kandidaten_und_Werte[:,0].argsort()]                            #Es wird nach dem kleinsten Residuum sortiert
    if (Anzahl_neue_Indizes_J_pro_Iteration >= i):                    #weil es weniger Kandidaten gibt als maximal möglich wären, werden alle hinzugefügt. Im Fall großer Matrizen eher unwahrscheinlich
        #print("HUHUHUHUHUHUHHUHUHUHUHUHUHUHUHUHUHUHUHUHUHUHUHUHUHUHHHH")
        for k in range(0,i):
            int_kandidaten.append(int(array_kandidaten_und_Werte[k][1]))       
    
    else:
        for k in range(0,Anzahl_neue_Indizes_J_pro_Iteration):
            int_kandidaten.append(int(array_kandidaten_und_Werte[k][1]))
            #int_kandidaten = np.append(int_kandidaten,int(array_kandidaten_und_Werte[i][1])) 
    J_tilde = np.sort(int_kandidaten,0)                 #Die Indizes werden noch sortiert
    #time_2 = time.time()
    #diff = 1000*time_2 - 1000*time_1
    #print(diff,"Sortieren")    
    return np.array(J_tilde,int)    
   
def set_I_tilde(Matrix_A,J_tilde,I,J,Dimension):                                                                  #ausgehend von J_tilde wird I_tilde erstellt
    I_tilde = np.array([],int)                                                           #Die alte Kandidatenliste wird resetet
    J_new = np.append(J,J_tilde)  
    sliced = Matrix_A[:,J_new]
    I_tilde = np.unique(sliced.indices)
    I_tilde = np.setdiff1d(I_tilde,I)
    return I_tilde

def set_m_to_M(m,index_column,M,J):                                                           #Der Vektor m, der die Approximation an eine Spalte von M darstellt wird wieder auf die "normale" Dimension gebracht und dann in M implementiert
    i = 0                                  #Die Approximation an die index_column Spalte wird zu M hinzugefügt

    (M[np.ix_(J,np.array([index_column]))]) = m

    return M
    
def convert_Indizes_from_I(index_i,I):               
        
    if (index_i in I):
        return np.where(I == index_i)[0]                            #gibt die Position des index_i in I zurück
    
def R_backwards_solver(index_k,Q,R,l,I):                                   #Problem: index_k ist Position der 1 des Einheitsvektors. Diese muss jedoch noch durch die Menge I umgewandelt werden
    m = np.zeros(l)
    index_k = convert_Indizes_from_I(index_k,I)
    l = R.shape[0]
    m = Q[index_k,0:l]
    m = sclin.solve_triangular(R,m.T)
    return m


def compute_QR_decomposition(I,J,Matrix_A):
        
    n_2 = J.shape[0]
    [Q,R_ganz] = np.linalg.qr((Matrix_A[np.ix_(I,J)]).toarray(),"complete")
    R = R_ganz[0:n_2,0:n_2]

    return [Q,R] 


def compose_QR_decomposition(I,J,I_tilde,J_tilde,Matrix_A,Q,R):     
    
    n_1 = I.shape[0]
    n_2 = J.shape[0]
    n_1_tilde = I_tilde.shape[0]
    n_2_tilde = J_tilde.shape[0]

    Test = Q.T@Matrix_A[np.ix_(I,J_tilde)]
    
    Q1 = np.zeros((n_1 + n_1_tilde,n_1 + n_1_tilde))
    Q2 = np.zeros((n_1 + n_1_tilde,n_1 + n_1_tilde))
        
    Q1[0:n_1,0:n_1] = Q                                                 #Die alte Q-Matrix wird wiederbenutzt
    Q1[n_1:n_1 + n_1_tilde,n_1:n_1 + n_1_tilde] = np.eye(n_1_tilde)
        
    Q2[0:n_2,0:n_2] = np.eye(n_2)
        

    R_neu = np.zeros((n_2_tilde + n_2,n_2_tilde + n_2))
        
    R_neu[0:n_2,0:n_2] = R                              #der linke obere R-Block ist noch der der alten QR-Zerlegung
        
    B_1 = Test[0:n_2,:]#(Q.T)[0:n_2,0:n_1]@Matrix_A[np.ix_(I,J_tilde)]
        

    B_2 = np.zeros((n_1 + n_1_tilde - n_2,n_2_tilde))
        
    B_2[0:(n_1-n_2),0:n_2_tilde] = Test[n_2:n_1,:]#((Q.T)[n_2:n_1,0:n_1])@Matrix_A[np.ix_(I,J_tilde)]
    B_2[(n_1-n_2):(n_1-n_2+n_1_tilde),0:n_2_tilde] = Matrix_A[np.ix_(I_tilde,J_tilde)].toarray()

    [Q_1,R_1] = np.linalg.qr(B_2,"complete")            #Die neue kleine QR-Zerlegung
        
    Q2[n_2:n_1 + n_1_tilde,n_2:n_1 + n_1_tilde] = Q_1
    R_neu[n_2:n_2 + n_2_tilde,n_2:n_2 + n_2_tilde] = R_1[0:n_2_tilde,0:n_2_tilde]
    
    R_neu[0:n_2,n_2:n_2 + n_2_tilde] = B_1

    Q = Q1@Q2

    return [Q,R_neu]


def Matrix_Vec_Mul(A,x):
    return A.dot(x)
    
def control_lists(I,J):                     #kontrolliert, ob eine Menge leer ist und damit die weitere Berechnung unnötig ist
    if (I.shape[0] == 0) or (J.shape[0] == 0):
        return False
    return True
     
def load_Matrix_A(Dateiname,Jan_Leibold,level):
        
    if(Jan_Leibold):
        Matrix_A = load_data(level)
        Dimension = Matrix_A.shape[0]
        M_default = sparse.eye(Dimension).tocsc()       #sparse.lil_matrix((self.Dimension,self.Dimension))
        M = sparse.lil_matrix((Dimension,Dimension))
        
        return [Matrix_A,M_default,M]
        
    else:
        Zeilen = []
        Zeile = [] 
        Matrix_txt = open(Dateiname)
        indptr = []
        indices = []
        data = []
        
        
        for line in Matrix_txt:
            Zeilen.append(line.rstrip())
        aktueller_Eintrag = ""
        for s in Zeilen[0]:
            if(s == " "):
                indptr.append(float(aktueller_Eintrag)) 
                aktueller_Eintrag = ""
                
            else:
                aktueller_Eintrag = aktueller_Eintrag + s
        indptr.append(float(aktueller_Eintrag))
        aktueller_Eintrag = ""
        for s in Zeilen[1]:
            if(s == " "):
                indices.append(float(aktueller_Eintrag)) 
                aktueller_Eintrag = ""
                
            else:
                aktueller_Eintrag = aktueller_Eintrag + s
        indices.append(float(aktueller_Eintrag))
        aktueller_Eintrag = ""
        for s in Zeilen[2]:
            if(s == " "):
                data.append(float(aktueller_Eintrag)) 
                aktueller_Eintrag = ""
                
            else:
                aktueller_Eintrag = aktueller_Eintrag + s
        data.append(float(aktueller_Eintrag))
        
        indptr = np.array(indptr)
        indices = np.array(indices)
        data = np.array(data)
        
        
        Matrix_A = sparse.csc_matrix((data,indices,indptr))
        Dimension = Matrix_A.shape[0]
        M_default = sparse.eye(Dimension).tocsc()
        M = sparse.lil_matrix((Dimension,Dimension))
        
        Matrix_txt.close()
        
        return [Matrix_A,M_default,M]


if __name__ == "__main__":
    epsilon = 0.1                     #obere Schranke Abbruchkriterium Residuum ||r-Am||
    max_fillin = 6                             #obere Schranke an Einträgen in den Spalten aus M
    case_Abbruchkriterium = 0          #Die Art des Abbruchkriteriums... Kombiniert?
    Anzahl_neue_Indizes_J_pro_Iteration = 2   #Anzahl an neuen Einträgen, die pro Iterationsschritt zu J hinzugefügt werden 
    
    Jan_Leibold = True
    level = 1
    Dateiname_Matrix = "Sparse_Matrix_full.txt"        #"Matrix_test.txt"    
    [Matrix_A,M_default,M] = load_Matrix_A(Dateiname_Matrix,Jan_Leibold,level)   #np.array([[3,0,0,0,3,4,0,1,1,1,1,2,3],[0,1,0,2,0,3,0,0,3,4],[0,0,1,1,0,1,0,0,1,0],[0,0,0,1,0,0,0,0,7,0],[0,0,0,0,1,10,0,0,0,0],[0,0,0,0,0,1,1,0,0,0],[0,0,0,0,0,0,1,4,3,4],[0,0,0,0,0,0,0,1,2,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1,1]])                        #Die Matrix  aus dem LGS
    #Matrix_A = Matrix_A.tocsc()
    Matrix_A.eliminate_zeros()
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
    [iterations,duration,M] = Start(Matrix_A,epsilon,max_fillin,case_Abbruchkriterium,Anzahl_neue_Indizes_J_pro_Iteration,Dateiname_Matrix,M_default,M)
    print(iterations, " Iterationen")
    print(duration, " Dauer")
    np.savez_compressed('Matrix_M_123', a = M)