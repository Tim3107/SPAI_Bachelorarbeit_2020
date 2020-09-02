import numpy as np
import scipy.sparse as scp


def load_data(level):
    ''' Load matrix A and vector b from file
    '''
    
    # load data
    data = np.load('data_level'+str(level)+'.npz')
    
    # generate sparse matrix
    indptrA = data['indptrA']
    indicesA = data['indicesA']
    dataA = data['dataA']
    A_sparse = scp.csr_matrix((dataA, indicesA, indptrA))
    
    #vector of right hand side
    #b_vec = data['rhs_vec'] wird hier nicht benötigt
    
    
    return A_sparse                     #, b_vec wird hier nicht benötigt