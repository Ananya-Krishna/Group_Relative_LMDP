import numpy as np

def check_transition(transition_kernel):
    """
    check the given transition kernel is valid 
    """
    tran_shape = np.shape(transition_kernel) # shape of transition kernel, (H, S, A, S')
    print(tran_shape)
    H = tran_shape[0]
    S = tran_shape[1]
    A = tran_shape[2]
    S1 = tran_shape[3]

    label = True 

    for h in range(H):
        for s in range(S):
            for a in range(A):
                prob = transition_kernel[h,s,a,:]
                if(np.absolute( np.sum(prob) - 1) > 1e-5 or np.sum(prob < 0)>0):
                    print(prob)
                    label = False 
                    break
    return label 

 


