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

def standard_feature_map(state, action, num_states, num_actions):
    """
    Standard one-hot encoding feature: phi(s,a) ∈ R^{num_states * num_actions}
    """
    feature = np.zeros(num_states * num_actions)
    idx = state * num_actions + action
    feature[idx] = 1.0
    return feature

def group_invariant_feature_map(state, action, num_states, num_actions, grid_size):
    """
    Group-invariant feature map: assume translation symmetry on grid
    Collapse (state, action) into bucket depending on (row offset, column offset) to goal
    """
    goal_state = num_states - 1  # assume bottom-right is goal
    goal_row, goal_col = divmod(goal_state, grid_size)
    row, col = divmod(state, grid_size)
    row_offset = row - goal_row
    col_offset = col - goal_col

    # Map (row_offset, col_offset, action) to a feature index
    # Shift offsets to positive values
    max_offset = grid_size - 1
    feature_size = (2 * max_offset + 1)**2 * num_actions
    feature = np.zeros(feature_size)

    idx = ((row_offset + max_offset) * (2 * max_offset + 1) + (col_offset + max_offset)) * num_actions + action
    feature[idx] = 1.0
    return feature

 


