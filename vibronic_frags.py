import numpy as np
from math import log2, ceil, pow
import pickle
from mode_selector import rank_modes
from pennylane.labs.trotter_error import RealspaceOperator, RealspaceCoeffs, RealspaceSum, RealspaceMatrix
from pennylane.labs.trotter_error import vibronic_fragments
from scipy.linalg import expm
from itertools import product
from scipy.optimize import minimize

def couplings_to_RealSpaceMatrix(nstates, n_blocks, p, lambdas, alphas, betas):
    '''RealSpaceMatrix for the potential part'''
    #assuming len coeff == 2, it is list of alphas and betas
    rs_matrix = RealspaceMatrix.zero(n_blocks, p)
    for i in range(nstates):
        for j in range(nstates):
                c_op = RealspaceOperator(p, (), RealspaceCoeffs(lambdas[i,j], label="lamdas"))
                l_op = RealspaceOperator(p, ("Q",), RealspaceCoeffs(alphas[i,j], label="alphas"))
                q_op = RealspaceOperator(p, ("Q","Q",), RealspaceCoeffs(betas[i,j], label="betas"))
                rs_sum = RealspaceSum(p, [c_op, l_op, q_op])
                rs_matrix.set_block(i,j, rs_sum)
    return rs_matrix

def construct_pauli_matrix(pauli_term):
    """Construct the matrix corresponding to a given Pauli term."""
    paulidict = {
        'I': np.array([[1, 0], [0, 1]], dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex)
    }
    matrix = 1
    for p in reversed(pauli_term):
        matrix = np.kron(matrix, paulidict[p])
    return matrix

def matrix_to_RealSpaceMatrix(n_blocks, p, pauli):
    '''RealSpaceMatrix for a matrix acting on the electronic part, identity in the vibrational part'''
    #assuming len coeff == 2, it is list of alphas and betas
    rs_matrix = RealspaceMatrix.zero(n_blocks,p)
    for i in range(n_blocks):
        for j in range(n_blocks):
                rs_matrix.set_block(i,j, RealspaceSum(p, [RealspaceOperator(p, (), RealspaceCoeffs(np.array(pauli[i,j])))]))
    return rs_matrix

def sum_to_RealSpaceMatrix(n_blocks, p, sum:RealspaceSum):
    '''RealSpaceMatrix for a matrix acting on the electronic part, identity in the vibrational part'''
    #assuming len coeff == 2, it is list of alphas and betas
    rs_matrix = RealspaceMatrix.zero(n_blocks,p)
    for i in range(n_blocks):
                rs_matrix.set_block(i,i, sum)
    return rs_matrix

def check_zero_sum(sum:RealspaceSum):
    params = {'gridpoints':1}
    return sum.norm(params)==0

def decompose_to_pauli_terms(n_blocks, p, M):
    # n: padded number of electronic states (power of 2)
    # p: number of vibrational modes
    # M: a RealSpaceMatrix instance
    decomposition = []
    m = int(np.log2(n_blocks))
    for term in product(['I', 'X', 'Y', 'Z'], repeat=m):
        pauli_mat = construct_pauli_matrix(term)/ n_blocks #so that trace of pauli & M gives coeff
        real_pauli = matrix_to_RealSpaceMatrix(n_blocks, p, pauli_mat)
        prod = real_pauli @ M 
        coeff_sum = RealspaceSum.zero(p)
        for i in range(n_blocks):
            coeff_sum += prod.block(i, i)
        if not check_zero_sum(coeff_sum):
            decomposition.append((term, coeff_sum)) 
    return decomposition

def commute(matrix1, matrix2):
    commutator = matrix1@matrix2 - matrix2@matrix1
    return not np.any(commutator)

def sort(decomposition, type='operator_norm'):
    if type == 'operator_norm':
        return sorted(decomposition, key=lambda item: item[1].norm({'gridpoints':1}), reverse=True)
    
# we could remove the term with I electronic part because that commutes with everything
def fc_grouping(decomposition,  type='operator_norm'):
    sorted_terms = sort(decomposition, type)
    groups = []

    while sorted_terms:
        h_alpha = [sorted_terms[0]]
        remaining_terms = []

        for pauli,real in sorted_terms[1:]:
            if all(commute(construct_pauli_matrix(pauli),construct_pauli_matrix(i)) for i,j in h_alpha):
                h_alpha.append((pauli, real))
            else:
                remaining_terms.append((pauli, real))
        
        groups.append(h_alpha)
        sorted_terms = remaining_terms
    return groups

def kin_frag(nstates, n_blocks, p, omegas):
    kin_term = RealspaceOperator(
        p,
        ("P", "P"),
        RealspaceCoeffs(np.diag(omegas) / 2, label="omega"),
    )
    kin_sum = RealspaceSum(p, (kin_term,))
    kin_blocks = {(i, i): kin_sum for i in range(nstates)}
    kin_frag = RealspaceMatrix(n_blocks, p, kin_blocks)

    return kin_frag

"""
this code is for using real unitaries (orthogonal matrices), if the coefficients of
the vibrational modes in the potential part of the LVC and QVC model are real. 
"""

def unitary_from_orthogonal(n, params):
    """
    Compute a real orthogonal matrix U = exp(A), where A is anti-symmetric.
    
    Parameters:
    - n (int): Number of qubits.
    - params (array): Length d*(d-1)/2, where d = 2^n, for the upper triangle of A.
    
    Returns:
    - U (numpy array): Real orthogonal matrix of size 2^n x 2^n.
    """
    d = 2**n
    num_A = d * (d - 1) // 2  # Number of parameters for anti-symmetric A
    
    # Ensure params has the correct length
    A_params = params[:num_A]
    
    # Construct anti-symmetric matrix A
    A = np.zeros((d, d), dtype=float)  # Real matrix
    k = 0
    for i in range(d):
        for j in range(i + 1, d):
            A[i, j] = A_params[k]
            A[j, i] = -A_params[k]
            k += 1
    
    # Compute U = exp(A)
    U = expm(A)  # Real orthogonal matrix
    return U

def frobenius(tensor):
    return np.sum(np.abs(tensor)**2)

def cost(params, V, n, p):
    """
    Objective function to minimize: sum of absolute values of remainder.
    
    Args:
        params (array): Parameters
        V (array): Tensor of shape (2^n, 2^n, M).
        n (int): Number of qubits.
        m (int): Number of modes.
    
    Returns:
        float: Sum of absolute differences |V - V_rotated|.
    """
    V_tilda = make(params, n, p)[2]
    remainder = V - V_tilda
    return frobenius(remainder)

def make(params, n, p):
    """
    Construct V_tilda using a real orthogonal U.
    
    Args:
        params (array): Parameters: first d*(d-1)/2 for U, then 2^n * 2p for lam.
        n (int): Number of qubits.
        p (int): Parameter related to M (M = 2*p).
    
    Returns:
        U, lam, V_tilda: Orthogonal matrix, diagonal params, and resulting tensor.
    """
    d = 2**n
    num_A = d * (d - 1) // 2
    uparams = params[:num_A]  # Parameters for A
    lparams = params[num_A:]  # Parameters for lam
    lam = lparams.reshape(d, 2*p)  # Real matrix
    
    U = unitary_from_orthogonal(n, uparams)  # Real orthogonal U
    # Since U is real, U_dagger = U.T
    V_tilda = np.einsum('ik,km,kj->ijm', U, lam, U.T)
    return U, lam, V_tilda

def find_fragments(n, p, V_initial, max_iterations=10):
    """
    Iteratively find fragments by optimizing V_rotated until remainder is small.

    Args:
        n (int): Number of qubits.
        m (int): Number of modes.
        V_initial (array): Initial potential tensor of shape (2^n, 2^n, 1).
        tolerance (float): Convergence threshold for m.
        max_iterations (int): Maximum number of iterations.
    
    Returns:
        list: List of V_rotated fragments and their corresponding optimized parameters.
    """
    V_current = V_initial.copy()
    print(f'The initial norm of the coeff tensor:{frobenius(V_initial)}')
    fragments = []
    iteration = 0
    d = 2**n
    M = V_initial.shape[2]  # Third dimension of V
    num_A = d * (d - 1) // 2
    num_lam = d * M  # Since lam is (2^n, M)
    total_params = num_A + num_lam
    
    while iteration < max_iterations:
        # Initialize parameters
        a_init = np.random.rand(total_params)
        
        # Optimize
        result = minimize(cost, a_init, args=(V_current, n, p), method='BFGS', tol=1e-20)
        a_opt = result.x
        
        # Compute optimized fragment
        U_opt, lam_opt, V_rotated = make(a_opt, n, p)
        
        # Store fragment
        fragments.append((V_rotated.copy(), a_opt.copy()))
        
        # Update remainder
        V_current = V_current - V_rotated
        m = result.fun
        print(f"Iteration {iteration + 1}: m = {m}")
        
        iteration += 1
    
    if iteration == max_iterations:
        print(f"Reached max iterations.")
    
    return fragments

def save_fragments(fragments, filename):
    with open(f'frags/{filename}', 'wb') as file:
        pickle.dump(fragments, file)
    print(f"Fragments saved to {filename}")


def load_fragments(filename):
    with open(f'frags/{filename}', 'rb') as file:
        fragments = pickle.load(file)
    print(f"Fragments loaded from {filename}")
    return fragments

def vibronic_frags(mol, p, type, subtype, find = True, num_frags = 10):
    """
    Generate a set of vibronic‐Hamiltonian fragments according to the given fragmentation scheme.

    This routine drives the full workflow—from reading the raw vibronic couplings for molecule
    `mol`, through mode‐selection or greedy fitting, to returning a list of PennyLane
    `RealspaceMatrix` fragments suitable for Trotterization.

    Parameters
    ----------
    mol : str
        Name of the molecule (used to load `mol/{mol}.pkl` containing `omegas` and `couplings`).
    p : int
        Target number of vibrational modes in the reduced model.
    type : {'FC', 'Greedy'}
        Fragmentation strategy.  
        - `'FC'` generates fragments by grouping Pauli‑decomposed blocks (``FC_grouping``).  
        - `'Greedy'` uses orthogonal rotations to fit and peel off the top couplings.
    subtype : str
        Sub‑type of the chosen strategy.  
        - If `type=='FC'`, valid subtypes include `'IZ-XY_grouping'`, `'IX-YZ_grouping'`, `'operator_norm'`.  
        - If `type=='Greedy'`, only `'fit_alpha_beta'` is currently supported.
    find : bool, optional
        If `True`, run the full fitting algorithm (or grouping) to generate fragments.  
        If `False`, load previously saved fragments via `greedy_Q_Real.load_fragments`.  
        Default is `True`.

    Returns
    -------
    List[RealspaceMatrix]
        A list of PennyLane `RealspaceMatrix` objects, each representing one fragment of the
        full vibronic Hamiltonian on a padded electronic subspace and `p` vibrational modes.

    Notes
    -----
    1. Mode selection: the top `p` modes are chosen by ranking each mode’s coupling
       strength (using `mode_selector.rank_modes` with “C1N+F”).  
    2. Coupling tensors: after selecting modes, the 0th, 1st, and 2nd‑order coupling arrays
       (`lambdas`, `alphas`, `betas`) are sliced to retain only these modes.  
    3. Fragment construction:
       - In FC‐grouping, the full matrix is decomposed into Pauli blocks, grouped by commutation,
         and reassembled into fragments.  
       - In Greedy, fragments that can be block diagonalized by unitary rotations.
    4. `Chatgpt` writes amazing docstrings.
    """

    filehandler = open(f'VCHLIB/{mol}.pkl', 'rb')
    omegas_total, couplings = pickle.load(filehandler)

    nstates, p_total = couplings[1].shape[0], couplings[1].shape[2]
    alphas_total = couplings[1]
    betas_total = couplings[2]

    n_blocks = int(pow(2, ceil(log2(nstates))))
    # omegas_r, couplings_r = get_reduced_model(omegas, couplings, m_max=p, order_max=2, strategy=None)

    # a bit redundant, i could have just modified the get_reduced_model function in rob's code to return modes_keep
    mode_measures = rank_modes(omegas_total, {order: couplings[order] for order in couplings if order > 0}, ranking_type="C1N+F")

    modes_keep = [mode for mode, _ in sorted(mode_measures.items(), key=lambda item: item[1], reverse=True)][:p]

    # these are the coupling coeffs for the reduced model
    omegas = omegas_total[modes_keep]
    lambdas = couplings[0]
    alphas = couplings[1][:, :, modes_keep]
    betas = couplings[2][:, :, modes_keep, :][:, :, :, modes_keep]

    '''
    so apperently betas isnt wasnt what i thought it would be. so i'll use it the same way when im using the default frag scheme,
    because will's code handles the adding the w/2 term in the realspacematrix. I need betas to be the coeffs of Q^2 so, for now,
    im going to sneak the w/2 coeff into the betas for my fragmentation codes.
    '''

    omegas_total = np.array(omegas_total)

    if type == 'FC': 
        if subtype == 'IZ-XY_grouping':
            fragments = vibronic_fragments(nstates, p, freqs=omegas, taylor_coeffs=[lambdas, alphas, betas])
            return fragments
        
        elif subtype == 'IX-YZ_grouping':
             raise ValueError("can't handle IX-YZ_grouping yet")
        
        else:
            # im sneaking in the harmonic term Q^2 into Betas here
            for i in range(nstates):
                for r in range(p):
                    betas[i, i, r, r] += omegas[r] / 2
                      
            M = couplings_to_RealSpaceMatrix(nstates, n_blocks, p, lambdas, alphas, betas)
            decomposition = decompose_to_pauli_terms(n_blocks, p, M)
            groups = fc_grouping(decomposition, subtype)

            FC_fragments = []
            for f in range(len(groups)):
                frag = RealspaceMatrix.zero(n_blocks, p)
                for term in groups[f]:
                    pauli = construct_pauli_matrix(term[0])
                    elec = matrix_to_RealSpaceMatrix(n_blocks, p, pauli)
                    sum = sum_to_RealSpaceMatrix(n_blocks, p, term[1])
                    term_matrix = sum@elec
                    frag += term_matrix
                FC_fragments.append(frag)

            FC_fragments.append(kin_frag(nstates, n_blocks, p, omegas))
            return FC_fragments
    
    elif type == 'Greedy' and subtype == 'fit_alpha_beta':

        # making the combined alpha and beta tensor for fitting
        V = np.zeros((n_blocks, n_blocks, 2*p_total))

        V[:nstates,:nstates,:p_total] = alphas_total[:,:,:]
        for j in range(nstates):
                for k in range(nstates):
                    for i in range(p_total):  
                        V[j][k][p_total + i] = betas_total[j][k][i][i]

                        # im sneaking in the harmonic term Q^2 into Betas here
                        if j == k:
                            V[j][k][p_total + i] += omegas_total[i]/2

        # finding the fragments from the full alphas and betas, without mode reduction.
        if find == True:
            greedy_fitted_couplings = find_fragments(ceil(log2(nstates)), p_total, V, 10)
            save_fragments(greedy_fitted_couplings, f'{mol}_{subtype}')

        else:
            greedy_fitted_couplings = load_fragments(f'{mol}_{subtype}')

        '''
        Greedy fitted couplings is a list of combined alphas and beta tensors,
        coefficients of the vibrational couplings which can be block diagonalized using
        the unitary rotations. Next we separate these tensors into alphas and betas,
        reduce the couplings to the p most important ones, and then convert them into 
        RealSpaceMatrices.
        '''
        Greedy_frags = []
        for V_frag,_ in greedy_fitted_couplings[:num_frags]:

            lambdas_frag = np.zeros((n_blocks, n_blocks))
            alphas_frag = np.zeros((n_blocks, n_blocks, p_total))
            betas_frag = np.zeros((n_blocks, n_blocks, p_total, p_total))

            alphas_frag[:,:,:] = V_frag[:,:,:p_total]
            for j in range(nstates):
                for k in range(nstates):
                    for i in range(p_total):  
                        betas_frag[j][k][i][i] = V_frag[j][k][p_total + i]
            
            # we're not including the constant couplings 
            alphas_frag = alphas_frag[:, :, modes_keep]
            betas_frag = betas_frag[:, :, modes_keep, :][:, :, :, modes_keep]

            # I'm adding h
            rsm_frag = couplings_to_RealSpaceMatrix(n_blocks, n_blocks, p, lambdas_frag, alphas_frag, betas_frag)
            Greedy_frags.append(rsm_frag)

        alphas_frag = np.zeros((nstates, nstates, p))
        betas_frag = np.zeros((nstates, nstates, p, p))
        kin = kin_frag(nstates, n_blocks, p, omegas) + couplings_to_RealSpaceMatrix(nstates, n_blocks, p, lambdas, alphas_frag, betas_frag)
        Greedy_frags.append(kin)
        return Greedy_frags
