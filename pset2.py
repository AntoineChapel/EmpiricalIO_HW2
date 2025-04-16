import pandas as pd
import numpy as np
from scipy.optimize import minimize

T = 5_000
M = np.array([[0.6, 0.4], [0.4, 0.6]])
X_vals = np.array([0, 1])
theta_true = np.array([-2, 6, -2.5, 4])
R = 0.95

X = np.empty((T))
X[0] = 0
for t in range(1, T):
    X[t] = np.random.choice(X_vals, p=M[int(X[t-1]), :])

def p_fixed_point_iteration(p, theta, Xt, Eti, Etj, M):
    R = 0.95
    alpha, beta, gamma, c = theta
    # Current p(Xt, Eti, Etj)
    p_current = p[int(Xt), int(Etj), int(Eti)]

    # Integral term over x' (next state)
    integral = 0.0
    for x_prime in [0, 1]:
        prob_x_prime = M[int(Xt), x_prime]
        log_term = (
            p_current * np.log(1 - np.clip(p[x_prime, 1, 1], 1e-10, 1-1e-10)) +
            (1 - p_current) * np.log(1 - np.clip(p[x_prime, 1, 0], 1e-10, 1-1e-10)))
        integral += prob_x_prime * log_term

    # RHS of the equation
    rhs = (
        alpha + beta * Xt + gamma * p_current - c * (Eti == 0) + 
        R * np.euler_gamma - R * integral
    )
    return 1 / (1 + np.exp(-rhs))

def F(p, theta, M):
    p_next = np.zeros_like(p)
    for Xt in [0, 1]:
        for Eti in [0, 1]:
            for Etj in [0, 1]:
                p_next[Xt, Eti, Etj] = p_fixed_point_iteration(p, theta, Xt, Eti, Etj, M)
    return p_next

def fixed_point(theta, M, tol=1e-6, max_iter=1000, verbose=False):
    p_init = np.random.uniform(0.1, 0.9, size=(2, 2, 2))
    itercount = 0
    for _ in range(max_iter):
        p_next = F(p_init, theta, M)
        if np.max(np.abs(p_next - p_init)) < tol:
            break
        if verbose:
            print(f"Iteration {itercount}: max diff = {np.max(np.abs(p_next - p_init))}")
        p_init = p_next
        itercount += 1
        
    return p_next

p_true = fixed_point(theta_true, M, verbose=False)

E = np.empty((T, 2))
E[0, :] = np.array([0, 1])

###Simulate choices
actions = np.empty((T, 2))
for t in range(T):
    player0_proba_action = p_true[int(X[t]), int(E[t, 0]), int(E[t, 1])]
    player1_proba_action = p_true[int(X[t]), int(E[t, 1]), int(E[t, 0])]
    player0_action = np.random.choice([0, 1], p=[1 - player0_proba_action, player0_proba_action])
    player1_action = np.random.choice([0, 1], p=[1 - player1_proba_action, player1_proba_action])
    actions[t, :] = np.array([player0_action, player1_action])
    if t < T - 1:
        E[t+1, 0] = player0_action
        E[t+1, 1] = player1_action


df = pd.DataFrame(np.hstack((X.reshape(-1, 1), E, actions)))
df.columns = ['X', 'E0', 'E1', 'A0', 'A1']
df = df.astype(int)
df_np = df.values


### Basis for this function obtained from DeepSeek
def estimate_M_hat(sequence):
    counts = np.zeros((2, 2))
    for i in range(len(sequence)-1):
        current = sequence[i]
        next = sequence[i+1]
        counts[current, next] += 1
    
    row_sums = counts.sum(axis=1, keepdims=True)
    transition_matrix = counts / row_sums
    
    return transition_matrix


M_hat = estimate_M_hat(df_np[:, 0])

def nfxp_log_likelihood(theta):
    CCP = fixed_point(theta, M_hat, verbose=False)
    logL = 0
    for t in range(T):
        ###firm 1:
        Xt = df_np[t, 0]
        Et_own = df_np[t, 1]
        Et_other = df_np[t, 2]
        action_own = df_np[t, 3]
        logL += np.log(action_own*CCP[Xt, Et_own, Et_other] 
                       + (1-action_own)*(1-CCP[Xt, Et_own, Et_other]))
        ###firm 2:
        Et_own = df_np[t, 2]
        Et_other = df_np[t, 1]
        action_own = df_np[t, 4]
        logL += np.log(action_own*CCP[Xt, Et_own, Et_other] 
                       + (1-action_own)*(1-CCP[Xt, Et_own, Et_other]))
    return -logL

## To facilitate convergence, we start close to the true parameter

x0_val = theta_true+np.random.uniform(-0.1, 0.1, size=theta_true.shape)


result_nfxp = minimize(nfxp_log_likelihood,x0=x0_val, method='Nelder-Mead')

theta_nfxp = result_nfxp.x

# ## Estimation via Hotz-Miller

def estimate_p_hat(df):
    p_hat = np.zeros((2, 2, 2))
    p_hat[0, 0, 0] = df.query('X == 0 & E0 == 0 & E1 == 0')[['A0', 'A1']].values.mean()
    p_hat[0, 0, 1] = (df.query('X == 0 & E0 == 0 & E1 == 1')['A0'].mean() \
                   + df.query('X == 0 & E0 == 1 & E1 == 0')['A1'].mean())/2
    p_hat[0, 1, 0] = (df.query('X == 0 & E0 == 1 & E1 == 0')['A0'].mean() \
                   + df.query('X == 0 & E0 == 0 & E1 == 1')['A1'].mean())/2
    p_hat[0, 1, 1] = df.query('X == 0 & E0 == 1 & E1 == 1')[['A0', 'A1']].values.mean()

    p_hat[1, 0, 0] = df.query('X == 1 & E0 == 0 & E1 == 0')[['A0', 'A1']].values.mean()
    p_hat[1, 0, 1] = (df.query('X == 1 & E0 == 0 & E1 == 1')['A0'].mean() \
                   + df.query('X == 1 & E0 == 1 & E1 == 0')['A1'].mean())/2
    p_hat[1, 1, 0] = (df.query('X == 1 & E0 == 1 & E1 == 0')['A0'].mean() \
                   + df.query('X == 1 & E0 == 0 & E1 == 1')['A1'].mean())/2
    p_hat[1, 1, 1] = df.query('X == 1 & E0 == 1 & E1 == 1')[['A0', 'A1']].values.mean()
    return p_hat
p_hat = np.clip(estimate_p_hat(df), 1e-10, 1-1e-10)

def CCP_tilde(theta):
    alpha, beta, gamma, c = theta
    V_tilde = np.zeros((2, 2, 2))
    for x in [0, 1]:
        for e_own in [0, 1]:
            for e_other in [0, 1]:
                p_current = p_hat[x, e_own, e_other]
                integral = M_hat[int(x), 0]*(p_current*np.log(1-p_hat[0, 1, 1]) + (1-p_current)*np.log(1-p_hat[0, 1, 0])) \
                         + M_hat[int(x), 1]*(p_current*np.log(1-p_hat[1, 1, 1]) + (1-p_current)*np.log(1-p_hat[1, 1, 0]))           
                V_tilde[x, e_own, e_other] = alpha + beta*x \
                                        + gamma*p_hat[x, e_other, e_own] - c * (e_own == 0) + R*np.euler_gamma \
                                        - R*integral
    CCP_tilde = np.clip(1 / (1 + np.exp(-V_tilde)), 1e-10, 1-1e-10)
    return CCP_tilde

def hm_log_likelihood(theta):
    CCP = CCP_tilde(theta)
    logL = 0
    for t in range(T):
        ###firm 1:
        Xt = df_np[t, 0]
        Et_own = df_np[t, 1]
        Et_other = df_np[t, 2]
        action_own = df_np[t, 3]
        logL += np.log(action_own*CCP[Xt, Et_own, Et_other] 
                       + (1-action_own)*(1-CCP[Xt, Et_own, Et_other]))
        ###firm 2:
        Et_own = df_np[t, 2]
        Et_other = df_np[t, 1]
        action_own = df_np[t, 4]
        logL += np.log(action_own*CCP[Xt, Et_own, Et_other] 
                       + (1-action_own)*(1-CCP[Xt, Et_own, Et_other]))
    return -logL

x0_val = theta_true+np.random.uniform(-0.1, 0.1, size=theta_true.shape)
result_hm = minimize(hm_log_likelihood,x0=x0_val, method='Nelder-Mead')

theta_hm = result_hm.x

###Results
print("True parameters: ", theta_true)
print("NFXP estimated parameters: ", theta_nfxp)
print("HM estimated parameters: ", theta_hm)