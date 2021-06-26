import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 2: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    # (1) Initialize mu and sigma by splitting the n_examples data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    # *** START CODE HERE ***
    
    #Shuffle x
    np.random.shuffle(x)
    
    #Split n examples into K groups
    x_shuffle_split = np.array_split(x, K)
        
    #Initialize mu
    mu = [np.mean(x, axis=0) for x in x_shuffle_split]

    #Initialize sigma
    sigma = [np.cov(x.T) for x in x_shuffle_split]
    
    #Initialize phi
    phi = np.ones((K,))/K
      
    #Initialize w
    m = x.shape[0]
    w = np.ones((m, K))/K
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:

        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    n = x.shape[0]
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)
    

#Function to generate multivariate gaussian
def multivariate_gaussian(X,mu,sigma):

    d = len(mu)
    X = X-mu.T
    den = (np.sqrt((2 * np.pi)**d * np.linalg.det(sigma)))
    #Check the calculation of num once
    num = np.exp(-0.5*np.sum(X.dot(np.linalg.pinv(sigma))*X,axis=1))
    p = num / den

    return p



def run_em(x, w, phi, mu, sigma, max_iter=1000):
    """Problem 2(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim)
        max_iter: Max iterations. No need to change this

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    
    K = w.shape[1]
    
    
    
    
    #Loop over each cluster and find the probability of each point belonging to a particular cluster
    for i in range(K):
        w[:,i] = phi[i]*multivariate_gaussian(x, mu[i], sigma[i])
    
        

    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # (1) E-step: Update your estimates in w
        # (2) M-step: Update the model parameters phi, mu, and sigma
        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # *** START CODE HERE
        
        #E-Step: 
        
        #Update the previous log_likelihood
        prev_ll = ll

        #K = number of clusters
        K = w.shape[1]
        
        #n = number of examples
        n = w.shape[0]
        
        
        
        #******************************************
        # normalize over all possible cluster assignments
        # w.sum shape is (980,), w.sum does sum along columns for each row
        w = w / w.sum(axis = 1, keepdims = 1)
        
        # Calculating w_sum
        w_sum = np.sum(w, axis=0)
        
        #******************************************
        # Updating phi
        phi = np.array([w_sum[k]/n for k in range(K)])
        
        #******************************************
        # Updating mu
        mu_update = [np.zeros_like(mu[k]) for k in range(K)]
        for j in range(K):
            for i in range(n):
                mu_update[j] += w[i][j] * x[i]
        mu = [mu_update[j]/w_sum[j] for j in range(K)]
        
        #******************************************
        # Updating the covariance matrices
        sigma_update = [np.zeros_like(sigma[k]) for k in range(w.shape[1])]
        for j in range(K):
            for i in range(n):                
                diff = x[i] - mu[j]
                diff = diff.reshape(2,1)
                sigma_update[j] += np.dot(w[i][j]*diff, diff.T)
    
        sigma = [sigma_update[j]/w_sum[j] for j in range(K)]
        
        
        #******************************************
        #Loop over each cluster and find the probability of each point belonging to a particular cluster
        for i in range(K):
            w[:,i] = phi[i]*multivariate_gaussian(x, mu[i], sigma[i])
            
        #******************************************
        
        #Calculate Log Likelihood
        ll = np.sum(np.log(np.sum(w, axis = 1)))
        print("This is ll: ", ll)
        
        
        if it > 0 and np.abs(ll - prev_ll) < eps:
            print("Converged after {} iterations".format(it))
            break
  

        it += 1
        
        print("This is iter in unsupervised em: ", it)        
        # *** END CODE HERE ***

    return w


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma, max_iter=1000):
    """Problem 2(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim)
        max_iter: Max iterations. No need to change this

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # (1) E-step: Update your estimates in w
        # (2) M-step: Update the model parameters phi, mu, and sigma
        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # *** START CODE HERE ***
        
        print("Inside run_semi_supervised")
        
        #Update the previous log_likelihood
        prev_ll = ll

        K = phi.shape[0]
        
        #n = number of examples
        n = w.shape[0]
        n_tilde = x_tilde.shape[0]
        
        #******************************************
        #Function to generate multivariate gaussian
        def multivariate_gaussian_v2(X,mu,sigma):    
                
            d = len(mu)

            X = X-mu.T
            den = (np.sqrt((2 * np.pi)**d * np.linalg.det(sigma)))
            den_2 = ((2*np.pi)**(d/2)*np.linalg.det(sigma)**(0.5))
            res = X.dot(np.linalg.pinv(sigma))*X
            res = res.reshape(1, 2)
            res_sum = np.sum(res, axis=1)
            num = np.exp(-0.5*res_sum)
            p = num / den
            return p[0]

        #******************************************        
        #E-Step:
    
        #Loop over each cluster
        for i in range(K):
            w[:,i] = phi[i]*multivariate_gaussian(x, mu[i], sigma[i])
                        
        # normalize over all possible cluster assignments
        # w.sum shape is (980,), w.sum does sum along columns for each row
        w = w / w.sum(axis = 1, keepdims = 1)

        
        #E-Step over
        
        #******************************************

        
        #Initialize w_tilde
        w_tilde = np.zeros((x_tilde.shape[0], K))
#         z_tilde_prob = np.zeros((x_tilde.shape[0], K))
        
        
        for i in range(z_tilde.shape[0]):
            K_col = int(z_tilde[i,0])
            w_tilde[i,K_col] = 1
#             z_tilde_prob[i,K_col] = 1        
    
        #******************************************
        
        #M Step: Update the parameters
        
        # Calculating w_sum        
        w_sum = np.sum(w, axis=0)
        w_tilde_sum = np.sum(w_tilde, axis=0)
        weighted_sum = w_sum + alpha* w_tilde_sum


        #******************************************
        # Updating mu
        mu_update = [np.zeros_like(mu[k]) for k in range(K)]
        mu_update_tilde = [np.zeros_like(mu[k]) for k in range(K)]

        
        den_mu = 0
        den_mu_tilde = 0
        for j in range(K):
            for i in range(n):
                mu_update[j] += w[i][j] * x[i]
                den_mu += w[i][j]
                
        for j in range(K):
            for i in range(n_tilde):
                mu_update_tilde[j] += w_tilde[i][j] * x_tilde[i]
                den_mu_tilde += w_tilde[i][j]
            mu_update_tilde[j] = int(alpha) * mu_update_tilde[j]
                
         
        den_mu_tilde = alpha * den_mu_tilde
        mu = np.add(mu_update,mu_update_tilde) / weighted_sum.reshape((K,1))       

        #******************************************
        
        #Updating sigma
        sigma_update = [np.zeros_like(sigma[k]) for k in range(w.shape[1])]
        for j in range(K):
            for i in range(n):
                diff = x[i] - mu[j]
                diff = diff.reshape(2,1)
                sigma_update[j] += np.dot(w[i][j]*diff, diff.T)

        #Updating sigma_tilde
        sigma_update_tilde = [np.zeros_like(sigma[k]) for k in range(w_tilde.shape[1])]
        for j in range(K):
            for i in range(n_tilde):
                diff = x_tilde[i] - mu[j]
                diff = diff.reshape(2,1)
                sigma_update_tilde[j] += np.dot(w_tilde[i][j]*diff, diff.T)
            sigma_update_tilde[j] = int(alpha) * sigma_update_tilde[j]

        sigma = np.add(sigma_update,sigma_update_tilde) / weighted_sum.reshape((K,1,1))

                                       
            
        #******************************************
        # Updating phi
        for k in range(K):
            num = w_sum[k] + alpha*w_tilde_sum[k]
            den = n + alpha*n_tilde
            phi[k] = num/den        
        
        
        #******************************************
        # Calculating unsupervised likelihood
        
        #Loop over each cluster
        for i in range(K):
            w[:,i] = phi[i]*multivariate_gaussian(x, mu[i], sigma[i])
        
        #Calculate Log Likelihood of Unsupervised Data
        ll_unsup = np.sum(np.log(np.sum(w, axis = 1)))
        print("This is ll_unsup: ", ll_unsup)
        
        
        
        #For calculating ll_sup
        ll_sup = 0
        for i in range(z_tilde.shape[0]):
            
            K_col = int(z_tilde[i,0])
            ll_sup = ll_sup + np.log(phi[K_col]*multivariate_gaussian_v2(x_tilde[i,:], mu[K_col], sigma[K_col]))
    
        #Calculate ll_sup
        print("This is ll_sup: ", ll_sup)
        
            
        #******************************************    
        #Calculate Log Likelihood of supervised Data
        ll = ll_unsup + int(alpha)*ll_sup
        print("THis is final ll: ",ll)
        
        

       
        if it > 0 and np.abs(ll - prev_ll) < eps:
           print("Algorithm converged")
           break
  

        it += 1
        
        print("This is iter in run supervised em: ", it)
        print("This is ll: ", ll)
        # *** END CODE HERE ***

    return w


# *** START CODE HERE ***
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)
        main(is_semi_supervised=True, trial_num=t)
