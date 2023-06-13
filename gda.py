import numpy as np
import util


def main(train_path, valid_path, save_path):
    """
    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    clf = GDA()
    clf.fit(x_train, y_train)

    # Plot decision boundary on validation set
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=False)
    plot_path = save_path.replace('.txt', '.png')
    util.plot(x_eval, y_eval, clf.theta, plot_path)

    x_eval = util.add_intercept(x_eval)
    p_eval = clf.predict(x_eval)
    y_hat = p_eval > 0.5
    print('GDA Accuracy: %.2f' % np.mean( (y_hat == 1) == (y_eval == 1)))
    np.savetxt(save_path, p_eval)


class GDA:

    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """
        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """

        m, n = x.shape

        if self.theta is None:
            self.theta = np.zeros(n + 1)

        phi = np.mean(y)
        mu_1 = y.dot(x) / np.sum(y)
        y_hat = 1 - y
        mu_0 = y_hat.dot(x) / (np.sum(y_hat))
        mu_y = np.zeros_like(x)

        for i in range(m):
            if y[i] == 0:
                mu_y[i] = mu_0
            else:
                mu_y[i] = mu_1
        
        sigma = (1/m) * (x - mu_y).T.dot(x - mu_y)
        sigma_inverse = np.linalg.inv(sigma)
  
        self.theta[1:] = -np.dot(sigma_inverse, (mu_0 - mu_1))
        self.theta[0] = (1/2) * (mu_0.T.dot(sigma_inverse).dot(mu_0) - mu_1.T.dot(sigma_inverse).dot(mu_1))  - np.log((1 - phi)/ phi)
        
        if self.verbose:
            print('Final theta (GDA): {}'.format(self.theta))

    def predict(self, x):
        """
        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        y_hat = 1 / (1 + np.exp(- (x.dot(self.theta))))
        
        return y_hat

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
