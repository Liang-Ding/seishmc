# -------------------------------------------------------------------
# Moment tensor inversion using Hamiltonian Monte Carlo sampling
# The base class
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

import os.path
import json
import numpy as np
import pickle
import time

class DHMCBase():
    '''The base class of HMC sampling'''

    def __init__(self, saving_dir,
                 b_save_cache=True, n_step_cache=10,
                 verbose=False):

        ##############################################################
        # Following parameters associated with Nq must be implemented by subclass.
        self.Nq = 2
        # Use identity matrix as the mass matrix by default.
        self.massinv    = np.eye(self.Nq)
        self.delta_q0   = 0.1 * np.ones(self.Nq)
        self.q0         = np.zeros(self.Nq)
        self.raw_q0     = np.zeros(self.Nq)
        self.dU_dqi     = np.zeros(self.Nq)
        # solution at lowest Uq (misfit)
        self.sol_misfit = np.zeros(self.Nq)
        ##############################################################

        # default parameters for sampling
        # constant encoding the uncertainties in observations
        self.sigma_d = 0.1
        self.sigma_q = np.inf

        # The saving directory
        self.saving_dir = saving_dir
        if not os.path.exists(self.saving_dir):
            os.makedirs(self.saving_dir)

        # Saving cache.
        self.b_save_cache = b_save_cache
        self.n_step_cache = n_step_cache

        self.b_init_epsilon = False
        self.b_init_n_step = False
        self.b_initial_q0 = False

        # Terminate if exceed the MAX_SAMPLE.
        self.MAX_SAMPLE = 100000
        self.verbose = verbose


    def _check(self):
        '''Check the required parameters.'''
        if (False in [self.b_init_n_step, self.b_init_epsilon, self.b_initial_q0]):
            return False
        else:
            return True

    def create_p(self):
        '''Using probability density function to create P. '''
        return np.random.normal(loc=0, scale=1.0, size=self.Nq)

    def Kp(self, p):
        '''Compute Kp = 1/2P'MP, M:mass matrix. '''
        return 0.5 * p.T @ (self.massinv @ p)

    def Uq(self, q):
        raise NotImplementedError("Must be implemented by subclass")

    def set_q(self, q):
        raise NotImplementedError("Must be implemented by subclass")

    def dUdq(self, q):
        raise NotImplementedError("Must be implemented by subclass")

    def set_epsilon(self, min, max):
        '''Set function to generate epsilon in sampling.'''
        self.min_epsilon = min
        self.max_epsilon = max
        self.b_init_epsilon = True

    def get_epsilon(self):
        '''Get epsilon'''
        if not self.b_init_epsilon:
            raise ValueError("Uninitialized epsilon !!!")
        else:
            return np.random.uniform(self.min_epsilon, self.max_epsilon)

    def set_n_step(self, min, max=None):
        '''Set nstep '''
        self.min_n_step = np.uint(min)
        self.max_n_step = np.uint(max)
        self.b_init_n_step = True

    def get_n_step(self):
        '''Get n_Step'''
        if not self.b_init_n_step:
            raise ValueError("Uninitialized n_step !!!")
        else:
            return int(np.random.uniform(self.min_n_step, self.max_n_step))

    def _find_solution(self, samples, misfit):
        '''Determine the solution based on the lowest misfit. '''
        self.sol_misfit = np.array(samples[np.where(misfit == np.min(misfit))[0][0]])

    def _get_solution(self):
        '''Return the solution.'''
        return self.sol_misfit

    def hmc(self):
        '''Hamiltonian Monte Carlo Sampling.'''
        if not self.b_initial_q0:
            ValueError("Uninitialized q0!")

        # Set the q(t=0) and p(t=0).
        q = self.q0.copy()
        current_q = self.q0.copy()
        p = self.create_p()
        current_p = p.copy()

        # set step size and number of step
        epsilon = self.get_epsilon()
        n_step = self.get_n_step()

        for i in range(n_step):
            # p(t+epsilon/2)
            p = p - (epsilon / 2.0) * self.dU_dqi
            # q(t+epsilon)
            q = q + epsilon * self.massinv @ p
            self.set_q(q)
            p = p - (epsilon / 2.0) * self.dU_dqi

        current_Uq = self.Uq(current_q)
        current_Kp = self.Kp(current_p)
        proposed_Uq = self.Uq(q)
        proposed_Kp = self.Kp(p)
        denergy = current_Uq - proposed_Uq + current_Kp - proposed_Kp

        # accept the sample with probability
        if np.isnan(denergy): denergy = -np.inf
        prob_accept = min(0, denergy)
        alpha = np.log(np.random.rand())
        # accept q(t=n_step), use it as the new q(t=0)
        if alpha < prob_accept:
            self.set_q(q)
            if self.verbose:
                print("! Accept q={}".format(np.round(self.q0, 2)))
            return self.q0, True, proposed_Uq
        # reject q(t=n_step), return to q(t=0)
        else:
            if self.verbose:
                print("Reject, q={}".format(np.round(q, 2)))
            self.set_q(current_q)
            return current_q, False, proposed_Uq


    def sampling(self, n_sample, task_id='hmc01'):
        '''Sampling process'''
        if not self.b_initial_q0:
            raise ValueError("q0 must be initialized!")

        if not self._check():
            raise ValueError("Initialized parameters: n_step, epsilon, ...")

        t0 = time.time()
        # data container
        samples = []
        misfit = []
        _tmp_n_sample = 0
        while (len(samples) < n_sample and _tmp_n_sample < self.MAX_SAMPLE):
            _tmp_n_sample += 1
            try:
                q, accept, U = self.hmc()
                if accept:
                    samples.append(q)
                    misfit.append(U)
            except:
                # reset
                self.set_q(self.raw_q0.copy())

            # save cache
            if self.b_save_cache:
                if len(samples) != 0 and np.mod(len(samples), self.n_step_cache) == 0:
                    self.save_samples('%s_cache' % task_id, samples, misfit)

        # save all samples
        self.save_samples('%s_samples' % task_id, samples, misfit)

        # determine solution based on waveform misfit
        self._find_solution(samples, misfit)

        # gather the information of HMC sampling
        dt = '%d' % (time.time() - t0)
        ar = '%.2f' % (len(samples) / _tmp_n_sample)    # accept rate
        info_dict = {
            'Task_id': task_id,
            'time_elapse_in_s': dt,
            'N_sample': len(samples),
            'Accept_rate': ar,
            'Nq': self.Nq,
            'Sigma_d': self.sigma_d,
            "q0": str(self.raw_q0),
            "sol_misfit": str(samples[np.where(misfit == np.min(misfit))[0][0]]),
        }
        self.save_info('%s_info' % task_id, info_dict)


    def save_samples(self, file_name, samples, misfit, fmt='pkl'):
        ''' Save all accepted samples to a PKL file. '''

        if not os.path.exists(self.saving_dir):
            os.makedirs(self.saving_dir)

        file_path = os.path.join(self.saving_dir, "%s_N%d.%s"%(file_name, len(samples), fmt))
        with open(file_path, 'wb') as f:
            pickle.dump(samples, f)
            pickle.dump(misfit, f)

    def save_info(self, file_name, info_dict, fmt='json'):
        '''Save the information into a json file'''

        if not os.path.exists(self.saving_dir):
            os.makedirs(self.saving_dir)

        file_path = os.path.join(self.saving_dir, "%s.%s"%(file_name, fmt))
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(info_dict, file, indent=4)
