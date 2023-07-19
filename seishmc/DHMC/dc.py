# -------------------------------------------------------------------
# Moment tensor inversion using Hamiltonian Monte Carlo sampling
# Double-couple (DC) solutions
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

from seishmc.utils import wrap_strike, wrap_rake, wrap_dip
from seishmc.DHMC.base import DHMCBase

from mtuq.grid import UnstructuredGrid
from mtuq.util.math import to_mij, to_rho
from mtuq.grid.moment_tensor import to_mt
from mtuq.event import MomentTensor
import numpy as np

class DHMC_DC(DHMCBase):
    ''' Utilizing the Hamiltonian Monte Carlo sampling method to sample the double-couple solutions.'''

    def __init__(self, misfit_bw, data_bw, greens_bw,
                 misfit_sw, data_sw, greens_sw,
                 saving_dir, b_save_cache=True, n_step_cache=100,
                 verbose=False):

        super(DHMC_DC, self).__init__(saving_dir, b_save_cache, n_step_cache, verbose)

        # double-couple solutions
        # [strike, dip, rake, magnitude]
        self.Nq = 4
        # mass matrix
        self.massinv = np.eye(self.Nq)
        self.massinv[3, 3] = 1.0 / 1000.    # magnitude
        # delta to calculate derivatives.
        self.delta_q0 = 0.2 * np.ones(self.Nq)
        self.delta_q0[3] = 0.02  # magnitude
        self.q0 = np.zeros(self.Nq)
        self.raw_q0 = np.zeros(self.Nq)
        self.dU_dqi = np.zeros(self.Nq)
        self.sol_misfit = np.zeros(self.Nq)

        # body wave
        self.misfit_bw = misfit_bw
        self.data_bw = data_bw
        self.greens_bw = greens_bw
        # surface wave
        self.misfit_sw = misfit_sw
        self.data_sw = data_sw
        self.greens_sw = greens_sw

        self.calc_scaling_factor()


    def get_solution(self):
        '''Get solution'''

        sol = self._get_solution()
        kappa   = sol[0]  # strike
        h       = np.cos(np.deg2rad(sol[1]))  # dip
        sigma   = sol[2]  # rake
        rho     = to_rho(sol[3])  # magnitude: Mw
        v       = 0.
        w       = 0.
        mij     = to_mij(rho, v, w, kappa, sigma, h)

        # lune dict
        self.lune_dict = dict({'rho': rho,
                          'v': v,
                          'w': w,
                          'kappa': kappa,
                          'sigma': sigma,
                          'h': h, })
        self.best_mt = MomentTensor(mij)
        return self.best_mt, self.lune_dict


    def calc_scaling_factor(self):
        '''Compute misfit scaling factor based on the data.'''
        self.scaling = 0.0

        # body wave
        if self.data_bw is not None and self.greens_bw is not None:
            for _st in self.data_bw:
                for _tr in _st:
                    self.scaling += np.dot(_tr.data, _tr.data)

        # surface wave
        if self.data_sw is not None and self.greens_sw is not None:
            for _st in self.data_sw:
                for _tr in _st:
                    self.scaling += np.dot(_tr.data, _tr.data)

        if self.scaling == 0.0:
            self.scaling = 1.0


    def Uq(self, q):
        '''Compute Uq and gradient'''

        npts = self.Nq + 1
        v = np.zeros(npts)
        w = np.zeros(npts)
        kappa = q[0] * np.ones(npts)                    # strike
        h = np.cos(np.deg2rad(q[1])) * np.ones(npts)    # dip
        sigma = q[2] * np.ones(npts)                    # rake
        rho = to_rho(q[3]) * np.ones(npts)              # magnitude: Mw

        # x+dx
        kappa[1] += self.delta_q0[0]
        h[2] = np.cos(np.deg2rad(q[1]+self.delta_q0[1]))
        sigma[3] += self.delta_q0[2]
        rho[4] = to_rho(q[3]+self.delta_q0[3])

        q_grid = UnstructuredGrid(
            dims=('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
            coords=(rho, v, w, kappa, sigma, h),
            callback=to_mt)

        misfit = np.zeros([npts, 1])

        try:
            if self.data_bw is not None and self.greens_bw is not None:
                res_bw = self.misfit_bw(self.data_bw, self.greens_bw, q_grid)
                misfit += res_bw
        except:
            pass
        try:
            if self.data_sw is not None and self.greens_sw is not None:
                res_sw = self.misfit_sw(self.data_sw, self.greens_sw, q_grid)
                misfit += res_sw
        except:
            pass

        # misfit scaling
        misfit /= self.scaling

        Uq = misfit / (2.0 * np.power(self.sigma_d, 2))
        # gradient: dU/dqi = (y(x+dx) - y(x))/dx
        self.dU_dqi = np.zeros(self.Nq)
        for i in range(self.Nq):
            self.dU_dqi[i] = (Uq[i+1][0] - Uq[0][0]) / self.delta_q0[i]

        self.dU_dqi[3] /= 100.   # magnitude
        return Uq[0][0]


    def set_q(self, q):
        '''
        Set position: q.
        q = [strike, dip, rake, magnitude] in degree and Mw
        '''

        if len(q) < self.Nq:
            ValueError("q should contain [strike, dip, rake, magnitude].")

        self.q0 = np.zeros(self.Nq)
        self.q0[0] = wrap_strike(q[0])
        self.q0[1] = wrap_dip(q[1])
        self.q0[2] = wrap_rake(q[2])
        self.q0[3] = q[3]

        if not self.b_initial_q0:
            self.raw_q0 = self.q0.copy()

        self.b_initial_q0 = True
        self.Uq(self.q0)

