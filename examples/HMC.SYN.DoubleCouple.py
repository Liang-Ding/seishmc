#!/usr/bin/env python

import os

from seishmc.DHMC.dc import DHMC_DC
from seishmc.visualization.viz_samples_dc import pairplot_samples_DC

from mtuq import read, open_db
from mtuq.event import Origin
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath
from mtuq.util.cap import parse_station_codes, Trapezoid
from mtuq.graphics import plot_beachball, plot_data_greens1

import numpy as np
# Set the random seed using our lab's room number.
np.random.seed(511)


if __name__=='__main__':
    #
    # Synthetic example of moment tensor inversion using Hamiltonian Monte Carlo (HMC) sampling.
    #
    path_data   = '../data/examples/synthetic/data/*.[zrt]'
    path_greens = '../data/examples/synthetic/greens'
    path_weights= '../data/examples/synthetic/weights.dat'
    event_id    = 'syn_example'
    model       = 'socal3D'
    taup_model  = 'ak135'

    # folder storing the cache, samples, and inversion results.
    saving_dir = "../output/examples/synthetic/HMC_DC/"

    #
    # Synthetic waveform
    #

    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=0.033333,
        freq_max=0.125,
        taup_model=taup_model,
        apply_scaling=False,
        window_type='synthetic',
        window_length=100.,
        capuaf_file=path_weights,
        )

    #
    # For our objective function, we will use zero time shift.
    #

    misfit_sw = Misfit(
        norm='L2',
        time_shift_min=-0.,
        time_shift_max=+0.,
        time_shift_groups=['ZR','T'],
        )

    #
    # User-supplied weights control how much each station contributes to the
    # objective function
    #

    station_id_list = parse_station_codes(path_weights)

    #
    # Next, we specify the source-time function
    #

    wavelet = Trapezoid(
        magnitude=3.0)

    #
    # Origin time and location will be fixed.
    #

    origin = Origin({
        'time': '2019-07-05T12:38:30.0000Z',
        'latitude': 35.771667,
        'longitude': -117.571,
        'depth_in_m': 6820.0,
        'id': '11057910'
        })


    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    #
    # The main I/O work starts now
    #
    if comm.rank==0:
        print('Reading data...\n')
        data = read(path_data, format='sac',
            event_id=event_id,
            station_id_list=station_id_list,
            tags=['units:cm', 'type:displacement'])

        data.sort_by_distance()
        stations = data.get_stations()

        print('Processing data...\n')
        data_sw = data.map(process_sw)

        print('Load Greens functions...\n')
        db = open_db(path_greens, format='SPECFEM3D_SGT', model='socal3D')
        greens = db.get_greens_tensors(stations, origin)

        print('Processing Greens functions...\n')
        greens.convolve(wavelet)
        greens_sw = greens.map(process_sw)

    else:
        stations = None
        data_sw = None
        greens_sw = None

    stations = comm.bcast(stations, root=0)
    data_sw = comm.bcast(data_sw, root=0)
    greens_sw = comm.bcast(greens_sw, root=0)

    #
    # The main computational work starts now
    #

    rank = comm.Get_rank()
    print('Sampling with HMC...\n')
    solver_hmc = DHMC_DC(None, None, None,
                            misfit_sw, data_sw, greens_sw,
                            saving_dir, b_save_cache=True,
                            n_step_cache=500, verbose=True)

    # set the range of number of step
    # solver_hmc.set_n_step(min=3, max=10)  # short chain, faster
    solver_hmc.set_n_step(min=20, max=50)   # long chain, slower

    # set the range of step interval
    solver_hmc.set_epsilon(min=0.01, max=1.0)
    # set the number of sample to be accepted
    n_sample = 2000

    # set initial solution in degree and Mw
    q0 = np.array([np.random.uniform(0, 360),
                   np.random.uniform(0, 90),
                   np.random.uniform(-90, 90),
                   np.random.uniform(2.8, 3.2)])
    solver_hmc.set_q(q0)

    # sampling...
    task_id = 'syn_DC_hmc_%d' % rank
    solver_hmc.sampling(n_sample=n_sample, task_id=task_id)

    print('Generating figures...\n')
    data_file = os.path.join(saving_dir, "%s_samples_N%d.pkl" % (task_id, n_sample))
    fig_path = os.path.join(saving_dir, task_id)

    # the reference solution is utilized for plotting only.
    ref_sol = np.array([30., 45., 60., 2.95])
    pairplot_samples_DC(file_path=data_file, fig_saving_path=fig_path, init_sol=q0, ref_sol=ref_sol)
    # if no reference solution provided, plot with:
    # pairplot_samples_DC(file_path=data_file, fig_saving_path=fig_path, init_sol=q0, ref_sol=None)

    # Get solution
    best_mt, lune_dict = solver_hmc.get_solution()

    fig_path = os.path.join(saving_dir, '%s_waveforms.png' % task_id)
    plot_data_greens1(fig_path,
                      data_sw,
                      greens_sw,
                      process_sw,
                      misfit_sw,
                      stations,
                      origin,
                      best_mt,
                      lune_dict)

    fig_path = os.path.join(saving_dir, '%s_beachball.png' % task_id)
    plot_beachball(fig_path, best_mt, stations, origin)

    MPI.Finalize()
    print('\nFinished\n')
