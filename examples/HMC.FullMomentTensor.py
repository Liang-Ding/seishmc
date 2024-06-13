#!/usr/bin/env python

import os

from seishmc.DHMC.fmt import DHMC_FMT
from seishmc.visualization.viz_samples_fmt import pairplot_samples_FMT

from mtuq import read, open_db
from mtuq.event import Origin
from mtuq.graphics import plot_data_greens2, plot_beachball
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath
from mtuq.util.cap import parse_station_codes, Trapezoid

import numpy as np
# Set the random seed using our lab's room number.
np.random.seed(511)


if __name__=='__main__':
    #
    # Moment tensor inversion using Hamiltonian Monte Carlo (HMC) sampling
    # Full moment tensor solution
    #
    # USAGE
    #   python HMC.FullMomentTensor.py
    #   or
    #   mpirun -n <NPROC> python HMC.FullMomentTensor.py
    #

    #
    # Real data example
    #

    path_data   = '../data/examples/SPECFEM3D/data/*.[zrt]'
    path_greens = '../data/examples/SPECFEM3D/greens/socal3D'
    path_weights= '../data/examples/SPECFEM3D/weights.dat'
    event_id    = 'evt11071294'
    model       = 'socal3D'
    taup_model  = 'ak135'

    # output folder
    saving_dir = '../output/examples/SPECFEM3D/HMC_FMT'

    #
    # Body and surface wave measurements will be made separately
    #

    process_bw = ProcessData(
        filter_type='Bandpass',
        freq_min= 0.05,
        freq_max= 0.125,
        pick_type='taup',
        taup_model=taup_model,
        window_type='body_wave',
        window_length=30.,
        capuaf_file=path_weights,
        )

    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=0.033333,
        freq_max=0.125,
        pick_type='taup',
        taup_model=taup_model,
        window_type='surface_wave',
        window_length=100.,
        capuaf_file=path_weights,
        )


    #
    # For our objective function, we will use a sum of body and surface wave
    # contributions
    #

    misfit_bw = Misfit(
        norm='L2',
        time_shift_min=-3.,
        time_shift_max=+3.,
        time_shift_groups=['ZR'],
        )

    misfit_sw = Misfit(
        norm='L2',
        time_shift_min=-3.,
        time_shift_max=+3.,
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
        magnitude=4.8)

    #
    # Origin time and location will be fixed.
    #

    origin = Origin({
        'time': '2019-07-12T13:11:37.0000Z',
        'latitude': 35.638333,
        'longitude': -117.585333,
        'depth_in_m': 9950.0,
        'id': 'evt11071294'
        })


    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    #
    # The main I/O work starts now
    #

    if comm.rank==0:
        print('Reading data...\n')
        data = read(path_data, format='sac',
            event_id=event_id,
            station_id_list=station_id_list,
            tags=['units:cm', 'type:velocity'])


        data.sort_by_distance()
        stations = data.get_stations()

        print('Processing data...\n')
        data_bw = data.map(process_bw)
        data_sw = data.map(process_sw)

        print('Reading Greens functions...\n')
        db = open_db(path_greens, format='SPECFEM3D_SGT', model='socal3D')
        greens = db.get_greens_tensors(stations, origin)

        print('Processing Greens functions...\n')
        greens.convolve(wavelet)
        greens_bw = greens.map(process_bw)
        greens_sw = greens.map(process_sw)

    else:
        stations = None
        data_bw = None
        data_sw = None
        greens_bw = None
        greens_sw = None

    stations = comm.bcast(stations, root=0)
    data_bw = comm.bcast(data_bw, root=0)
    data_sw = comm.bcast(data_sw, root=0)
    greens_bw = comm.bcast(greens_bw, root=0)
    greens_sw = comm.bcast(greens_sw, root=0)


    #
    # The main computational work starts now
    #

    rank = comm.Get_rank()
    print('Initialize HMC.\n')
    solver_hmc = DHMC_FMT(misfit_bw, data_bw, greens_bw,
                              misfit_sw, data_sw, greens_sw,
                              saving_dir, b_save_cache=True,
                              n_step_cache=500, verbose=True)

    # set the range of number of step
    solver_hmc.set_n_step(min=3, max=10)

    # set the range of step interval
    solver_hmc.set_epsilon(min=0.05, max=1.0)

    # set sigma_d
    solver_hmc.set_sigma_d(0.05)

    # set the number of accepted samples
    n_sample = 1000

    # set initial solution in degree and Mw for FMT solver
    # [strike, dip, rake, mag, co-lat., lune-lon.]
    q0 = np.array([np.random.uniform(0, 360),
                   np.random.uniform(0, 90),
                   np.random.uniform(0, 180),
                   np.random.uniform(4.5, 5.),
                   np.random.uniform(0, 180),
                   np.random.uniform(-30, 30)])
    solver_hmc.set_q(q0)

    print('Sampling ...\n')
    task_id = '%s_FMT_HMC_%d' % (event_id, rank)
    solver_hmc.sampling(n_sample=n_sample, task_id=task_id)


    print('Generating figures...\n')
    data_file = os.path.join(saving_dir, "%s_samples_N%d.pkl" % (task_id, n_sample))
    fig_path = os.path.join(saving_dir, task_id)

    pairplot_samples_FMT(file_path=data_file, fig_saving_path=fig_path, init_sol=q0)

    # Get the solution
    best_mt, lune_dict = solver_hmc.get_solution()

    fig_path = os.path.join(saving_dir, '%s_waveforms.png' % task_id)
    plot_data_greens2(fig_path,
                      data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw,
                      misfit_bw, misfit_sw, stations, origin, best_mt, lune_dict)

    fig_path = os.path.join(saving_dir, '%s_beachball.png' % task_id)
    plot_beachball(fig_path, best_mt, stations, origin)

    MPI.Finalize()
    print('\nFinished\n')
