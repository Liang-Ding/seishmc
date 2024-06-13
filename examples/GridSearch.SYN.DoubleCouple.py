#!/usr/bin/env python

import os

from mtuq import read, open_db
from mtuq.event import Origin
from mtuq.graphics import plot_beachball, plot_misfit_dc, plot_data_greens1
from mtuq.grid import DoubleCoupleGridRegular
from mtuq.grid_search import grid_search
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid

import numpy as np


if __name__=='__main__':
    #
    # Synthetic example of moment tensor inversion using Grid Search (MTUQ).
    # Double-couple solution
    #

    path_data   = '../data/examples/synthetic/data/*.[zrt]'
    path_greens = '../data/examples/synthetic/greens'
    path_weights= '../data/examples/synthetic/weights.dat'
    event_id    = 'syn_example'
    model       = 'socal3D'
    taup_model  = 'ak135'

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
        time_shift_max=0.,
        time_shift_groups=['ZR','T'],
        )


    #
    # User-supplied weights control how much each station contributes to the
    # objective function
    #

    station_id_list = parse_station_codes(path_weights)


    #
    # Next, we specify the moment tensor grid and source-time function
    #

    grid = DoubleCoupleGridRegular(
        npts_per_axis=20,
        magnitudes=[2.9, 3.0, 3.1])

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

    #
    # The main I/O work starts now
    #

    print('Reading data...\n')
    data = read(path_data, format='sac',
        event_id=event_id,
        station_id_list=station_id_list,
        tags = ['units:cm', 'type:displacement'])

    data.sort_by_distance()
    stations = data.get_stations()

    print('Processing data...\n')
    data_sw = data.map(process_sw)

    print('Reading Greens functions...\n')
    db = open_db(path_greens, format='SPECFEM3D_SGT', model=model)
    greens = db.get_greens_tensors(stations, origin)

    print('Processing Greens functions...\n')
    greens.convolve(wavelet)
    greens_sw = greens.map(process_sw)


    #
    # The main computational work starts now
    #

    print('Evaluating wave misfit...\n')
    results_sw = grid_search(data_sw, greens_sw, misfit_sw, origin, grid)

    results = results_sw

    # `grid` index corresponding to minimum misfit
    idx = results.source_idxmin()

    best_mt = grid.get(idx)
    lune_dict = grid.get_dict(idx)
    mt_dict = best_mt.as_dict()


    #
    # Generate figures and save results
    #

    print('Generating figures...\n')

    plot_data_greens1(event_id+'DC_waveforms.png',
                      data_sw,
                      greens_sw,
                      process_sw,
                      misfit_sw,
                      stations,
                      origin,
                      best_mt,
                      lune_dict)


    plot_beachball(event_id+'DC_beachball.png',
        best_mt, stations, origin)


    plot_misfit_dc(event_id+'DC_misfit.png', results)


    print('Saving results...\n')

    # collect information about best-fitting source
    merged_dict = merge_dicts(
        mt_dict,
        lune_dict,
        {'M0': best_mt.moment()},
        {'Mw': best_mt.magnitude()},
        origin,
        )

    # save best-fitting source
    save_json(event_id+'DC_solution.json', merged_dict)


    # save misfit surface
    results.save(event_id+'DC_misfit.nc')


    print('\nFinished\n')



# python GridSearch.SYN.DoubleCouple.py





