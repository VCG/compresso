#!/usr/bin/env python

import argparse
import os
import sys
import cPickle as pickle

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import compression as C


def render_plots(pickle_file, output):
    with open(pickle_file, 'rb') as f:
        results = pickle.load(f)

    C.Util.plot(
        method_labels=results['methods'],
        data_bytes=results['comp_bytes'],
        ratios=results['ratios'],
        com_speed=results['total_comp_speed'],
        com_speed_stderr=results['total_comp_speed_std'],
        dcom_speed=results['total_decomp_speed'],
        dcom_speed_stderr=results['total_decomp_speed_std'],
        save=output,
        dpi=300,
        bw=False
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'results',
        metavar='PATH',
        type=str,
        help='path to pickled results'
    )

    parser.add_argument(
        '--output',
        '-o',
        metavar='PATH',
        dest='output',
        action='store',
        type=str,
        default='figures',
        help='output (default: figures/<results>.eps)'
    )

    args = parser.parse_args()

    if not os.path.isfile(args.results):
        print('Results file not found')
        sys.exit()

    output = os.path.basename(args.results)

    if args.output:
        output = os.path.join(args.output, output)

    render_plots(args.results, output)
