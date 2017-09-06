#!/usr/bin/env python

import argparse
import os
import sys
import cPickle as pickle

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import compression as C


def run_experiments(
    enc_name, com_name, dataset, N, data_loc=None, slices=-1, verbose=False
):
    try:
        enc_alg = getattr(C, enc_name)

        # This is a stupid test to ensure that the correct encoder has been
        # found, which is needed since we have uppercase module imports and
        # lowercase filenames
        enc_alg.name()
    except Exception:
        print 'Encoding scheme not found!'
        sys.exit()

    try:
        com_alg = getattr(C, com_name)

        # This is a stupid test to ensure that the correct compressor has been
        # found, which is needed since we have uppercase module imports and
        # lowercase filenames
        com_alg.name()
    except Exception:
        print 'Encoding scheme not found!'
        sys.exit()

    data = C.Util.load_data(dataset, slices, data_loc)

    results = C.Util.run_experiment(
        com=com_alg,
        enc=enc_alg,
        data=data,
        N=N,
        verbose=verbose
    )

    filename = '_'.join([enc_name, com_name, dataset, str(N), str(slices)])
    keepcharacters = ('-', '.', '_')

    filename = ''.join(
        [c for c in filename if c.isalnum() or c in keepcharacters]
    ).rstrip()

    res = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')

    if not os.path.exists(res):
        os.makedirs(res)

    print(results)

    with open(os.path.join(res, filename), 'w') as f:
        for result in results:
            f.write('{}: {}\n'.format(result, results[result]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'encoding',
        type=str,
        help='name of encoding scheme'
    )

    parser.add_argument(
        'compression',
        type=str,
        help='name of compression scheme'
    )

    parser.add_argument(
        'dataset',
        type=str,
        help='name of data set'
    )

    parser.add_argument(
        '--directory',
        '-d',
        dest='dir',
        metavar='PATH',
        action='store',
        type=str,
        default=None,
        help='path to data directory'
    )

    parser.add_argument(
        '--runs',
        '-r',
        dest='runs',
        metavar='NUM',
        action='store',
        type=int,
        default=1,
        help='number of runs (default: 1)'
    )

    parser.add_argument(
        '--slices',
        '-s',
        dest='slices',
        metavar='NUM',
        action='store',
        type=int,
        default=-1,
        help='number of slices per dataset (default: -1 (all))'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        dest='verbose',
        action='store_true',
        help='print progress (default: False)'
    )

    args = parser.parse_args()

    run_experiments(
        args.encoding,
        args.compression,
        args.dataset,
        args.runs,
        args.dir,
        args.slices,
        args.verbose
    )
