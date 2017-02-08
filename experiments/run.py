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
    except Exception:
        print 'Encoding scheme not found. Try BOCKWURST!'
        sys.exit()

    try:
        com_alg = getattr(C, com_name)
    except Exception:
        print 'Encoding scheme not found. Try LZ78!'
        sys.exit()

    data = C.Util.load_data(dataset, slices, data_loc)

    results = C.Util.run_experiments(
        data=data,
        N=N,
        com_alg=[com_alg],
        enc_alg=[enc_alg],
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

    with open(os.path.join(res, filename), 'wb') as f:
        pickle.dump(results, f)


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
        '--bockwurst',
        '-b',
        dest='bockwurst',
        action='store_true',
        help='show me some bockwurst (default: False)'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        dest='verbose',
        action='store_true',
        help='print progress (default: False)'
    )

    args = parser.parse_args()

    if (args.bockwurst):
        print(
            "\n"
            "                 `:+syhdmmmmmmdhys+:`          \n"
            "             `/sdmmmmmmmmmmmmmmmmmmmmds/`      \n"
            "          `/ymmmmdy+:.:mmmmmmmmmmmmmmmmmmy/`   \n"
            "        `+dmmmh+- `/oydmmmmmmmmmmmmmmmmmmmmd+` \n"
            "       /dmmms- -odmmmmmmmmmmmmmmmmmmmmmmmmmmmh`\n"
            "     `ymmmy. /hmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm+\n"
            "    .dmmm/ -hmmmmmmmmmyo/::::/oymmmmmmmmmmmmmm/\n"
            "   .dmmm- /mmmmmmmmo-            -ommmmmmmmmmh`\n"
            "   hmmm- +mmmmmmmo`                `/ydmmmhdds:\n"
            "  /mmmo :mmmmmmm:                          :`  \n"
            "  ymmm` hmmmmmm:                               \n"
            "  mmmh`-mmmmmmh                                \n"
            "  mmmmmmmmmmmmy                                \n"
            "  mmmmmmmmmmmmh                                \n"
            "  ymmmmmmmmmmmm:                               \n"
            "  /mmmmmmmmmmmmm:                              \n"
            "   hmmmmmmmmmmmmmo`                            \n"
            "   `dmmmmmmmmmmmmmmo-                          \n"
            "    .dmmmmmmmmmmmmmmmmyo/::::/oo+/-            \n"
            "     `ymmmmmmmmmmmmmmmmmmmmmmmmmmmmdoy         \n"
            "       /dmmmmmmmmmmmmmmmmmmmmmmmmmmmmmo        \n"
            "        `+dmmmmmmmmmmmmmmmmmmmmmmmmmmd         \n"
            "          `/ymmmmmmmmmmmmmmmmmmmmmmmmo         \n"
            "             `/sdmmmmmmmmmmmmmmmmmmmo          \n"
            "                 .:+syhdmmmmmmdhys/`           \n"
        )

    run_experiments(
        args.encoding,
        args.compression,
        args.dataset,
        args.runs,
        args.dir,
        args.slices,
        args.verbose
    )
