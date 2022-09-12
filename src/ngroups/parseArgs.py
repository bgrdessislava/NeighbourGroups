#!/usr/bin/env python3

""" Neighbour Groups - ML(ST) Classification """

import sys
import logging
import argparse
from timeit import default_timer as timer
from .main import splitTestTrain, trainNG, testNG, runNG, downloadExample
from ._version import __version__


def parseArgs() -> argparse.Namespace:
    epilog = (
        'Dessislava Veltcheva, University of Oxford '
        '(bgrdessislava@gmail.com)'
    )
    baseParser = getBaseParser(__version__)
    parser = argparse.ArgumentParser(
        epilog=epilog, description=__doc__, parents=[baseParser])
    subparser = parser.add_subparsers(
        title='required commands',
        description='',
        dest='command',
        metavar='Commands',
        help='Description:')

    sp1 = subparser.add_parser(
        'prepare',
        description=splitTestTrain.__doc__,
        help='Split isolates into test and training set.',
        parents=[baseParser],
        epilog=parser.epilog)
    sp1.add_argument(
        'data', help='Path to data file in .csv format')
    sp1.add_argument(
        'prefix', help='File prefix to read/write data.')
    sp1.add_argument(
        '--trainSize', type=float, default=0.8,
        help='Proportion of data to use as training (default: %(default)s)')
    sp1.set_defaults(function=splitTestTrain)

    sp2 = subparser.add_parser(
        'train',
        description=trainNG.__doc__,
        help='Train the CatBoost classifer.',
        parents=[baseParser],
        epilog=parser.epilog)
    sp2.add_argument(
        'newick', help='Path to newick file of the training dataset.')
    sp2.add_argument(
        'prefix', help='File prefix to read/write data.')
    sp2.add_argument(
        '--nGroup', type=int, default=20,
        help='Number of Neighbour Groups to define '
             'from tree (default: %(default)s)')
    sp2.set_defaults(function=trainNG)

    sp3 = subparser.add_parser(
        'test',
        description=testNG.__doc__,
        help='Test the CatBoost classifer.',
        parents=[baseParser],
        epilog=parser.epilog)
    sp3.add_argument(
        'newick', help='Path to newick file of the full dataset.')
    sp3.add_argument(
        'prefix', help='File prefix to read/write data.')
    sp3.set_defaults(function=testNG)

    sp4 = subparser.add_parser(
        'predict',
        description=runNG.__doc__,
        help='Classify isolates using the trained model.',
        parents=[baseParser],
        epilog=parser.epilog)
    sp4.add_argument(
        'data', help='Path to data file in .csv format')
    sp4.add_argument(
        'model', help='Path to trained NeighbourGroup model.')
    sp4.set_defaults(function=runNG)

    sp5 = subparser.add_parser(
        'getExample',
        description=downloadExample.__doc__,
        help='Download example data.',
        parents=[baseParser],
        epilog=parser.epilog)
    sp4.add_argument(
        'path', default='.'
        help='Directory to save example data.')
    sp5.set_defaults(function=downloadExample)

    sp6 = subparser.add_parser(
        'getModel',
        description=downloadExample.__doc__,
        help='Download pre-trained model from publication.',
        parents=[baseParser],
        epilog=parser.epilog)
    sp6.add_argument(
        'path', default='.'
        help='Directory to save example data.')
    sp6.set_defaults(function=downloadModel)

    args = parser.parse_args()
    if 'function' not in args:
        parser.print_help()
        sys.exit()

    rc = executeCommand(args)
    return rc


def executeCommand(args):
    # Initialise logging
    logFormat = '%(asctime)s - %(levelname)s - %(funcName)s - %(message)s'
    logging.basicConfig(level=args.verbose, format=logFormat)
    del args.verbose, args.command
    # Pop main function and excute script
    function = args.__dict__.pop('function')
    start = timer()
    rc = function(**vars(args))
    end = timer()
    logging.info(f'Total execution time: {end - start:.3f} seconds.')
    logging.shutdown()
    return rc


def getBaseParser(version: str) -> argparse.Namespace:
    """ Create base parser of verbose/version. """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--version', action='version', version='%(prog)s {}'.format(version))
    parser.add_argument(
        '--verbose', action='store_const', const=logging.DEBUG,
        default=logging.ERROR, help='verbose logging for debugging')
    return parser
