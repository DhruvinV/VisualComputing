# CSC320 Fall 2019
# Assignment 3
# (c) Olga (Ge Ya) Xu, Kyros Kutulakos
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY PROHIBITED. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

#
# DO NOT MODIFY ANY PART OF THIS FILE
#

# import basic packages
import sys
import argparse
from patchMatch import *

# Routine for parsing command-line arguments
# Upon success, it returns any unprocessed/unrecognized arguments
# back to the caller
#
# Input arguments
#   - argv:   input command line arguments (as returned by sys.argv[1:])
#   - prog:   name of the executable (as returned by sys.argv[0])
# Return values
#   - True if all the required arguments are specified and False otherwise
#   - a Namespace() object containing the user-specified arguments and
#     any optional arguments that take default values
#   - a string containing any unrecognized/unprocessed command-line
#     arguments
#   - an error message (if success=False)
#
'''This function parses and return arguments passed in'''


def parse_arguments(argv, prog=''):
    # Initialize the command-line parser
    parser = argparse.ArgumentParser(prog,
                                     description='Script for patch match.')

    #
    # Main input/output arguments
    #

    parser.add_argument('--source',
                        type=str,
                        help='Path to source image',
                        required=True)
    parser.add_argument('--target',
                        type=str,
                        help='Path to targe image',
                        required=True)
    parser.add_argument('--output',
                        type=str,
                        help='Path to save the results',
                        default='output',
                        required=False)
    #
    # Basic parameters of the PatchMatch algorithm
    #

    # Number of PatchMatch iteractions
    parser.add_argument('--iters',
                        type=int,
                        help='Number of iterations',
                        default=4,
                        required=False)
    # Patch size
    parser.add_argument('--patch-size',
                        type=int,
                        help='Patch window size',
                        default=7,
                        required=False)
    # Ratio between search window sizes. This is the alpha parameter in Eq.(1)
    parser.add_argument('--alpha',
                        type=float,
                        help='Ratio between random search window sizes',
                        default=0.5,
                        required=False)
    # Maximum radius for random search, expressed in pixels. This is the
    # the w parameter in Eq.(1). If the parameter is omitted, the search
    # radius will be unrestricted
    parser.add_argument('--w',
                        type=int,
                        default=0,
                        help='Maximum radius for random search (in pixels)',
                        required=False)
    # Rather than generating a new random nearest-neighbor field (NNF) each time PatchMatch
    # is run, you can provide the NNF as an additional input to PatchMatch.
    # This makes it possible to enforce a predictable behavior.
    # The starter code provides initial NNFs for you to use for testing purposes.
    parser.add_argument('--init-nnf',
                        type=str,
                        help='Path to initial nearest-neighbour field',
                        required=False,
                        default=None)

    # The remaining arguments are for debugging/visualization purposes

    # The two main arguments for controlling program execution are
    #    --disable-random for disabling random search during PatchMatch
    #    --disable-propagation for disabling propagation during PatchMatch
    #
    # Execution of the full PatchMatch algorithm requires both to be enabled.
    # Use these flags for debugging purposes, to make sure that your implementations
    # of random search and/or propagation are correct
    parser.add_argument('--disable-random',
                        action='store_true',
                        help='Disable random search',
                        required=False)
    parser.add_argument('--disable-propagation',
                        action='store_true',
                        help='Disable propagation',
                        required=False)
    # Output a visualization of the nearest-neighbor field as an image
    parser.add_argument('--nnf-image',
                        action='store_true',
                        help='Output the NNF as a color image',
                        required=False)
    # Output the partial results after each iteration
    parser.add_argument('--partial-results',
                        action='store_true',
                        help='Output partial results after each iteration',
                        required=False)
    # Output a visualization of the nearest-neighbor field as an image composite
    # that includes vectors between corresponding patches in the source and target
    parser.add_argument('--nnf-vectors',
                        action='store_true',
                        help='Output the NNF as a correspondence field between images',
                        required=False)
    # Reconstruct the source image using the NNF patches from the target image
    parser.add_argument('--rec-source',
                        action='store_true',
                        help='Reconstruct the source using patches from the target',
                        required=False)
    # Arguments related to NNF visualization
    parser.add_argument('--nnf-subsampling',
                        type=int,
                        help='Subsampling of the correspondence field (in pixels)',
                        default=100,
                        required=False)
    parser.add_argument('--nnf-line-width',
                        type=float,
                        help='Line width for drawing the correspondence field ',
                        default=0.5,
                        required=False)
    parser.add_argument('--nnf-line-color',
                        type=str,
                        help='Line color for drawing the correspondence field ',
                        default='r',
                        required=False)
    parser.add_argument('--server',
                        action='store_true',
                        help='Remote-server execution of matplotlib commands',
                        required=False)
    parser.add_argument('--tmpdir',
                        type=str,
                        help='Directory for storing temporary images',
                        default='./',
                        required=False)

    # Run the python argument-parsing routine, leaving any
    #  unrecognized arguments intact
    args, unprocessed_argv = parser.parse_known_args(argv)

    success = True
    msg = ''

    if args.disable_random and args.disable_propagation:
        msg = 'Error you can disable random search or propagation but not both'
        success = False

    # return any arguments that were not recognized by the parser
    return success, args, unprocessed_argv, msg  #


# Top-level routine for PatchMatch when running the code from the command line
#
# Input arguments
#   - argv:   list of command-line arguments (as returned by sys.argv[1:])
#   - prog:   name of the executable (as returned by sys.argv[0])
# Return value
#   - list that holds any unrecognized command-line arguments
#     (in the same format as sys.argv[1:])

def main(argv, prog=''):
    # Create an instance of the patch match class
    patmat = PatchMatch()

    # Parse the command line arguments
    success, args, unprocessed_argv, msg = parse_arguments(argv, prog)

    if not success:
        print 'Error: Argument Parsing: %s' % msg
        return success, unprocessed_argv

    # process input arguments
    ok, msg = patmat.read_image(args.source, 'source')
    if not ok:
        print 'Error: read_image: ',msg
        exit(1)
    ok, msg = patmat.read_image(args.target, 'target')
    if not ok:
        print 'Error: read_image: ',msg
        exit(1)

    # process output arguments
    patmat.set_output(args.output)
    patmat.set_NNF_image(args.nnf_image)
    patmat.set_NNF_vectors(args.nnf_vectors)
    patmat.set_rec_source(args.rec_source)

    # process algorithm parameters
    patmat.set_init_NNF(nnf_file=args.init_nnf)
    patmat.set_iterations(args.iters)
    patmat.set_patch_size(args.patch_size)
    patmat.set_alpha(args.alpha)
    patmat.set_w(args.w)
    # propagation and random search are enabled by default
    # so we toggle them to disable
    patmat.set_propagation(not args.disable_propagation)
    patmat.set_random(not args.disable_random)

    # process visualization parameters
    patmat.set_partial_results(args.partial_results)
    patmat.set_NNF_subsampling(args.nnf_subsampling)
    patmat.set_NNF_line_width(args.nnf_line_width)
    patmat.set_NNF_line_color(args.nnf_line_color)
    patmat.set_server(args.server)
    patmat.set_tmpdir(args.tmpdir)

    # run the PatchMatch algorithm
    patmat.run_iterations()

    print 'Done.'


# Include these lines so we can run the script from the command line
if __name__ == '__main__':
    main(sys.argv[1:], sys.argv[0])
