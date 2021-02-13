import bilby_pipe
from ..inference.parser import _create_base_parser
from bilby_pipe.utils import nonefloat, noneint, nonestr

def create_hierarchical_analysis_parser(prog, prog_version):
    base_parser = _create_base_parser(prog, prog_version)

    parser = bilby_pipe.bilbyargparser.BilbyArgParser(  
        prog=prog,
        ignore_unknown_config_file_keys=False,
        allow_abbrev=False,
        add_help=False,
        parents=[base_parser]       
    )

    parser.add("ini", type=str, is_config_file=True, help="Configuration ini file")

    input_parser = parser.add_argument_group(title="Data input", description="")
    input_parser.add_argument(
        "--n-triggers",
        type=int,
        help="Number of triggers analyzed jointly"
    )
    input_parser.add_argument(
        "--inference-result",
        type=str,
        help="The path to the hanabi.inference output result file"
    )

    sampler_input_parser = parser.add_argument_group(title="Sampling", description="")
    sampler_input_parser.add_argument(
        "--sampling-seed",
        type=int,
        help="The random seed for sampling"
    )
    sampler_input_parser.add_argument(
        "--request-cpus",
        type=int,
        help="Use multiple cores for calculation"
    )

    src_pop_model_parser = parser.add_argument_group(title="Source population model", description="")

    lensing_model_parser = parser.add_argument_group(title="Lensing model", description="")
    lensing_model_parser.add_argument(
        "--optical-depth",
        type=str,
        help=""
    )
    lensing_model_parser.add_argument(
        "--absolute-magnifications",
        action="append",
        help=(
            "A list of prior for the absolute magnifications. "
            "The number of priors must match with --n-triggers"
        )
    )
    lensing_model_parser.add_argument(
        "--redshift-prior",
        type=str,
        help=(
            "Use this as the prior for redshift instead of "
            "the one inferred from the optical depth and the merger rate density"
        )
    )

    return parser