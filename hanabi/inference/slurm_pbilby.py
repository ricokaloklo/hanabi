from os.path import abspath
from .parser import create_joint_analysis_pbilby_parser
from parallel_bilby.utils import get_cli_args
import parallel_bilby
from parallel_bilby.slurm import (
    setup_submit,
    BaseNode,
    MergeNodes,
)
from .._version import __version__


def setup_submit(data_dump_files, inputs, args):

    # Create analysis nodes
    analysis_nodes = []
    for idx in range(args.n_parallel):
        node = AnalysisNode(data_dump_files, inputs, idx, args)
        node.write()
        analysis_nodes.append(node)

    if len(analysis_nodes) > 1:
        final_analysis_node = MergeNodes(analysis_nodes, inputs, args)
        final_analysis_node.write()
    else:
        final_analysis_node = analysis_nodes[0]

    bash_script = "{}/bash_{}.sh".format(inputs.submit_directory, inputs.label)
    with open(bash_script, "w+") as ff:
        dependent_job_ids = []
        for ii, node in enumerate(analysis_nodes):
            print("jid{}=$(sbatch {})".format(ii, node.filename), file=ff)
            dependent_job_ids.append("${{jid{}##* }}".format(ii))
        if len(analysis_nodes) > 1:
            print(
                "sbatch --dependency=afterok:{} {}".format(
                    ":".join(dependent_job_ids), final_analysis_node.filename
                ),
                file=ff,
            )
        print('squeue -u $USER -o "%u %.10j %.8A %.4C %.40E %R"', file=ff)

    return bash_script



class AnalysisNode(parallel_bilby.slurm.AnalysisNode):
    def __init__(self, data_dump_files, inputs, idx, args):
        self.data_dump_files = data_dump_files
        self.inputs = inputs
        self.args = args
        self.idx = idx
        self.filename = "{}/analysis_{}_{}.sh".format(
            self.inputs.submit_directory, self.inputs.label, self.idx
        )

        self.job_name = "{}_{}".format(self.idx, self.inputs.label)
        self.nodes = self.args.nodes
        self.ntasks_per_node = self.args.ntasks_per_node
        self.time = self.args.time
        self.mem_per_cpu = self.args.mem_per_cpu
        self.logs = self.inputs.data_analysis_log_directory

        # Pass the complete config ini
        self.complete_ini_file = inputs.complete_ini_file

    @property
    def executable(self):
        if self.args.sampler == "dynesty":
            return "hanabi_joint_analysis_pbilby"
        else:
            raise ValueError(
                "Unable to determine sampler to use from {}".format(self.args.sampler)
            )

    def get_run_string(self):
        run_list = [self.inputs.complete_ini_file]
        run_list += ["--data-dump-files {}".format(data_dump_file) for data_dump_file in self.data_dump_files]
        run_list.append("--label {}".format(self.label))
        run_list.append("--outdir {}".format(abspath(self.inputs.result_directory)))
        run_list.append(
            "--sampling-seed {}".format(self.inputs.sampling_seed + self.idx)
        )
        # Override bad --mpi-timing-interval
        if self.inputs.mpi_timing_interval == False:
            run_list.append(
                "--mpi-timing-interval {}".format(0)
            )

        return " ".join(run_list)
