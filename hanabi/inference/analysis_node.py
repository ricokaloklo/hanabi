import os

from bilby_pipe.job_creation.node import Node
from .fake_generation_node import FakeGenerationNode
from bilby_pipe.job_creation.bilby_pipe_dag_creator import get_trigger_time_list


class JointAnalysisNode(Node):
    def __init__(self, joint_main_input, single_trigger_pe_inputs, parallel_idx, dag, analysis_prog_name="hanabi_joint_analysis"):
        self.joint_main_input = joint_main_input
        self.single_trigger_pe_inputs = single_trigger_pe_inputs
        super().__init__(self.joint_main_input)

        self.dag = dag
        self.parallel_idx = parallel_idx
        self.request_cpus = joint_main_input.request_cpus
        self.n_triggers = joint_main_input.n_triggers

        self.analysis_prog_name = analysis_prog_name

        self.base_job_name = "{}_{}".format(self.joint_main_input.label, self.analysis_prog_name)
        if parallel_idx != "":
            self.job_name = f"{self.base_job_name}_{parallel_idx}"
        else:
            self.job_name = self.base_job_name
        self.label = self.job_name

        self.setup_arguments()

        # Loop over triggers
        for single_trigger_pe_input in self.single_trigger_pe_inputs:
            # Initialize FakeGenerationNode (and actually generate data if --local-generation is set)
            generation_node_list = []
            # FIXME Turn off local generation?
            single_trigger_pe_input.local_generation = False

            trigger_times = get_trigger_time_list(single_trigger_pe_input)
            for idx, trigger_time in enumerate(trigger_times):
                # FakeGenerationNode does not need to know about dag
                kwargs = dict(trigger_time=trigger_time, idx=idx, dag=None)
                generation_node = FakeGenerationNode(single_trigger_pe_input, **kwargs)
                generation_node_list.append(generation_node)

            # FIXME There should really be ONLY ONE generation node per trigger!
            assert len(generation_node_list) == 1, "Currently only support the case with one data dump file per trigger"
            for generation_node in generation_node_list:
                if self.joint_main_input.transfer_files or self.joint_main_input.osg:
                    data_dump_file = generation_node.data_dump_file
                    input_files_to_transfer = [
                        str(data_dump_file),
                        str(single_trigger_pe_input.complete_ini_file),
                        str(self.joint_main_input.complete_ini_file)
                    ]
                    self.extra_lines.extend(
                        self._condor_file_transfer_lines(
                            input_files_to_transfer,
                            [self._relative_topdir(single_trigger_pe_input.outdir, self.joint_main_input.initialdir)],
                        )
                    )
                    self.arguments.add("outdir", os.path.basename(self.joint_main_input.outdir))

                # Add path to data dump file for this trigger
                self.arguments.add("data-dump-files", str(generation_node.data_dump_file))
            
            # Add path to the complete ini file for this trigger
            self.arguments.add("trigger-ini-files", str(single_trigger_pe_input.complete_ini_file))

        self.arguments.add("label", str(self.label))
        self.extra_lines.extend(self._checkpoint_submit_lines())
        self.process_node()

    @property
    def executable(self):
        return self._get_executable_path(self.analysis_prog_name)

    @property
    def request_memory(self):
        return self.joint_main_input.request_memory

    @property
    def log_directory(self):
        return self.joint_main_input.data_analysis_log_directory

    @property
    def result_file(self):
        return f"{self.joint_main_input.result_directory}/{self.job_name}_result.json"

    @property
    def slurm_walltime(self):
        """ Default wall-time for base-name """
        # Seven days
        return self.joint_main_input.scheduler_analysis_time

class JointAnalysisNodeUsingParallelBilby(JointAnalysisNode):
    def __init__(self, joint_main_input, single_trigger_pe_inputs, parallel_idx, dag, analysis_prog_name="hanabi_joint_analysis_pbilby"):
        super(JointAnalysisNodeUsingParallelBilby, self).__init__(
            joint_main_input,
            single_trigger_pe_inputs,
            parallel_idx,
            dag,
            analysis_prog_name
        )