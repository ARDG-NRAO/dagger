"""
HTCondor backend implementation for Dagger.

This module provides the CondorDAG class that inherits from DAGGraph
and implements HTCondor-specific job submission and DAG management.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import htcondor2
from htcondor2 import dags

from .dag_graph import DAGGraph, Layer


class CondorDAG(DAGGraph):
    """
    HTCondor backend for DAG workflow execution.

    This class extends DAGGraph with HTCondor-specific functionality:
    - Creates HTCondor submit files
    - Manages HTCondor DAG structure
    - Handles job dependencies via HTCondor DAGMAN
    - Supports containerized execution via container_image

    :param dag_dir: Directory where DAG files and scripts will be stored
    :type dag_dir: str
    :param dag_name: Name of the DAG
    :type dag_name: str
    :param overwrite_dag_dir: Whether to clear existing files in dag_dir
    :type overwrite_dag_dir: bool

    :example:
    >>> from dagger import CondorDAG
    >>> dag = CondorDAG(dag_dir='my_workflow', dag_name='calibration')
    >>> def calibrate(input_ms: str, niter: int = 100):
    ...     import numpy as np
    ...     print(f"Calibrating {input_ms} with {niter} iterations")
    >>> dag.add_layer(
    ...     calibrate,
    ...     layer_name='cal1',
    ...     layer_vars=[{'input_ms': 'data1.ms', 'niter': 50}]
    ... )
    >>> dag.write_dag()
    """

    def __init__(
        self, dag_dir: str, dag_name: str, overwrite_dag_dir: bool = False
    ) -> None:
        """Initialize HTCondor DAG backend."""
        super().__init__()
        self.dag_dir = dag_dir
        self.dag_name = dag_name
        self.overwrite_dag_dir = overwrite_dag_dir

        # HTCondor-specific state
        self.htcondor_dag = htcondor2.dags.DAG()
        self._job_layers: Dict[str, dags.NodeLayer] = {}

        self._setup_directory()

    def _setup_directory(self) -> None:
        """Create DAG directory and optionally clear existing files."""
        if not os.path.exists(self.dag_dir):
            os.makedirs(self.dag_dir)

        if self.overwrite_dag_dir:
            # Clear existing files in the dag_dir
            for file in os.listdir(self.dag_dir):
                file_path = os.path.join(self.dag_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    def _create_submit_object(
        self, script_path: str, submit_vars: Dict
    ) -> htcondor2.Submit:
        """
        Create HTCondor Submit object for a Python script.

        :param script_path: Path to the Python script (relative to dag_dir)
        :type script_path: str
        :param submit_vars: Backend-agnostic submit variables
        :type submit_vars: Dict
        :return: HTCondor Submit object
        :rtype: htcondor2.Submit
        """
        # Start with basic submit dictionary
        submit_dict = {
            "executable": os.path.basename(script_path),
            "transfer_executable": "True",
            "output": "$(ClusterId).$(ProcId).out",
            "error": "$(ClusterId).$(ProcId).err",
            "log": "$(ClusterId).$(ProcId).log",
        }

        # Map backend-agnostic variables to HTCondor-specific ones
        if "memory" in submit_vars:
            submit_dict["request_memory"] = submit_vars["memory"]
        if "cpus" in submit_vars:
            submit_dict["request_cpus"] = str(submit_vars["cpus"])
        if "disk" in submit_vars:
            submit_dict["request_disk"] = submit_vars["disk"]

        # Container support - HTCondor handles this natively
        if "container_image" in submit_vars:
            submit_dict["container_image"] = submit_vars["container_image"]
            # HTCondor automatically detects image type (Docker, Singularity, OCI)

        # Pass through any other HTCondor-specific variables
        for key, value in submit_vars.items():
            if key not in [
                "memory",
                "cpus",
                "disk",
                "container_image",
            ]:
                submit_dict[key] = value

        return htcondor2.Submit(submit_dict)

    def add_layer(
        self,
        func: callable,
        layer_name: str,
        parent_layer_name: Optional[str] = None,
        layer_vars: Optional[List[Dict]] = None,
        submit_vars: Optional[Dict] = None,
        py_script_name: str = "",
        **kwargs,
    ) -> dags.NodeLayer:
        """
        Add a function as a layer to the HTCondor DAG.

        Uses parent class script generation methods, then creates HTCondor-specific
        submit objects and adds to the HTCondor DAG structure.

        :param func: Python function to convert to a DAG layer
        :type func: callable
        :param layer_name: Unique name for this layer
        :type layer_name: str
        :param parent_layer_name: Name of parent layer (for dependencies)
        :type parent_layer_name: Optional[str]
        :param layer_vars: List of variable dicts for multiple jobs
        :type layer_vars: Optional[List[Dict]]
        :param submit_vars: Backend-agnostic submit variables
        :type submit_vars: Optional[Dict]
        :param py_script_name: Custom script filename
        :type py_script_name: str
        :param kwargs: Additional HTCondor-specific arguments
        :return: HTCondor NodeLayer object
        :rtype: dags.NodeLayer

        :example:
        >>> dag.add_layer(
        ...     calibrate,
        ...     layer_name='calibrate',
        ...     layer_vars=[{'input_ms': 'data1.ms'}, {'input_ms': 'data2.ms'}],
        ...     submit_vars={'memory': '4GB', 'cpus': 2}
        ... )
        """
        if not callable(func):
            raise TypeError("func must be callable")

        # Validate parent exists if specified
        if parent_layer_name and parent_layer_name not in self.layers:
            raise ValueError(f"Parent layer '{parent_layer_name}' does not exist")

        # Set defaults
        layer_vars = layer_vars or [{}]
        submit_vars = submit_vars or {}

        # Generate script path
        if not py_script_name:
            py_script_name = f"{layer_name}.py"
        script_path = os.path.join(self.dag_dir, py_script_name)

        # Use parent class method to generate standalone script
        self.function_to_script(func, script_path, add_argparse=True)

        # Create HTCondor submit object
        submit_obj = self._create_submit_object(script_path, submit_vars)

        # Convert layer_vars to HTCondor vars format
        # Each dict becomes arguments passed to the script
        htcondor_vars = []
        for var_dict in layer_vars:
            args = " ".join([f"--{k} {v}" for k, v in var_dict.items()])
            htcondor_vars.append({"arguments": args})

        # Add to HTCondor DAG structure
        if parent_layer_name:
            # Child layer (depends on parent)
            parent_job = self._job_layers[parent_layer_name]
            job_layer = parent_job.child_layer(
                name=layer_name,
                submit_description=submit_obj,
                vars=htcondor_vars,
                **kwargs,
            )
        else:
            # Root layer (no dependencies)
            job_layer = self.htcondor_dag.layer(
                name=layer_name,
                submit_description=submit_obj,
                vars=htcondor_vars,
                **kwargs,
            )

        # Store for future parent references
        self._job_layers[layer_name] = job_layer

        # Update abstract DAGGraph representation
        layer = Layer(
            name=layer_name,
            executable=py_script_name,
            job_vars=layer_vars,
            inputs=[],
            outputs=[],
            submit_vars=submit_vars,
        )
        parent_layers = [parent_layer_name] if parent_layer_name else None
        super().add_layer(layer, parent_layers)

        return job_layer

    def write_dag(self, **kwargs) -> None:
        """
        Write HTCondor DAG file and submit files to disk.

        Creates a .dag file that can be submitted to HTCondor via condor_submit_dag.

        :param kwargs: Additional arguments for htcondor2.dags.write_dag
        :type kwargs: dict

        :example:
        >>> dag.write_dag()
        >>> # Then submit: condor_submit_dag my_workflow/calibration.dag
        """
        dag_file_path = os.path.join(self.dag_dir, f"{self.dag_name}.dag")

        dags.write_dag(
            self.htcondor_dag,
            dag_dir=self.dag_dir,
            dag_file_name=dag_file_path,
            **kwargs,
        )

        print(f"HTCondor DAG written to: {dag_file_path}")
        print(f"Submit with: condor_submit_dag {dag_file_path}")
