"""
Module for abstract DAG representation.

This module provides classes for representing DAGs in a backend-agnostic way,
enabling portability between execution backends (HTCondor, SLURM) and
interoperability with other workflow systems (Dask, etc.).
"""

import json
from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set


@dataclass
class Layer:
    """
    Represents a computational layer (node) in the DAG.
    A layer can contain multiple jobs with different parameters.

    :param name: Unique name for this layer
    :type name: str
    :param executable: Name of the function or path to executable
    :type executable: str
    :param job_vars: List of variable dictionaries, one per job in this layer
    :type job_vars: List[Dict[str, Any]]
    :param inputs: Input file paths or data references
    :type inputs: List[str]
    :param outputs: Output file paths or data references
    :type outputs: List[str]
    :param submit_vars: Backend-agnostic submit variables
    :type submit_vars: Dict[str, Any]
    :param metadata: Additional metadata
    :type metadata: Dict[str, Any]

    :example:
    >>> layer = Layer(
    name="calibration",
    executable="calibrate_data.py",
    job_vars=[{"input_file": "raw1.ms"}, {"input_file": "raw2.ms"}],
    inputs=["/data/raw1.ms"],
    outputs=["/data/calibrated1.ms"],
    submit_vars={"memory": "4GB", "cpus": 2, "container_image": "/path/to/container.sif"},
    metadata={"description": "Calibration layer"}
    )
    >>> print(layer)
    Layer(name='calibration', executable='calibrate_data.py',
        job_vars=[{'input_file': 'raw1.ms'}, {'input_file': 'raw2.ms'}],
        inputs="/data/raw1.ms",
        outputs="/data/calibrated1.ms",
        submit_vars={'memory': '4GB', 'cpus': 2}, container_image='/path/to/container.sif',
        metadata={'description': 'Calibration layer'})

    :note: Layer instances are typically created and managed by the DAGGraph when building the workflow.
    """

    name: str
    executable: str

    # Job variables - each dict represents one job in this layer
    job_vars: List[Dict[str, Any]] = field(default_factory=list)

    # Data products
    inputs: List = field(default_factory=list)
    outputs: List = field(default_factory=list)

    # Backend-agnostic submit variables
    submit_vars: Dict[str, Any] = field(default_factory=dict)

    # Optional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """
        Serialize to dictionary.

        :return: Dictionary representation
        :rtype: dict
        """
        return {
            "name": self.name,
            "executable": self.executable,
            "job_vars": self.job_vars,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "submit_vars": self.submit_vars,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Layer":
        """
        Deserialize from dictionary.

        :param data: Dictionary representation
        :type data: dict
        :return: Layer instance
        :rtype: Layer
        """
        return cls(
            name=data["name"],
            executable=data["executable"],
            job_vars=data.get("job_vars", []),
            inputs=data["inputs"],
            outputs=data["outputs"],
            submit_vars=data.get("submit_vars", {}),
            metadata=data.get("metadata", {}),
        )


class DAGGraph:
    """
    Abstract DAG representation inspired by Dask's HighLevelGraph.
    Stores layers (nodes) and dependencies (edges) separately.

    Structure:
    - layers: Dict[str, Layer] - the computational nodes
    - dependencies: Dict[str, Set[str]] - parent relationships

    :example:
    >>> dag_graph = DAGGraph()
    >>> layer1 = Layer(name="calibration", executable="calibrate")
    >>> layer2 = Layer(name="imaging", executable="image")
    >>> dag_graph.add_layer(layer1)
    >>> dag_graph.add_layer(layer2, parent_layers=["calibration"])
    >>> print(dag_graph.topological_order())
    ['calibration', 'imaging']
    """

    def __init__(self):
        """Initialize an empty DAG graph."""
        self.layers: Dict[str, Layer] = {}
        self.dependencies: Dict[str, Set[str]] = {}

    def parse_function(
        self,
        func: callable,
        return_as_string: bool = True,
        return_name: bool = False,
        trim_whitespace: bool = True,
    ) -> tuple:
        """
        Parse a Python function and extract its source code.

        This method uses the Python inspect module to retrieve the source code.
        The function can be returned as a list of strings or formatted into
        a valid string that can be directly placed into a Python script.

        :param func: The function to parse.
        :type func: callable
        :param return_as_string: If True, return the function as a formatted string.
                                 If False, return as a list of strings.
        :type return_as_string: bool
        :param return_name: If True, return the function name as well.
        :type return_name: bool
        :param trim_whitespace: If True, trim leading whitespace from the function source code.
        :type trim_whitespace: bool
        :return: The function source code as a string or list of strings, optionally with name
        :rtype: Union[str, List[str], Tuple[str, str], Tuple[List[str], str]]
        :raises TypeError: If the input is not a callable function.

        :example:
        >>> def example_function(x: int, y: str) -> None:
        ...     print(f"Example: x={x}, y={y}")
        >>> dag = DAGGraph()
        >>> code = dag.parse_function(example_function)
        """
        if not callable(func):
            raise TypeError("Input must be a callable function.")

        import inspect

        name = func.__name__
        funcstr = inspect.getsource(func)
        funcstr = funcstr.split("\n")[1:]  # Remove function signature line

        if trim_whitespace:
            # Trim leading whitespace from the function source code
            # Determine the global indentation level from the first line
            if funcstr:
                indent_level = len(funcstr[0]) - len(funcstr[0].lstrip())
                funcstr = [line[indent_level:] for line in funcstr]

        if return_as_string:
            funcstr = "\n".join(funcstr)

        if return_name:
            return funcstr, name
        else:
            return funcstr

    def generate_argparse_cli(self, func: callable) -> str:
        """
        Generate argparse CLI code based on function signature and type hints.

        Uses inspect.signature() to extract parameter information and creates
        argparse ArgumentParser code with proper type conversions.

        Type mapping:
        - int/float/str -> type=int/float/str
        - bool -> action='store_true'
        - Optional[T] -> required=False
        - List[T] -> nargs='+'
        - No annotation -> type=str (default)

        :param func: The function to generate argparse CLI for
        :type func: callable
        :return: String containing argparse setup code
        :rtype: str
        :raises TypeError: If the input is not a callable function

        :example:
        >>> def process(input_file: str, threshold: int = 10, verbose: bool = False):
        ...     pass
        >>> dag = DAGGraph()
        >>> cli_code = dag.generate_argparse_cli(process)
        """
        if not callable(func):
            raise TypeError("Input must be a callable function.")

        import inspect
        from typing import get_origin, get_args

        sig = inspect.signature(func)
        lines = ["import argparse", "", "parser = argparse.ArgumentParser()"]

        for param_name, param in sig.parameters.items():
            annotation = param.annotation
            default = param.default

            # Build argument configuration
            arg_config = []

            # Handle type annotations
            if annotation == inspect.Parameter.empty:
                # No type hint - default to str
                arg_config.append("type=str")
            elif annotation == bool or annotation == "bool":
                # Boolean parameters use store_true
                arg_config.append("action='store_true'")
            elif annotation == int or annotation == "int":
                arg_config.append("type=int")
            elif annotation == float or annotation == "float":
                arg_config.append("type=float")
            elif annotation == str or annotation == "str":
                arg_config.append("type=str")
            else:
                # Handle generic types (Optional, List, etc.)
                origin = get_origin(annotation)
                if origin is list:
                    arg_config.append("nargs='+'")
                    args = get_args(annotation)
                    if args:
                        inner_type = args[0]
                        if inner_type in (int, float, str):
                            arg_config.append(f"type={inner_type.__name__}")
                        else:
                            arg_config.append("type=str")
                    else:
                        arg_config.append("type=str")
                else:
                    # Default to str for complex types
                    arg_config.append("type=str")

            # Handle default values
            if default != inspect.Parameter.empty:
                if annotation != bool and annotation != "bool":
                    if isinstance(default, str):
                        arg_config.append(f"default='{default}'")
                    else:
                        arg_config.append(f"default={default}")
            else:
                # No default means required (unless it's a boolean flag)
                if annotation != bool and annotation != "bool":
                    arg_config.append("required=True")

            # Build the parser.add_argument line
            arg_line = f"parser.add_argument('--{param_name}', {', '.join(arg_config)})"
            lines.append(arg_line)

        lines.append("args = parser.parse_args()")
        return "\n".join(lines)

    def function_to_script(
        self, func: callable, script_path: str, add_argparse: bool = True
    ) -> str:
        """
        Convert a Python function to a complete standalone script with argparse CLI.

        Creates a complete Python script file with:
        - Shebang line
        - Argparse CLI (if add_argparse=True)
        - Function definition
        - Main block that calls the function with parsed arguments

        :param func: The function to convert to a script
        :type func: callable
        :param script_path: Path where the script will be saved
        :type script_path: str
        :param add_argparse: Whether to add argparse CLI code
        :type add_argparse: bool
        :return: Path to the created script
        :rtype: str
        :raises TypeError: If the input is not a callable function

        :example:
        >>> def calibrate(input_ms: str, output_ms: str, niter: int = 100):
        ...     import numpy as np
        ...     print(f"Calibrating {input_ms} -> {output_ms} with {niter} iterations")
        >>> dag = DAGGraph()
        >>> script_path = dag.function_to_script(calibrate, "/tmp/calibrate.py")
        """
        if not callable(func):
            raise TypeError("Input must be a callable function.")

        import inspect
        from pathlib import Path

        # Get function source and name
        func_body, func_name = self.parse_function(
            func, return_as_string=True, return_name=True
        )

        # Get function signature to recreate the function definition
        sig = inspect.signature(func)

        # Build the script content
        script_lines = ["#!/usr/bin/env python3"]

        if add_argparse:
            # Add argparse CLI code
            argparse_code = self.generate_argparse_cli(func)
            script_lines.append("")
            script_lines.append(argparse_code)
            script_lines.append("")

        # Add the function definition
        script_lines.append(f"def {func_name}{sig}:")
        script_lines.append(func_body)
        script_lines.append("")

        # Add main block
        script_lines.append('if __name__ == "__main__":')
        if add_argparse:
            script_lines.append(f"    result = {func_name}(**vars(args))")
        else:
            script_lines.append(f"    {func_name}()")

        # Write to file
        script_content = "\n".join(script_lines)
        Path(script_path).write_text(script_content)

        # Make executable
        import os
        os.chmod(script_path, 0o755)

        return script_path

    def add_layer(
        self, layer: Layer, parent_layers: Optional[List[str]] = None
    ) -> None:
        """
        Add a layer to the DAG.

        :param layer: Layer object to add
        :type layer: Layer
        :param parent_layers: List of parent layer names (dependencies)
        :type parent_layers: Optional[List[str]]
        :raises ValueError: If layer already exists or parent doesn't exist
        """
        if layer.name in self.layers:
            raise ValueError(f"Layer '{layer.name}' already exists in DAG")

        # Validate parent layers exist
        parent_layers = parent_layers or []
        for parent in parent_layers:
            if parent not in self.layers:
                raise ValueError(f"Parent layer '{parent}' does not exist")

        self.layers[layer.name] = layer
        self.dependencies[layer.name] = set(parent_layers)

    def get_layer(self, layer_name: str) -> Layer:
        """
        Get a layer by name.

        :param layer_name: Name of the layer
        :type layer_name: str
        :return: Layer object
        :rtype: Layer
        :raises KeyError: If layer not found
        """
        if layer_name not in self.layers:
            raise KeyError(f"Layer '{layer_name}' not found in DAG")
        return self.layers[layer_name]

    def get_parents(self, layer_name: str) -> Set[str]:
        """
        Get parent layer names for a given layer.

        :param layer_name: Name of the layer
        :type layer_name: str
        :return: Set of parent layer names
        :rtype: Set[str]
        """
        return self.dependencies.get(layer_name, set())

    def get_children(self, layer_name: str) -> Set[str]:
        """
        Get child layer names for a given layer.

        :param layer_name: Name of the layer
        :type layer_name: str
        :return: Set of child layer names
        :rtype: Set[str]
        """
        children = set()
        for name, parents in self.dependencies.items():
            if layer_name in parents:
                children.add(name)
        return children

    def validate(self) -> bool:
        """
        Validate DAG structure - check for cycles.
        Uses Python's graphlib for cycle detection.

        :return: True if valid (acyclic)
        :rtype: bool
        :raises ValueError: If cycle detected
        """
        try:
            sorter = TopologicalSorter(self.dependencies)
            sorter.prepare()
            return True
        except ValueError as e:
            raise ValueError(f"DAG contains a cycle: {e}")

    def topological_order(self) -> List[str]:
        """
        Return layers in topological order.

        :return: List of layer names in execution order
        :rtype: List[str]
        :raises ValueError: If DAG contains cycles
        """
        self.validate()
        sorter = TopologicalSorter(self.dependencies)
        return list(sorter.static_order())

    def to_dict(self) -> dict:
        """
        Serialize entire DAG to dictionary.
        Can be saved as JSON for persistence or exchange.

        :return: Dictionary representation of the entire DAG
        :rtype: dict
        """
        return {
            "layers": {name: layer.to_dict() for name, layer in self.layers.items()},
            "dependencies": {
                name: list(deps) for name, deps in self.dependencies.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DAGGraph":
        """
        Deserialize DAG from dictionary.
        Enables loading from JSON files.

        :param data: Dictionary representation of DAG
        :type data: dict
        :return: DAGGraph instance
        :rtype: DAGGraph
        :raises ValueError: If DAG is invalid
        """
        dag = cls()

        # First pass: create all layers
        for name, layer_data in data["layers"].items():
            layer = Layer.from_dict(layer_data)
            dag.layers[name] = layer

        # Second pass: set dependencies
        for name, deps in data["dependencies"].items():
            dag.dependencies[name] = set(deps)

        dag.validate()
        return dag

    def to_json(self, file_path: Optional[str] = None) -> str:
        """
        Export DAG as JSON string or file.

        :param file_path: Optional path to save JSON file
        :type file_path: Optional[str]
        :return: JSON string
        :rtype: str
        """
        json_str = json.dumps(self.to_dict(), indent=2)
        if file_path:
            with open(file_path, "w") as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, json_input: str) -> "DAGGraph":
        """
        Import DAG from JSON string or file path.

        :param json_input: JSON string or path to JSON file
        :type json_input: str
        :return: DAGGraph instance
        :rtype: DAGGraph
        :raises ValueError: If JSON is invalid or DAG contains cycles
        """
        try:
            # Try as file path first
            with open(json_input, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, OSError):
            # Treat as JSON string
            data = json.loads(json_input)

        return cls.from_dict(data)

    def to_graphviz(self) -> "graphviz.Digraph":
        """
        Convert DAG to Graphviz for visualization.

        :return: Graphviz Digraph object
        :rtype: graphviz.Digraph
        :raises ImportError: If graphviz package not installed
        """
        try:
            from graphviz import Digraph
        except ImportError:
            raise ImportError(
                "graphviz package required for visualization. "
                "Install with: pip install graphviz"
            )

        dot = Digraph("DAGGraph", comment="Dagger Workflow")
        dot.attr(rankdir="TB")  # Top to bottom layout

        # Add nodes (layers)
        for layer_name, layer in self.layers.items():
            label = f"{layer_name}\\n({layer.executable})"
            if layer.job_vars:
                label += f"\\n{len(layer.job_vars)} jobs"
            dot.node(layer_name, label=label, shape="box")

        # Add edges (dependencies)
        for layer_name, parents in self.dependencies.items():
            for parent in parents:
                dot.edge(parent, layer_name)

        return dot

    def visualize(self, output_path: Optional[str] = None, view: bool = True) -> None:
        """
        Visualize DAG and optionally save to file.

        :param output_path: Path to save visualization (without extension)
        :type output_path: Optional[str]
        :param view: Whether to open the visualization
        :type view: bool
        :raises ImportError: If graphviz package not installed
        """
        dot = self.to_graphviz()
        if output_path:
            dot.render(output_path, view=view, cleanup=True)
        else:
            # Just display
            dot.view(cleanup=True)

    def __len__(self) -> int:
        """Return number of layers in DAG."""
        return len(self.layers)

    def __contains__(self, layer_name: str) -> bool:
        """Check if layer exists in DAG."""
        return layer_name in self.layers

    def __getitem__(self, layer_name: str) -> Layer:
        """Dict-like access to layers."""
        return self.get_layer(layer_name)

    def __iter__(self) -> Iterator[str]:
        """Iterate over layer names."""
        return iter(self.layers)

    def __repr__(self) -> str:
        return f"DAGGraph(layers={len(self.layers)}, dependencies={len(self.dependencies)})"
