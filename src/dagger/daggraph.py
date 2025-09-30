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
class DataProduct:
    """
    Represents a data product (edge) in the DAG.

    :param name: Unique identifier for this data product
    :type name: str
    :param file_path: Path to the data file
    :type file_path: Optional[Path]
    :param data_type: Type of data (e.g., "MeasurementSet", "FITS", "HDF5", arbitrary string)
    :type data_type: Optional[str]
    :param metadata: Additional metadata about the data product
    :type metadata: Dict[str, Any]

    :example:
    >>> dp = DataProduct(name="calibrated_ms", file_path=Path("/data/calibrated.ms"), data_type="MeasurementSet")
    >>> print(dp)
    DataProduct(name='calibrated_ms', file_path=PosixPath('/data/calibrated.ms'), data_type='MeasurementSet', metadata={})

    :note: DataProduct instances are typically created and managed by the DAGGraph when layers are added.
    """

    name: str
    file_path: Optional[Path] = None
    data_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """
        Serialize to dictionary.

        :return: Dictionary representation
        :rtype: dict
        """
        return {
            "name": self.name,
            "file_path": str(self.file_path) if self.file_path else None,
            "data_type": self.data_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DataProduct":
        """
        Deserialize from dictionary.

        :param data: Dictionary representation
        :type data: dict
        :return: DataProduct instance
        :rtype: DataProduct
        """
        data = data.copy()
        if data.get("file_path"):
            data["file_path"] = Path(data["file_path"])
        return cls(**data)


@dataclass
class Layer:
    """
    Represents a computational layer (node) in the DAG.
    A layer can contain multiple jobs with different parameters.

    :param name: Unique name for this layer
    :type name: str
    :param executable: Name of the function to execute
    :type executable: str
    :param exec_environment: Source code of the function
    :type exec_environment: Optional[str]
    :param job_vars: List of variable dictionaries, one per job in this layer
    :type job_vars: List[Dict[str, Any]]
    :param inputs: Input data products
    :type inputs: List[DataProduct]
    :param outputs: Output data products
    :type outputs: List[DataProduct]
    :param submit_vars: Backend-agnostic submit variables
    :type submit_vars: Dict[str, Any]
    :param metadata: Additional metadata
    :type metadata: Dict[str, Any]

    :example:
    >>> layer = Layer(
    name="calibration",
    executable="calibrate_data.py",
    exec_environment='/path/to/container.sif',
    job_vars=[{"input_file": "raw1.ms"}, {"input_file": "raw2.ms"}],
    inputs=[DataProduct(name="raw1", file_path=Path("/data/raw1.ms"), data_type="MeasurementSet")],
    outputs=[DataProduct(name="calibrated1", file_path=Path("/data/calibrated1.ms"), data_type="MeasurementSet")],
    submit_vars={"memory": "4GB", "cpus": 2},
    metadata={"description": "Calibration layer"}
    )
    >>> print(layer)
    Layer(name='calibration', executable='calibrate_data.py', exec_environment='/path/to/container.sif',
        job_vars=[{'input_file': 'raw1.ms'}, {'input_file': 'raw2.ms'}],
        inputs=[DataProduct(name='raw1', file_path=PosixPath('/data/raw1.ms'),
        data_type='MeasurementSet', metadata={})],
        outputs=[DataProduct(name='calibrated1', file_path=PosixPath('/data/calibrated1.ms'),
        data_type='MeasurementSet', metadata={})],
        submit_vars={'memory': '4GB', 'cpus': 2}, metadata={'description': 'Calibration layer'})

    :note: Layer instances are typically created and managed by the DAGGraph when building the workflow.
    """

    name: str
    executable: str
    exec_environment: Optional[str] = None

    # Job variables - each dict represents one job in this layer
    job_vars: List[Dict[str, Any]] = field(default_factory=list)

    # Data products
    inputs: List[DataProduct] = field(default_factory=list)
    outputs: List[DataProduct] = field(default_factory=list)

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
            "exec_environment": self.exec_environment,
            "job_vars": self.job_vars,
            "inputs": [inp.to_dict() for inp in self.inputs],
            "outputs": [out.to_dict() for out in self.outputs],
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
        data = data.copy()
        data["inputs"] = [DataProduct.from_dict(d) for d in data.get("inputs", [])]
        data["outputs"] = [DataProduct.from_dict(d) for d in data.get("outputs", [])]
        return cls(**data)


class DAGGraph:
    """
    Abstract DAG representation inspired by Dask's HighLevelGraph.
    Stores layers (nodes) and dependencies (edges) separately.

    Similar to Dask's structure:
    - layers: Dict[str, Layer] - the computational nodes
    - dependencies: Dict[str, Set[str]] - parent relationships

    Additional for radio astronomy:
    - data_products: Dict[str, DataProduct] - edge metadata

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
        self.data_products: Dict[str, DataProduct] = {}

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

        # Register data products
        for data_product in layer.inputs + layer.outputs:
            self.data_products[data_product.name] = data_product

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
            "data_products": {
                name: dp.to_dict() for name, dp in self.data_products.items()
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

        # Third pass: data products
        for name, dp_data in data.get("data_products", {}).items():
            dag.data_products[name] = DataProduct.from_dict(dp_data)

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
                # Try to find data product connecting these layers
                edge_label = ""
                for dp_name, dp in self.data_products.items():
                    if dp.data_type:
                        edge_label = dp.data_type
                        break
                dot.edge(parent, layer_name, label=edge_label)

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
