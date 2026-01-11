"""
Module for all the wrapper classes that will place the wrapped Python functions
into their own scripts, and within a DAG framework via HTCondor DAGMAN.
"""

import inspect
import os
from typing import Optional

import htcondor2
from htcondor2 import dags

from dag_graph import DAGGraph, Layer

# Module-level state for static decorator pattern
_GLOBAL_DAG_INSTANCE: Optional["DAGDecorator"] = None


class DAGBuilderCondor(DAGGraph):
    """
    Base class for building DAGs for HTCondor. This class contains the core DAG-building
    functionality without any user-facing concerns like directory management.
    It provides low-level methods for function parsing, submit object creation,
    and DAG layer management.
    """

    def __init__(self):
        """Initialize base DAG building state."""
        self.dag = htcondor2.dags.DAG()
        self._nlayers = 1
        self._submit_functions = {}
        self._layer_list = []
        self._job_list = []
        self._job_names = []

    def parse_function(
        self,
        func: callable,
        return_as_string: bool = True,
        return_name: bool = False,
        trim_whitespace: bool = True,
    ) -> tuple[str | list[str], str]:
        """
        Parse a Python function and turn it into a list or string
        representation. This function uses the Python inspect module to
        retrieve the source code. This can be dangerous on unverified code, so
        please make sure you know what the source code is doing before wrapping
        it.

        The function can be return as a list of strings, or formatted into
        a valid string that can be directly placed into a Python script.
        Formatting as a list allows for further manipulation and modification
        of the code prior to writing it to a file, for example to add command
        line argument parsing, shebangs etc.

        :param func: The function to parse.
        :type func: callable
        :param return_as_string: If True, return the function as a formatted string.
                                 If False, return as a list of strings.
        :type return_as_string: bool
        :param return_name: If True, return the function name as well.
                            This is useful if you want to use the function name in the script.
        :type return_name: bool
        :param trim_whitespace: If True, trim leading whitespace from the function source code.
                                This will not trim _all_ the whitespace, but only the global indentation level.
        :type trim_whitespace: bool

        :return: The function source code as a string or list of strings.
        :rtype: str or list
        :raises TypeError: If the input is not a callable function.

        :example:
        >>> def example_function(x: int, y: str) -> None:
        ...     print(f"Example function with x={x} and y={y}")
        >>> dagger_instance = dagger()
        >>> dagger_instance.parse_function(example_function)
        """

        if not callable(func):
            raise TypeError("Input must be a callable function.")

        name = func.__name__
        funcstr = inspect.getsource(func)
        funcstr = funcstr.split("\n")[1:]

        # signature = inspect.signature(func)
        # sigstr = f"{name}{signature}:"
        # funcstr = [sigstr] + funcstr

        if trim_whitespace:
            # Trim leading whitespace from the function source code
            # This will not trim _all_ the whitespace, but only the global indentation level
            # Assume first line is never indented, and use that to determine the indent level
            indent_level = len(funcstr[0]) - len(funcstr[0].lstrip())
            funcstr = [line[indent_level:] for line in funcstr]

        if return_as_string:
            funcstr = "\n".join(funcstr)

        if return_name:
            return funcstr, name
        else:
            return funcstr

    def function_to_submit_obj(
        self,
        func: callable,
        dag_dir: str,
        py_script_name: str = "",
        submit_vars: dict = {},
        delimiter: Optional[str] = None,
    ) -> htcondor2.Submit:
        """
        Convert a Python function into a HTCondor Submit object.

        :param func: The function to convert.
        :type func: callable
        :param py_script_name: The name of the Python script to create.
                            If not provided, it defaults to the function name with a `.py` extension.
                            The Python script is created when this function is called. The name can
                            contain the full path to the script. If only a name is provided, it will be generated
                            in the current directory.
        :type py_script_name: str
        :param submit_script_name: The name of the submit script to create.
                            If not provided, it defaults to the function name with a `.submit` extension.
        :type submit_script_name: str
        :param submit_vars: Additional arguments to include in the submit script.
                            This can include command line arguments, Condor requirements, container paths etc.
        :type submit_vars: dict
        :param delimiter: Optional delimiter to use for the function source code. This allows wrapping arbitrary code in
                            the function source code. If provided, the function source code will be wrapped in the delimiter.
                            Any arbitrary string can be used as a delimited, but it is recommended to use a string that
                            is unlikely to appear in the function source code, such as `\"\"\"` or `'\'\'`.
        :type delimiter: str

        :return: HTCondor Submit object
        :rtype: htcondor2.dags.Submit

        :raises TypeError: If the input is not a callable function.
        """

        if not callable(func):
            raise TypeError("Input must be a callable function.")

        funcstr, funcname = self.parse_function(
            func, return_as_string=True, return_name=True
        )

        if delimiter is not None:
            # If a delimiter is provided, we will strip out the first and last instances.
            # This allows embedding arbitrary code into a Python function
            funcstr = funcstr.strip()
            funcstr = funcstr.lstrip(delimiter)
            funcstr = funcstr.rstrip(delimiter)
            funcstr = funcstr.strip()

        if len(py_script_name) == 0:
            py_script_name = os.path.join(dag_dir, f"{func.__name__}.py")
        elif not os.path.isabs(py_script_name):
            # If the script name is not absolute, make it relative to the dag_dir
            py_script_name = os.path.join(dag_dir, py_script_name)

        with open(py_script_name, "w") as f:
            f.write(funcstr)

        # This has to be a relative path to the script, not an absolute path
        # because HTCondor will look for the script in the current working directory
        submit_dict = {
            "executable": os.path.basename(py_script_name),
        }
        submit_dict.update(submit_vars)

        submit_obj = htcondor2.Submit(submit_dict)

        self._submit_functions[funcname] = submit_obj

        return submit_obj

    def dag_layer(
        self,
        submit_obj: htcondor2.Submit,
        submit_vars: list[dict],
        layer_name: str = "",
        parent_layer_name: str = "",
        **kwargs,
    ) -> dags.NodeLayer:
        """
        Create a new layer in the DAG. If no layer name is provided,
        a default name will be generated based on the current number of layers,
        which is incremented each time a new layer is created.

        :param submit_obj: The HTCondor Submit object to associate with this layer.
        :type submit_obj: htcondor2.Submit
        :param submit_vars: Additional arguments to include in the submit script for this layer.
                            This can include command line arguments, Condor requirements, container paths etc.
                            Each dictionary in the list represents a set of variables to be used
                            for a single job in the layer.
        :type submit_vars: list[dict]

        :param layer_name: Optional : Name of the layer to create. If not provided,
                           a default name will be generated based on the current number of layers.
                           The default format is "layer_{n}", where n is the current number of layers.
        :type layer_name: str

        :param parent_layer_name: Name of the parent layer to link to.
                                  If provided, this layer will be added as a child of the parent layer.
        :type parent_layer_name: str

        :param kwargs: Additional keyword arguments to pass to the layer creation.
                       This can include any additional parameters supported by htcondor2.dags.NodeLayer.
        :type kwargs: dict

        :raises ValueError: If parent_layer_name does not exist in the DAG.
        :raises TypeError: If submit_obj is not an instance of htcondor2.Submit.

        :return: A NodeLayer object representing the new layer in the DAG.
        :rtype: htcondor2.dags.NodeLayer

        :example:
        >>> dag = Dagger()
        >>> dag.dag_layer(layer_name="my_layer")
        >>> dag.dag_layer(parent_layer_name="my_layer")
        """

        if not isinstance(submit_obj, htcondor2.Submit):
            raise TypeError(
                "submit_obj must be an instance of htcondor2.Submit or None."
            )

        # Check if parent layer isspecified and exists in the DAG
        if parent_layer_name and parent_layer_name not in self._layer_list:
            raise ValueError(
                f"Parent layer '{parent_layer_name}' does not exist in the DAG. "
                "Please define it first before adding any children."
            )

        if not layer_name:
            layer_name = f"layer_{self._nlayers}"

        if parent_layer_name:
            idx = self._layer_list.index(parent_layer_name)
            job = self._job_list[idx].child_layer(
                name=layer_name,
                submit_description=submit_obj,
                vars=submit_vars,
                **kwargs,
            )
        else:
            job = self.dag.layer(
                name=layer_name,
                submit_description=submit_obj,
                vars=submit_vars,
                **kwargs,
            )

        self._nlayers += 1
        self._layer_list.append(layer_name)
        self._job_list.append(job)
        self._job_names.append(job.name)

        return job


class Dagger(DagBuilderBase):
    """
    Store DAG state, provide methods to wrap Python functions, turn them into submit scripts,
    and add layers to the DAG, define parent/child relationships, and submit the DAG to HTCondor.
    """

    def __init__(self, dag_dir: str, dag_name: str, overwrite_dag_dir: bool = False):
        """
        Initialize the dagger object with a DAG name and directory.

        :param dag_dir: Directory where the DAG files, scripts, and submit files will be stored.
        :type dag_dir: str
        :param dag_name: Name of the DAG.
        :type dag_name: str
        :param overwrite_dag_dir: Whether to overwrite existing files in the DAG directory.
        :type overwrite_dag_dir: bool
        """
        super().__init__()  # Initialize base class
        self.dag_dir = dag_dir
        self.dag_name = dag_name
        self.overwrite_dag_dir = overwrite_dag_dir
        self._setup()

    def _setup(self):
        """
        This method creates the DAG directory if it does not exist and initializes
        the DAG object with the specified name.
        """
        if not os.path.exists(self.dag_dir):
            os.makedirs(self.dag_dir)

        if self.overwrite_dag_dir:
            # Clear existing files in the dag_dir if overwrite is enabled
            for file in os.listdir(self.dag_dir):
                file_path = os.path.join(self.dag_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    @property
    def submit_functions(self) -> dict:
        """
        Return the dictionary of functions that have been submitted to the DAG.
        This is useful for debugging and checking which functions are part of the DAG.

        :return: Dictionary of submitted functions.
        :rtype: dict
        """
        return self._submit_functions

    @property
    def layer_list(self) -> list:
        """
        Return the list of layers in the DAG. This is useful for debugging and checking
        the structure of the DAG.

        :return: List of layers in the DAG.
        :rtype: list
        """
        return self._layer_list

    @property
    def job_list(self) -> list:
        """
        Return the list of jobs in the DAG. This is useful for debugging and checking
        the jobs that have been added to the DAG.

        :return: List of jobs in the DAG.
        :rtype: list
        """
        return self._job_list

    @property
    def job_names(self) -> list:
        """
        Return the list of job names in the DAG. This is useful for debugging and checking
        the names of the jobs that have been added to the DAG.

        :return: List of job names in the DAG.
        :rtype: list
        """
        return self._job_names

    def function_to_submit_obj(self, func: callable, **kwargs) -> htcondor2.Submit:
        """User-facing wrapper that automatically provides dag_dir."""
        return super().function_to_submit_obj(func, self.dag_dir, **kwargs)

    def add_function_to_layer(
        self,
        func: callable,
        py_script_name: str = "",
        submit_vars: dict = {},
        layer_name: str = "",
        parent_layer_name: str = "",
        layer_vars: list[dict] = [],
        delimiter: Optional[str] = None,
        **kwargs,
    ) -> dags.NodeLayer:
        """
        Convert a Python function into a layer in the DAG. This is a convenience method that combines
        the `function_to_submit_obj` and `dag_layer` methods into one. It will automatically convert the function
        to a submit object and add it to the DAG as a layer.

        :param func: The function to convert.
        :type func: callable
        :param py_script_name: The name of the Python script to create.
                               If not provided, it defaults to the function name with a `.py` extension.
        :type py_script_name: str
        :param submit_vars: Additional arguments to include in the submit script.
                            This can include command line arguments, Condor requirements, container paths etc.
        :type submit_vars: dict
        :param layer_name: Optional : Name of the layer to create. If not provided,
                           a default name will be generated based on the current number of layers.
                           The default format is "layer_{n}", where n is the current number of layers.
        :type layer_name: str
        :param parent_layer_name: Name of the parent layer to link to.
                                  If provided, this layer will be added as a child of the parent layer.
        :type parent_layer_name: str
        :param layer_vars: List of dictionaries representing variables to be used for each job in the layer.
                           Each dictionary in the list represents a set of variables to be used
                           for a single job in the layer.
        :type layer_vars: list[dict]
        :param kwargs: Additional keyword arguments to pass to the layer creation.
                       This can include any additional parameters supported by htcondor2.dags.NodeLayer.
        :type kwargs: dict
        :param delimiter: Optional delimiter to use for the function source code. This allows wrapping arbitrary code in
                            the function source code. If provided, the function source code will be wrapped in the delimiter.
                            Any arbitrary string can be used as a delimited, but it is recommended to use a string that
                            is unlikely to appear in the function source code, such as `\"\"\"` or `'\'\'`.
        :type delimiter: str
        :raises TypeError: If the input is not a callable function.

        :return: A NodeLayer object representing the new layer in the DAG.
        :rtype: htcondor2.dags.NodeLayer
        :example:
        >>> from dagger import Dagger
        >>> dag = Dagger(dag_dir="my_dag_dir", dag_name="my_dag")
        >>> def my_function(x: int, y: str) -> None:
        ...     print(f"My function with x={x} and y={y}")
        >>> layer = dag.function_to_layer(
        ...     func=my_function,
        ...     py_script_name="my_function.py",
        ...     submit_vars={"requirements": "Machine == 'my_machine'"},
        ...     layer_name="my_layer",
        ...     parent_layer_name="parent_layer",
        ...     layer_vars=[{"x": 1, "y": "test"}, {"x": 2, "y": "example"}]
        ... )
        """

        if not callable(func):
            raise TypeError("Input must be a callable function.")

        submit_obj = super().function_to_submit_obj(
            func=func,
            dag_dir=self.dag_dir,
            py_script_name=py_script_name,
            submit_vars=submit_vars,
            delimiter=delimiter,
        )

        return super().dag_layer(
            submit_obj=submit_obj,
            submit_vars=layer_vars,
            layer_name=layer_name,
            parent_layer_name=parent_layer_name,
            **kwargs,
        )

    def write_dag(self, **kwargs) -> None:
        """
        Write the current DAG to a file. This will create a .dag file that can be submitted to HTCondor.
        This is a simple wrapepr around the htcondor2.dags.write_dag method, for ease of use with
        this class.

        :param kwargs: Keyword arguments to pass to the DAG writing function.
                       This can include any parameters supported by htcondor2.dags.write_dag
        :type kwargs: dict
        :example:
        >>> dag = Dagger()
        >>> dag.write_dag(filename="my_dag.dag", submit_description="my_submit.sub")

        :return: None
        """

        dags.write_dag(
            self.dag,
            dag_dir=self.dag_dir,
            dag_file_name=f"{self.dag_name}.dag",
            **kwargs,
        )


class DAGDecorator(DagBuilderBase):
    """
    Decorator-based DAG builder that provides the same functionality as Dagger
    but allows functions to be added to the DAG using Python decorators.

    Usage pattern:
    >>> from dagger import DAGDecorator as dd
    >>> @dd.initialize(dag_dir='my_dag', dag_name='workflow')
    ... @dd.layer(layer_name='step1')
    ... def process_data(input_file: str) -> str:
    ...     # Function implementation
    ...     return "processed_data.txt"
    >>> @dd.layer(layer_name='step2', parent_layer_name='step1',
    ...          layer_vars=[{"input": "data1.txt"}, {"input": "data2.txt"}])
    ... def analyze_data(input: str) -> str:
    ...     # Function implementation
    ...     return "analysis_results.txt"
    >>> dd.write_dag()

    This will create a DAG with two layers, where 'step2' depends on 'step1'.
    It will also create the necessary Python scripts and submit files in the specified dag_dir.
    """

    def __init__(self, dag_dir: str, dag_name: str, overwrite_dag_dir: bool = False):
        """
        Initialize decorator-based DAG builder.

        :param dag_dir: Directory where the DAG files, scripts, and submit files will be stored.
        :type dag_dir: str
        :param dag_name: Name of the DAG.
        :type dag_name: str
        :param overwrite_dag_dir: Whether to overwrite existing files in the DAG directory.
        :type overwrite_dag_dir: bool
        """
        super().__init__()
        self.dag_dir = dag_dir
        self.dag_name = dag_name
        self.overwrite_dag_dir = overwrite_dag_dir
        self._setup()

    def _setup(self):
        """
        Create the DAG directory if it does not exist and clear existing files if requested.
        """
        if not os.path.exists(self.dag_dir):
            os.makedirs(self.dag_dir)

        if self.overwrite_dag_dir:
            # Clear existing files in the dag_dir if overwrite is enabled
            for file in os.listdir(self.dag_dir):
                file_path = os.path.join(self.dag_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    @property
    def submit_functions(self) -> dict:
        """
        Return the dictionary of functions that have been submitted to the DAG.

        :return: Dictionary of submitted functions.
        :rtype: dict
        """
        return self._submit_functions

    @property
    def layer_list(self) -> list:
        """
        Return the list of layers in the DAG.

        :return: List of layers in the DAG.
        :rtype: list
        """
        return self._layer_list

    @property
    def job_list(self) -> list:
        """
        Return the list of jobs in the DAG.

        :return: List of jobs in the DAG.
        :rtype: list
        """
        return self._job_list

    @property
    def job_names(self) -> list:
        """
        Return the list of job names in the DAG.

        :return: List of job names in the DAG.
        :rtype: list
        """
        return self._job_names

    # Static methods for global decorator pattern
    @classmethod
    def initialize(cls, dag_dir: str, dag_name: str, overwrite_dag_dir: bool = False):
        """
        Initialize the global DAGDecorator instance for static decorator usage.

        This enables the usage pattern:
        @DAGDecorator.initialize(dag_dir='my_dag', dag_name='workflow')
        @DAGDecorator.layer(layer_name='layer1')
        def func1():
            pass

        :param dag_dir: Directory where the DAG files, scripts, and submit files will be stored.
        :type dag_dir: str
        :param dag_name: Name of the DAG.
        :type dag_name: str
        :param overwrite_dag_dir: Whether to overwrite existing files in the DAG directory.
        :type overwrite_dag_dir: bool
        :return: Decorator function that returns the original function unchanged
        :rtype: callable
        """

        def decorator(func: callable):
            global _GLOBAL_DAG_INSTANCE
            _GLOBAL_DAG_INSTANCE = cls(
                dag_dir=dag_dir, dag_name=dag_name, overwrite_dag_dir=overwrite_dag_dir
            )
            return func

        return decorator

    @classmethod
    def layer(
        cls,
        layer_name: str = "",
        parent_layer_name: str = "",
        layer_vars: Optional[list[dict]] = None,
        py_script_name: str = "",
        submit_vars: Optional[dict] = None,
        delimiter: Optional[str] = None,
        **kwargs,
    ):
        """
        Static decorator to add a function to the global DAG as a layer.

        Must be used after @DAGDecorator.initialize().

        :param layer_name: Name of the layer. If not provided, uses function name.
        :type layer_name: str
        :param parent_layer_name: Name of parent layer for dependencies.
        :type parent_layer_name: str
        :param layer_vars: List of variable dictionaries for multiple jobs in this layer.
        :type layer_vars: list[dict]
        :param py_script_name: Custom name for the Python script file.
        :type py_script_name: str
        :param submit_vars: Additional variables for the submit script.
        :type submit_vars: dict
        :param delimiter: Optional delimiter for function source code wrapping.
        :type delimiter: str
        :param kwargs: Additional arguments for layer creation.
        :return: Decorator function
        :rtype: callable
        :raises RuntimeError: If DAGDecorator.initialize() has not been called first.

        :example:
        >>> from dagger import DAGDecorator as dd
        >>> @dd.initialize(dag_dir='my_dag', dag_name='workflow')
        ... @dd.layer(layer_name='step1')
        ... def process_data(input_file: str) -> str:
        ...
        ...
        >>>     return "processed_data.txt"
        >>> @dd.layer(layer_name='step2', parent_layer_name='step1',
        ...          layer_vars=[{"input": "data1.txt"}, {"input": "data2.txt"}])
        ... def analyze_data(input: str) -> str:
        ...
        >>>     return "analysis_results.txt"
        >>> dd.write_dag()
        """

        def decorator(func: callable):
            global _GLOBAL_DAG_INSTANCE
            if _GLOBAL_DAG_INSTANCE is None:
                raise RuntimeError(
                    "DAGDecorator.initialize() must be called before using DAGDecorator.layer(). "
                    "Use @DAGDecorator.initialize(dag_dir='...', dag_name='...') on a function first."
                )

            return _GLOBAL_DAG_INSTANCE.layer(
                layer_name=layer_name,
                parent_layer_name=parent_layer_name,
                layer_vars=layer_vars,
                py_script_name=py_script_name,
                submit_vars=submit_vars,
                delimiter=delimiter,
                **kwargs,
            )(func)

        return decorator

    @classmethod
    def write_dag(cls, **kwargs) -> None:
        """
        Write the global DAG to a file using the static decorator pattern.

        :param kwargs: Keyword arguments to pass to the DAG writing function.
        :type kwargs: dict
        :return: None
        :raises RuntimeError: If DAGDecorator.initialize() has not been called first.
        """
        global _GLOBAL_DAG_INSTANCE
        if _GLOBAL_DAG_INSTANCE is None:
            raise RuntimeError(
                "DAGDecorator.initialize() must be called before using DAGDecorator.write_dag(). "
                "Use @DAGDecorator.initialize(dag_dir='...', dag_name='...') on a function first."
            )

        _GLOBAL_DAG_INSTANCE.write_dag(**kwargs)

    @classmethod
    def reset(cls) -> None:
        """
        Reset the global DAGDecorator instance. Useful for testing or starting fresh.

        :return: None
        """
        global _GLOBAL_DAG_INSTANCE
        _GLOBAL_DAG_INSTANCE = None

    @classmethod
    def get_global_instance(cls) -> Optional["DAGDecorator"]:
        """
        Get the current global DAGDecorator instance.

        :return: The global DAGDecorator instance or None if not initialized.
        :rtype: DAGDecorator or None
        """
        return _GLOBAL_DAG_INSTANCE


# TODO : Add the ability to read common attributes from a TOML file or something
