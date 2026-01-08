import os
import shutil
import tempfile

import htcondor2

from dagger.dagger import DagBuilderBase, Dagcorator, Dagger


def test_base_class_parse_function():
    """Test DagBuilderBase.parse_function method."""
    base = DagBuilderBase()

    def sample_function(x: int, y: str) -> str:
        """A sample function to test parsing."""
        return f"Received {x} and {y}"

    func_str = base.parse_function(sample_function, return_as_string=True)
    # assert "def sample_function(x: int, y: str) -> str:" in func_str
    assert 'return f"Received {x} and {y}"' in func_str


def test_base_class_parse_function_as_list():
    """Test DagBuilderBase.parse_function method with return_as_string=False."""
    base = DagBuilderBase()

    def another_function(a: float, b: float) -> float:
        """Another sample function to test parsing."""
        return a + b

    func_list = base.parse_function(another_function, return_as_string=False)
    # assert "def another_function(a: float, b: float) -> float:" in func_list[0]
    assert "return a + b" in func_list[-2]


def test_base_class_parse_function_invalid_input():
    """Test DagBuilderBase.parse_function method with invalid input."""
    base = DagBuilderBase()

    try:
        base.parse_function("not_a_function")
    except TypeError as e:
        assert str(e) == "Input must be a callable function."
    else:
        assert False, "Expected TypeError was not raised."


def test_base_class_function_to_submit_obj():
    """Test DagBuilderBase.function_to_submit_obj method."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base = DagBuilderBase()

        def test_function(x: int) -> str:
            return f"Test {x}"

        submit_obj = base.function_to_submit_obj(test_function, temp_dir)
        assert isinstance(submit_obj, htcondor2.Submit)
        assert "test_function" in base._submit_functions

        # Check that the Python script was created
        script_path = os.path.join(temp_dir, "test_function.py")
        assert os.path.exists(script_path)


def test_base_class_dag_layer():
    """Test DagBuilderBase.dag_layer method."""
    base = DagBuilderBase()

    submit_obj = htcondor2.Submit({"executable": "test.py"})
    layer = base.dag_layer(submit_obj, [{"arg": "value"}], "test_layer")

    assert layer.name == "test_layer"
    assert "test_layer" in base._layer_list
    assert len(base._job_list) == 1
    assert base._nlayers == 2


def test_base_class_dag_layer_with_parent():
    """Test DagBuilderBase.dag_layer method with parent layer."""
    base = DagBuilderBase()

    # Create parent layer first
    submit_obj = htcondor2.Submit({"executable": "parent.py"})
    parent_layer = base.dag_layer(submit_obj, [{"arg": "parent"}], "parent_layer")

    # Create child layer
    child_layer = base.dag_layer(
        submit_obj, [{"arg": "child"}], "child_layer", "parent_layer"
    )

    assert child_layer.name == "child_layer"
    assert "child_layer" in base._layer_list
    assert len(base._job_list) == 2


def test_dagger_initialization():
    """Test Dagger class initialization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dag = Dagger(dag_dir=temp_dir, dag_name="test")
        assert dag.dag_name == "test"
        assert dag.dag_dir == temp_dir
        assert isinstance(dag.dag, htcondor2.dags.DAG)
        assert dag._nlayers == 1
        assert os.path.exists(temp_dir)


def test_dagger_add_function_to_layer():
    """Test Dagger.add_function_to_layer method (high-level wrapper)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dag = Dagger(dag_dir=temp_dir, dag_name="test")

        def sample_function(a: int, b: int) -> int:
            """A sample function to test func_to_layer."""
            return a + b

        layer = dag.add_function_to_layer(sample_function, layer_name="addition_layer")
        assert layer.name == "addition_layer"
        assert len(dag.layer_list) == 1
        assert "addition_layer" in dag.layer_list


def test_dagger_properties():
    """Test Dagger class properties."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dag = Dagger(dag_dir=temp_dir, dag_name="test")

        def test_func():
            pass

        dag.function_to_submit_obj(test_func)

        assert "test_func" in dag.submit_functions
        assert isinstance(dag.layer_list, list)
        assert isinstance(dag.job_list, list)
        assert isinstance(dag.job_names, list)


def test_dagger_write_dag():
    """Test Dagger.write_dag method."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dag = Dagger(dag_dir=temp_dir, dag_name="test_dag")

        def simple_function():
            print("Hello World")

        dag.add_function_to_layer(simple_function, layer_name="hello_layer")
        dag.write_dag()

        # Check that DAG file was created
        dag_file = os.path.join(temp_dir, "test_dag.dag")
        assert os.path.exists(dag_file)


def test_full_workflow():
    """Test complete workflow from function to DAG layer."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dag = Dagger(dag_dir=temp_dir, dag_name="full_test")

        def step1(input_data: str) -> str:
            return f"Processed: {input_data}"

        def step2(processed_data: str) -> str:
            return f"Final: {processed_data}"

        # Add first layer
        layer1 = dag.add_function_to_layer(
            step1, layer_name="process_layer", layer_vars=[{"input_data": "test"}]
        )

        # Add second layer with dependency
        layer2 = dag.add_function_to_layer(
            step2,
            layer_name="final_layer",
            parent_layer_name="process_layer",
            layer_vars=[{"processed_data": "test_processed"}],
        )

        assert len(dag.layer_list) == 2
        assert "process_layer" in dag.layer_list
        assert "final_layer" in dag.layer_list

        # Write the complete DAG
        dag.write_dag()

        # Verify DAG file exists
        dag_file = os.path.join(temp_dir, "full_test.dag")
        assert os.path.exists(dag_file)


def test_error_handling():
    """Test error handling for invalid inputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dag = Dagger(dag_dir=temp_dir, dag_name="error_test")

        # Test invalid function input
        try:
            dag.function_to_submit_obj("not_a_function")
        except TypeError as e:
            assert str(e) == "Input must be a callable function."
        else:
            assert False, "Expected TypeError was not raised."

        # Test invalid parent layer
        def test_func():
            pass

        try:
            dag.add_function_to_layer(test_func, parent_layer_name="nonexistent_layer")
        except ValueError as e:
            assert "does not exist in the DAG" in str(e)
        else:
            assert False, "Expected ValueError was not raised."


# Tests for static decorator pattern
def test_dagdecorator_static_initialize():
    """Test DAGDecorator static initialize method."""
    from dagger.dagger import DAGDecorator

    # Reset global state
    DAGDecorator.reset()
    assert DAGDecorator.get_global_instance() is None

    with tempfile.TemporaryDirectory() as temp_dir:
        @DAGDecorator.initialize(dag_dir=temp_dir, dag_name="static_test")
        def dummy_func():
            pass

        # Verify global instance was created
        instance = DAGDecorator.get_global_instance()
        assert instance is not None
        assert instance.dag_name == "static_test"
        assert instance.dag_dir == temp_dir

        # Clean up
        DAGDecorator.reset()


def test_dagdecorator_static_layer():
    """Test DAGDecorator static layer decorator."""
    from dagger.dagger import DAGDecorator

    # Reset global state
    DAGDecorator.reset()

    with tempfile.TemporaryDirectory() as temp_dir:
        @DAGDecorator.initialize(dag_dir=temp_dir, dag_name="static_layer_test")
        def dummy_init():
            pass

        @DAGDecorator.layer(layer_name="test_layer")
        def test_function():
            """Test function for static decorator."""
            return "test result"

        # Verify the layer was added
        instance = DAGDecorator.get_global_instance()
        assert "test_layer" in instance.layer_list
        assert "test_function" in instance.submit_functions

        # Clean up
        DAGDecorator.reset()


def test_dagdecorator_static_write_dag():
    """Test DAGDecorator static write_dag method."""
    from dagger.dagger import DAGDecorator

    # Reset global state
    DAGDecorator.reset()

    with tempfile.TemporaryDirectory() as temp_dir:
        @DAGDecorator.initialize(dag_dir=temp_dir, dag_name="static_write_test")
        def dummy_init():
            pass

        @DAGDecorator.layer(layer_name="test_layer")
        def test_function():
            """Test function for write DAG."""
            return "test result"

        # Write the DAG
        DAGDecorator.write_dag()

        # Verify DAG file was created
        dag_file = os.path.join(temp_dir, "static_write_test.dag")
        assert os.path.exists(dag_file)

        # Clean up
        DAGDecorator.reset()


def test_dagdecorator_static_error_handling():
    """Test DAGDecorator static methods error handling."""
    from dagger.dagger import DAGDecorator

    # Reset global state
    DAGDecorator.reset()

    # Test layer decorator without initialization
    try:
        @DAGDecorator.layer(layer_name="test_layer")
        def test_func():
            pass
    except RuntimeError as e:
        assert "DAGDecorator.initialize() must be called" in str(e)
    else:
        assert False, "Expected RuntimeError was not raised."

    # Test write_dag without initialization
    try:
        DAGDecorator.write_dag()
    except RuntimeError as e:
        assert "DAGDecorator.initialize() must be called" in str(e)
    else:
        assert False, "Expected RuntimeError was not raised."

    # Clean up
    DAGDecorator.reset()


def test_dagdecorator_static_parent_layers():
    """Test DAGDecorator static layer decorator with parent layers."""
    from dagger.dagger import DAGDecorator

    # Reset global state
    DAGDecorator.reset()

    with tempfile.TemporaryDirectory() as temp_dir:
        @DAGDecorator.initialize(dag_dir=temp_dir, dag_name="static_parent_test")
        def dummy_init():
            pass

        @DAGDecorator.layer(layer_name="parent_layer")
        def parent_function():
            """Parent layer function."""
            return "parent result"

        @DAGDecorator.layer(layer_name="child_layer", parent_layer_name="parent_layer")
        def child_function():
            """Child layer function."""
            return "child result"

        # Verify both layers were added
        instance = DAGDecorator.get_global_instance()
        assert "parent_layer" in instance.layer_list
        assert "child_layer" in instance.layer_list
        assert "parent_function" in instance.submit_functions
        assert "child_function" in instance.submit_functions

        # Clean up
        DAGDecorator.reset()


def test_dagcorator_initialization():
    """Test Dagcorator class initialization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dagcorator = Dagcorator(dag_dir=temp_dir, dag_name="test_decorator")
        assert dagcorator.dag_name == "test_decorator"
        assert dagcorator.dag_dir == temp_dir
        assert isinstance(dagcorator.dag, htcondor2.dags.DAG)
        assert dagcorator._nlayers == 1
        assert os.path.exists(temp_dir)


def test_dagcorator_layer_decorator():
    """Test Dagcorator.layer decorator functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dagcorator = Dagcorator(dag_dir=temp_dir, dag_name="decorator_test")

        @dagcorator.layer(layer_name="test_layer")
        def sample_function(x: int, y: str) -> str:
            """A sample function for decorator testing."""
            return f"Result: {x}, {y}"

        # Check that function was added to DAG
        assert "test_layer" in dagcorator.layer_list
        assert "sample_function" in dagcorator.submit_functions
        assert len(dagcorator.job_list) == 1

        # Check that function still works normally
        result = sample_function(42, "test")
        assert result == "Result: 42, test"

        # Check that function has DAG layer attribute
        assert hasattr(sample_function, "_dag_layer")


def test_dagcorator_layer_with_parent():
    """Test Dagcorator layer decorator with parent dependencies."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dagcorator = Dagcorator(dag_dir=temp_dir, dag_name="parent_test")

        @dagcorator.layer(layer_name="parent_layer")
        def parent_function():
            return "parent_result"

        @dagcorator.layer(layer_name="child_layer", parent_layer_name="parent_layer")
        def child_function():
            return "child_result"

        assert len(dagcorator.layer_list) == 2
        assert "parent_layer" in dagcorator.layer_list
        assert "child_layer" in dagcorator.layer_list


def test_dagcorator_layer_with_variables():
    """Test Dagcorator layer decorator with multiple job variables."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dagcorator = Dagcorator(dag_dir=temp_dir, dag_name="vars_test")

        @dagcorator.layer(
            layer_name="multi_job_layer",
            layer_vars=[
                {"param1": "value1"},
                {"param1": "value2"},
                {"param1": "value3"},
            ],
        )
        def multi_job_function(param1: str):
            return f"Processed {param1}"

        assert "multi_job_layer" in dagcorator.layer_list
        assert "multi_job_function" in dagcorator.submit_functions


def test_dagcorator_default_layer_name():
    """Test Dagcorator using function name as default layer name."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dagcorator = Dagcorator(dag_dir=temp_dir, dag_name="default_name_test")

        @dagcorator.layer()
        def my_custom_function_name():
            return "result"

        # Should use function name as layer name
        assert "my_custom_function_name" in dagcorator.layer_list


def test_dagcorator_properties():
    """Test Dagcorator class properties."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dagcorator = Dagcorator(dag_dir=temp_dir, dag_name="props_test")

        @dagcorator.layer(layer_name="prop_test_layer")
        def test_function():
            pass

        assert isinstance(dagcorator.submit_functions, dict)
        assert isinstance(dagcorator.layer_list, list)
        assert isinstance(dagcorator.job_list, list)
        assert isinstance(dagcorator.job_names, list)
        assert "test_function" in dagcorator.submit_functions


def test_dagcorator_write_dag():
    """Test Dagcorator.write_dag method."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dagcorator = Dagcorator(dag_dir=temp_dir, dag_name="write_test")

        @dagcorator.layer(layer_name="write_layer")
        def simple_function():
            print("Hello from decorator!")

        dagcorator.write_dag()

        # Check that DAG file was created
        dag_file = os.path.join(temp_dir, "write_test.dag")
        assert os.path.exists(dag_file)


def test_dagcorator_full_workflow():
    """Test complete Dagcorator workflow with multiple layers and dependencies."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dagcorator = Dagcorator(dag_dir=temp_dir, dag_name="full_decorator_workflow")

        @dagcorator.layer(layer_name="step1", layer_vars=[{"input": "file1.txt"}])
        def process_step(input: str) -> str:
            return f"Processed {input}"

        @dagcorator.layer(
            layer_name="step2",
            parent_layer_name="step1",
            layer_vars=[{"data": "processed_data"}],
        )
        def analyze_step(data: str) -> str:
            return f"Analyzed {data}"

        @dagcorator.layer(layer_name="step3", parent_layer_name="step2")
        def finalize_step() -> str:
            return "Workflow complete"

        # Verify DAG structure
        assert len(dagcorator.layer_list) == 3
        assert "step1" in dagcorator.layer_list
        assert "step2" in dagcorator.layer_list
        assert "step3" in dagcorator.layer_list

        # Verify functions still work
        assert process_step("test.txt") == "Processed test.txt"
        assert analyze_step("test_data") == "Analyzed test_data"
        assert finalize_step() == "Workflow complete"

        # Write and verify DAG file
        dagcorator.write_dag()
        dag_file = os.path.join(temp_dir, "full_decorator_workflow.dag")
        assert os.path.exists(dag_file)


def test_dagcorator_error_handling():
    """Test Dagcorator error handling."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dagcorator = Dagcorator(dag_dir=temp_dir, dag_name="error_test")

        # Test invalid parent layer
        try:

            @dagcorator.layer(
                layer_name="child", parent_layer_name="nonexistent_parent"
            )
            def invalid_child():
                pass

        except ValueError as e:
            assert "does not exist in the DAG" in str(e)
        else:
            assert False, "Expected ValueError was not raised."


# Tests for static decorator pattern
def test_dagdecorator_static_initialize():
    """Test DAGDecorator static initialize method."""
    from dagger.dagger import DAGDecorator

    # Reset global state
    DAGDecorator.reset()
    assert DAGDecorator.get_global_instance() is None

    with tempfile.TemporaryDirectory() as temp_dir:
        @DAGDecorator.initialize(dag_dir=temp_dir, dag_name="static_test")
        def dummy_func():
            pass

        # Verify global instance was created
        instance = DAGDecorator.get_global_instance()
        assert instance is not None
        assert instance.dag_name == "static_test"
        assert instance.dag_dir == temp_dir

        # Clean up
        DAGDecorator.reset()


def test_dagdecorator_static_layer():
    """Test DAGDecorator static layer decorator."""
    from dagger.dagger import DAGDecorator

    # Reset global state
    DAGDecorator.reset()

    with tempfile.TemporaryDirectory() as temp_dir:
        @DAGDecorator.initialize(dag_dir=temp_dir, dag_name="static_layer_test")
        def dummy_init():
            pass

        @DAGDecorator.layer(layer_name="test_layer")
        def test_function():
            """Test function for static decorator."""
            return "test result"

        # Verify the layer was added
        instance = DAGDecorator.get_global_instance()
        assert "test_layer" in instance.layer_list
        assert "test_function" in instance.submit_functions

        # Clean up
        DAGDecorator.reset()


def test_dagdecorator_static_write_dag():
    """Test DAGDecorator static write_dag method."""
    from dagger.dagger import DAGDecorator

    # Reset global state
    DAGDecorator.reset()

    with tempfile.TemporaryDirectory() as temp_dir:
        @DAGDecorator.initialize(dag_dir=temp_dir, dag_name="static_write_test")
        def dummy_init():
            pass

        @DAGDecorator.layer(layer_name="test_layer")
        def test_function():
            """Test function for write DAG."""
            return "test result"

        # Write the DAG
        DAGDecorator.write_dag()

        # Verify DAG file was created
        dag_file = os.path.join(temp_dir, "static_write_test.dag")
        assert os.path.exists(dag_file)

        # Clean up
        DAGDecorator.reset()


def test_dagdecorator_static_error_handling():
    """Test DAGDecorator static methods error handling."""
    from dagger.dagger import DAGDecorator

    # Reset global state
    DAGDecorator.reset()

    # Test layer decorator without initialization
    try:
        @DAGDecorator.layer(layer_name="test_layer")
        def test_func():
            pass
    except RuntimeError as e:
        assert "DAGDecorator.initialize() must be called" in str(e)
    else:
        assert False, "Expected RuntimeError was not raised."

    # Test write_dag without initialization
    try:
        DAGDecorator.write_dag()
    except RuntimeError as e:
        assert "DAGDecorator.initialize() must be called" in str(e)
    else:
        assert False, "Expected RuntimeError was not raised."

    # Clean up
    DAGDecorator.reset()


def test_dagdecorator_static_parent_layers():
    """Test DAGDecorator static layer decorator with parent layers."""
    from dagger.dagger import DAGDecorator

    # Reset global state
    DAGDecorator.reset()

    with tempfile.TemporaryDirectory() as temp_dir:
        @DAGDecorator.initialize(dag_dir=temp_dir, dag_name="static_parent_test")
        def dummy_init():
            pass

        @DAGDecorator.layer(layer_name="parent_layer")
        def parent_function():
            """Parent layer function."""
            return "parent result"

        @DAGDecorator.layer(layer_name="child_layer", parent_layer_name="parent_layer")
        def child_function():
            """Child layer function."""
            return "child result"

        # Verify both layers were added
        instance = DAGDecorator.get_global_instance()
        assert "parent_layer" in instance.layer_list
        assert "child_layer" in instance.layer_list
        assert "parent_function" in instance.submit_functions
        assert "child_function" in instance.submit_functions

        # Clean up
        DAGDecorator.reset()
