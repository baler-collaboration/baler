import io
import os
import pstats
import cProfile
from pstats import SortKey
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import codecarbon
from ..modules import profile_plotting
import subprocess


def pytorch_profile(f, *args, **kwargs):
    """
    This function performs PyTorch profiling of CPU, GPU time and memory
    consumed by the function f execution.

    Args:
        f (callable): The function to be profiled.

    Returns:
        result: The result of the function `f` execution.
    """

    if torch.cuda.is_available():
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    else:
        activities = [ProfilerActivity.CPU]

    # Start profiler before the function will be executed
    with profile(
        activities=activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "log/baler", worker_name="worker0"
        ),
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        with record_function(f"{f.__name__}"):
            # Call the function
            result = f(*args, **kwargs)
            prof.step()
            prof.stop()

    # Print the CPU time for each torch operation
    print(prof.key_averages().table(sort_by="cpu_time_total"))

    # Store the information about CPU and GPU usage
    if torch.cuda.is_available():
        prof.export_stacks("profiler_stacks.json", "self_cuda_time_total")

    # Store the results to the .json file
    prof.export_stacks("/tmp/profiler_stacks.json", "self_cpu_time_total")

    return result


def energy_profiling(f, output_path, project_name, measure_power_secs, *args, **kwargs):
    """
    Energy Profiling measures the amount of electricity that
    was consumed by the given function f and the amount of CO2 emission.
    It utilizes the codecarbon package for tracking this information.

    Args:
        f (callable): The function to be profiled.
        output_path (str): The path where the profiling logs and reports are to be saved.
        project_name (str): The name of the project.
        measure_power_secs (int): The number of seconds to measure power.

    Returns:
        result: The result of the function `f` execution.
        profile_plotting.plot(profiling_path, f): Subsequently called to generate plots from the codecarbon log files.
    """

    profiling_path = os.path.join(output_path, "profiling")
    tracker = codecarbon.EmissionsTracker(
        project_name=project_name,
        measure_power_secs=measure_power_secs,
        save_to_file=True,
        output_dir=profiling_path,
        co2_signal_api_token="script-overwrite",
        experiment_id="235b1da5-aaaa-aaaa-aaaa-893681599d2c",
        log_level="DEBUG",
        tracking_mode="process",
    )
    tracker.start_task(f"{f.__name__}")

    # Execute the function and get its result
    result = f(*args, **kwargs)

    tracker.stop_task()
    emissions = tracker.stop()

    print(
        "----------------------------------Energy Profile-----------------------------------------------"
    )
    print(
        "-----------------------------------------------------------------------------------------------"
    )
    print(f"Emissions : {1000 * emissions} g CO₂")
    for task_name, task in tracker._tasks.items():
        print(
            f"Emissions : {1000 * task.emissions_data.emissions} g CO₂ for task {task_name} \nEmission Rate : {3600*task.emissions_data.emissions_rate} Kg/h"
        )
        print(
            "-----------------------------------------------------------------------------------------------"
        )
        print("Energy Consumption")
        print(
            f"CPU : {1000 * task.emissions_data.cpu_energy} Wh \nGPU : {1000 * task.emissions_data.gpu_energy} Wh \nRAM : {1000 * task.emissions_data.ram_energy} Wh"
        )
        print(
            "-----------------------------------------------------------------------------------------------"
        )
        print("Power Consumption")
        print(
            f"CPU : { task.emissions_data.cpu_power} W \nGPU : { task.emissions_data.gpu_power} W \nRAM : { task.emissions_data.ram_power} W"
            + f"\nduring {task.emissions_data.duration} seconds."
        )

    return result, profile_plotting.plot(profiling_path, f)


def c_profile(func, output_path, *args, **kwargs):
    """
    Profile the function func with cProfile.

    Args:
        func (callable): The function to be profiled.
        output_path (str): The path where the profiling logs and reports are to be saved.

    Returns:
        result: The result of the function `func` execution.
    """
    profiling_path_prof = os.path.join(output_path, f"profiling/{func.__name__}.prof")
    profiling_path_pstats = os.path.join(
        output_path, f"profiling/{func.__name__}.pstats"
    )
    pr = cProfile.Profile()
    pr.enable()
    # Execute the function and get its result
    result = func(*args, **kwargs)
    pr.disable()
    pr.dump_stats(profiling_path_prof)
    pr.dump_stats(profiling_path_pstats)
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

    generate_call_graphs(func, profiling_path_pstats, output_path)

    visualize_profiling_data(profiling_path_prof)

    return result


def generate_call_graphs(func, profiling_path_pstats, output_path):
    """
    Generate call graphs and directed graphs (digraphs) for a given Python function.

    Args:
      func (callable): The Python function for which call graphs will be generated.
      profiling_path_pstats (str): The path to the profiling data in pstats format.
      output_path (str): The directory where the generated graphs will be saved.

    Returns:
     Void. The call graphs are created and saved in the `output_path` directory

    Note:
    - This function requires Graphviz to be installed and configured separately.
    - Ensure that the 'dot' executable from Graphviz is in the system's PATH.

    """

    print("Creating call graphs")
    formats = ["svg", "pdf"]  # Formats in which the call graph / digraph is saved

    for format in formats:
        di_graphs = os.path.join(output_path, f"profiling/{func.__name__}.{format}")

        try:
            # Use 'poetry run' to execute gprof2dot within the Poetry environment
            command = f"poetry run gprof2dot {profiling_path_pstats} | dot -T{format} -o {di_graphs}"
            subprocess.run(command, shell=True, check=True)
            print(f"Call graph generated and saved to {di_graphs}")
        except subprocess.CalledProcessError as e:
            print(f"Error generating call graph: {e}")


def visualize_profiling_data(profiling_path_prof, PORT=8998):
    """
    Visualize cProfile profiling data using SnakeViz.

    Args:
        profiling_path_prof (str): The path to the cProfile profiling data in `.prof` format.
        PORT (int): The port at which SnakeViz server runs (default is 8998), can be configured to any open port.

    Returns:
        Void. The SnakeViz starts running at the `PORT` showing the icicle and sunburst plots

    SnakeViz:
    - SnakeViz is a web-based interactive viewer for cProfile or profile (Python built-in module) output.

    Profiling (.prof) File:
    - Profiling data generated by cProfile, a built-in Python profiling module.
    - Columns typically include 'ncalls', 'tottime', 'percall', 'cumtime', 'percall', 'filename:lineno(function)'.

    Visualization Plots:
    - SnakeViz can generate visualizations like icicle and sunburst plots for profiling data.

    Exiting the SnakeViz Server:
    - To exit the SnakeViz server, a keyboard interrupt is needed.
    - On Windows, press Ctrl + C or Ctrl + Break
    - On Linux and macOS, press Control + C.

    """

    try:
        command = f"poetry run snakeviz -p {PORT} {profiling_path_prof}"
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating visualizations from profiling data: {e}")
