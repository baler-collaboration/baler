import io
import pstats
import cProfile
from pstats import SortKey
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import codecarbon


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


def energy_profiling(f, project_name, measure_power_secs, *args, **kwargs):
    """
    Energy Profiling measures the amount of electricity that
    was consumed by the given function f and the amount of CO2 emission.
    It utilizes the codecarbon package for tracking this information.

    Args:
        f (callable): The function to be profiled.
        project_name (str): The name of the project.
        measure_power_secs (int): The number of seconds to measure power.

    Returns:
        result: The result of the function `f` execution.
    """

    tracker = codecarbon.EmissionsTracker(
        project_name=project_name, measure_power_secs=measure_power_secs
    )
    tracker.start_task(f"{f.__name__}")

    # Execute the function and get its result
    result = f(*args, **kwargs)

    emissions = tracker.stop_task()
    print("CO2 emission [kg]: ", emissions.emissions)
    print("CO2 emission rate [kg/h]: ", 3600 * emissions.emissions_rate)
    print("CPU energy consumed [kWh]: ", emissions.cpu_energy)
    print("GPU energy consumed [kWh]: ", emissions.gpu_energy)
    print("RAM energy consumed [kWh]: ", emissions.ram_energy)

    return result


def c_profile(func, *args, **kwargs):
    """
    Profile the function func with cProfile.

    Args:
        func (callable): The function to be profiled.

    Returns:
        result: The result of the function `func` execution.
    """

    pr = cProfile.Profile()
    pr.enable()
    # Execute the function and get its result
    result = func(*args, **kwargs)
    pr.disable()

    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

    return result
