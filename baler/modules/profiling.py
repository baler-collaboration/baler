import io
import os
import pstats
import cProfile
from pstats import SortKey
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import codecarbon
from ..modules import profile_plotting


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
        project_name (str): The name of the project.
        measure_power_secs (int): The number of seconds to measure power.

    Returns:
        result: The result of the function `f` execution.
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
