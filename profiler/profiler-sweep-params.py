import datetime
import subprocess
import psutil
import time
import os
import signal
import traceback
from multiprocessing import Process, Queue
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for script-based plotting
import matplotlib.pyplot as plt
import json
from itertools import product
import logging
import pandas as pd
import queue as QueueModule

def setup_logger(log_file):
    logger = logging.getLogger('MonitorLogger')
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def get_child_processes(pid):
    """Get a list of child processes for a given PID."""
    try:
        process = psutil.Process(pid)
        return process.children(recursive=True)
    except psutil.NoSuchProcess:
        return []

def is_process_ready(stdout_path, logger, max_wait_time=60, check_interval=2):
    """Check if the monitored process is ready by verifying if it contains the readiness indicator in stdout.txt."""
    if not os.path.exists(stdout_path):
        logger.error(f"stdout file not found at {stdout_path}")
        return False

    elapsed_time = 0
    while elapsed_time < max_wait_time:
        try:
            with open(stdout_path, 'r') as file:
                content = file.read()
                if "------ITER " in content:
                    logger.info("Monitored process is ready.")
                    return True
                else:
                    logger.debug(f"Readiness check failed: '------ITER ' not found in {stdout_path}")
        except Exception as e:
            logger.error(f"Error reading stdout file {stdout_path}: {e}\n{traceback.format_exc()}")
        
        time.sleep(check_interval)
        elapsed_time += check_interval
    
    logger.error("Monitored process did not become ready in time.")
    return False

def monitor_swap_and_interrupt(parent_pid, threshold, queue, logger):
    logger.info("Starting swap monitoring.")
    try:
        while True:
            child_processes = get_child_processes(parent_pid)
            if not child_processes:
                break

            try:
                swap = psutil.swap_memory()
                swap_usage_percent = swap.percent

                queue.put(('swap', swap_usage_percent))
                logger.debug(f"Swap usage: {swap_usage_percent}%")

                if swap_usage_percent > threshold:
                    logger.warning(f"Swap usage exceeded {threshold}%, terminating process group...")

                    # Log child processes before terminating
                    logger.debug(f"Child processes of PID {parent_pid}: {[p.pid for p in child_processes]}")

                    os.killpg(os.getpgid(parent_pid), signal.SIGTERM)  # Terminate the process group
                    time.sleep(5)  # Wait a moment to ensure the signal is processed

                    # Check if process and its children are terminated
                    if any(p.is_running() for p in child_processes):
                        logger.error("Failed to terminate process group, retrying...")
                        os.killpg(os.getpgid(parent_pid), signal.SIGKILL)  # Force kill the process group

                        # Log child processes after force kill attempt
                        child_processes = get_child_processes(parent_pid)
                        logger.debug(f"Child processes after force kill attempt: {[p.pid for p in child_processes]}")
                    else:
                        logger.debug("Process terminated successfully.")
                    break

                time.sleep(1)  # Sleep for a short interval before checking again
            except Exception as e:
                logger.error(f"Error in swap monitor: {e}\n{traceback.format_exc()}")
    finally:
        logger.info("Swap monitoring ended.")

def monitor_cpu_memory(parent_pid, queue, logger):
    logger.info("Starting CPU/memory monitoring.")
    try:
        while True:
            child_processes = get_child_processes(parent_pid)
            if not child_processes:
                logger.info("No child processes found.")
                break

            try:
                total_cpu_percent = 0.0
                total_memory_usage_percent = 0.0
                num_children = len(child_processes)

                for child in child_processes:
                    try:
                        total_cpu_percent += child.cpu_percent(interval=1)
                        memory_info = child.memory_info()
                        total_memory_usage_percent += (memory_info.rss / psutil.virtual_memory().total) * 100

                    except psutil.NoSuchProcess:
                        logger.info(f"Process no longer exists (pid={child.pid})")
                        continue  # Skip this process

                average_cpu_percent = total_cpu_percent / num_children if num_children > 0 else 0
                average_memory_usage_percent = total_memory_usage_percent / num_children if num_children > 0 else 0

                queue.put(('cpu', average_cpu_percent))
                queue.put(('memory', average_memory_usage_percent))
                logger.debug(f"Average CPU usage: {average_cpu_percent}%, Average Memory usage: {average_memory_usage_percent}%")

                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in CPU/memory monitor: {e}\n{traceback.format_exc()}")
    finally:
        logger.info("CPU/memory monitoring ended.")
        queue.put(('terminate', None))  # Signal to terminate the plot process

def monitor_gpu_usage(folder, logger):
    logger.info("Starting GPU monitoring.")
    gpu_csv_path = os.path.join(folder, 'gpu.csv')
    gpu_command = f"nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory --format=csv -l 1 -f {gpu_csv_path}"
    gpu_process = subprocess.Popen(gpu_command, shell=True, preexec_fn=os.setsid)
    logger.info(f"Started GPU monitoring with command: {gpu_command}")
    return gpu_process, gpu_csv_path

def plot_usage(queue, folder, logger, plot_interval=60):
    logger.info("Starting plot generation.")
    cpu_data = []
    memory_data = []
    swap_data = []
    gpu_data = None

    gpu_csv_path = os.path.join(folder, 'gpu.csv')
    cpu_csv_path = os.path.join(folder, 'cpu_usage.csv')
    memory_csv_path = os.path.join(folder, 'memory_usage.csv')
    swap_csv_path = os.path.join(folder, 'swap_usage.csv')

    last_plot_time = time.time()

    try:
        while True:
            try:
                data_type, data_value = queue.get_nowait()

                if data_type == 'terminate':
                    break

                current_time = datetime.datetime.now().strftime('%H:%M:%S')

                if data_type == 'cpu':
                    cpu_data.append((current_time, data_value))
                    pd.DataFrame(cpu_data, columns=['Time', 'CPU Usage (%)']).to_csv(cpu_csv_path, index=False)
                elif data_type == 'memory':
                    memory_data.append((current_time, data_value))
                    pd.DataFrame(memory_data, columns=['Time', 'Memory Usage (%)']).to_csv(memory_csv_path, index=False)
                elif data_type == 'swap':
                    swap_data.append((current_time, data_value))
                    pd.DataFrame(swap_data, columns=['Time', 'Swap Usage (%)']).to_csv(swap_csv_path, index=False)

                # Plot at intervals
                if time.time() - last_plot_time >= plot_interval:
                    last_plot_time = time.time()

                    fig, axs = plt.subplots(figsize=(10, 5))
                    if cpu_data:
                        times, values = zip(*cpu_data)
                        axs.plot(times, values)
                        axs.set_title('Average CPU Usage (%)')
                        axs.set_ylim([0, 100])
                        axs.set_xlabel('Time')
                        axs.set_ylabel('CPU Usage (%)')
                        axs.xaxis.set_tick_params(rotation=45)
                        plt.tight_layout()  # Adjust layout to prevent clipping
                        fig.savefig(os.path.join(folder, 'cpu_usage.png'))
                        plt.close(fig)

                    fig, axs = plt.subplots(figsize=(10, 5))
                    if memory_data:
                        times, values = zip(*memory_data)
                        axs.plot(times, values)
                        axs.set_title('Average Memory Usage (%)')
                        axs.set_ylim([0, 100])
                        axs.set_xlabel('Time')
                        axs.set_ylabel('Memory Usage (%)')
                        axs.xaxis.set_tick_params(rotation=45)
                        plt.tight_layout()  # Adjust layout to prevent clipping
                        fig.savefig(os.path.join(folder, 'memory_usage.png'))
                        plt.close(fig)

                    fig, axs = plt.subplots(figsize=(10, 5))
                    if swap_data:
                        times, values = zip(*swap_data)
                        axs.plot(times, values)
                        axs.set_title('Swap Usage (%)')
                        axs.set_ylim([0, 100])
                        axs.set_xlabel('Time')
                        axs.set_ylabel('Swap Usage (%)')
                        axs.xaxis.set_tick_params(rotation=45)
                        axs.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))  # Less frequent ticks
                        plt.tight_layout()  # Adjust layout to prevent clipping
                        fig.savefig(os.path.join(folder, 'swap_usage.png'))
                        plt.close(fig)

                    if os.path.exists(gpu_csv_path):
                        if os.path.getsize(gpu_csv_path) > 0:
                            try:
                                gpu_data = pd.read_csv(gpu_csv_path)
                                gpu_data.columns = [col.strip() for col in gpu_data.columns]  # Strip any extra spaces
                                fig, axs = plt.subplots(figsize=(10, 5))
                                axs.plot(pd.to_datetime(gpu_data['timestamp']), gpu_data['utilization.gpu [%]'].str.rstrip(' %').astype(float), label='GPU Utilization (%)')
                                axs.set_title('GPU and Memory Utilization')
                                axs.set_xlabel('Time')
                                axs.set_ylabel('Utilization (%)')
                                axs.legend()
                                axs.xaxis.set_tick_params(rotation=45)
                                plt.tight_layout()  # Adjust layout to prevent clipping
                                fig.savefig(os.path.join(folder, 'gpu_usage.png'))
                                plt.close(fig)
                            except ValueError as ve:
                                if 'time data' in str(ve):
                                    logger.debug(f"Skipping GPU plot due to time data parsing error: {ve}")
                                else:
                                    logger.error(f"Error in plot usage: {ve}\n{traceback.format_exc()}")
                        else:
                            logger.warning(f"GPU CSV file is empty: {gpu_csv_path}")

            except QueueModule.Empty:
                time.sleep(1)
    finally:
        logger.info("Plot generation ended.")

def launch_and_monitor(threshold, folder, logger):
    stdout_path = os.path.join(folder, 'stdout.txt')
    stderr_path = os.path.join(folder, 'stderr.txt')
    command = ["/home/wsl/repo/alphazero-general/venv/bin/python3", "-m", "alphazero.envs.gobang.train"]

    with open(stdout_path, 'w') as stdout_file, open(stderr_path, 'w') as stderr_file:
        monitored_process = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file, preexec_fn=os.setsid)
    
    parent_pid = monitored_process.pid
    logger.info(f"Monitored process PID: {parent_pid}")
    
    if not is_process_ready(stdout_path, logger):
        logger.error("Monitored process did not become ready, aborting monitoring.")
        os.killpg(os.getpgid(parent_pid), signal.SIGTERM)
        return

    # Start monitoring processes
    queue = Queue()
    swap_monitor = Process(target=monitor_swap_and_interrupt, args=(parent_pid, threshold, queue, logger))
    cpu_memory_monitor = Process(target=monitor_cpu_memory, args=(parent_pid, queue, logger))
    plot_process = Process(target=plot_usage, args=(queue, folder, logger))
    gpu_process, gpu_csv_path = monitor_gpu_usage(folder, logger)
    
    swap_monitor.start()
    cpu_memory_monitor.start()
    plot_process.start()
    logger.debug(f"Swap monitor PID: {swap_monitor.pid}")
    logger.debug(f"CPU/memory monitor PID: {cpu_memory_monitor.pid}")
    logger.debug(f"Plot process PID: {plot_process.pid}")
    logger.debug(f"GPU monitoring PID: {gpu_process.pid}")

    # Wait for child processes to complete
    while True:
        child_processes = get_child_processes(parent_pid)
        if not child_processes:
            break
        time.sleep(1)
    
    # Cleanup
    swap_monitor.join()
    cpu_memory_monitor.join()
    queue.put(('terminate', None))  # Signal the plot process to terminate
    plot_process.join()
    
    os.killpg(os.getpgid(gpu_process.pid), signal.SIGTERM)
    gpu_process.wait()

    logger.info("Monitoring process ended.")

def generate_combinations(args):
    keys, values = zip(*args.items())
    combinations = [dict(zip(keys, combination)) for combination in product(*values)]
    return combinations

def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'profiler/monitor_{timestamp}.log'
    logger = setup_logger(log_file)

    with open('profiler/args-sweep.json', 'r') as f:
        args = json.load(f)

    with open('profiler/args-initial.json', 'r') as f:
        args_initial = json.load(f)

    combinations = generate_combinations(args)
    swap_threshold = 30  # Set the swap usage threshold (N percent)

    for i, combo in enumerate(combinations):
        combo_run_args = dict(args_initial)
        combo_run_args.update(combo)
        combo_folder = f"profiler/output/run_{i}"
        if not os.path.exists(combo_folder):
            os.makedirs(combo_folder)

        combo_run_args["run_name"] += combo_folder
        combo_run_args["past_data_run_name"] += combo_folder

        # this is for the record
        combo_args_file = os.path.join(combo_folder, 'args.json')
        with open(combo_args_file, 'w') as f:
            json.dump(combo_run_args, f, indent=4)
            
        # this is used by the train script
        with open("profiler/args.json", 'w') as f:
            json.dump(combo_run_args, f, indent=4)

        logger.info(f"Running combination {i}: {combo}")
        launch_and_monitor(swap_threshold, combo_folder, logger)

if __name__ == "__main__":
    main()
