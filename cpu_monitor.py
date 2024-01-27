import psutil
import time
import matplotlib.pyplot as plt
from numpy import array, append
import json
import util



def monitor_cpu(interval, duration, idle_value=0):
    """
    Monitors the CPU usage.
    
    :param interval: Time in seconds between each CPU usage check.
    :param duration: Total duration in seconds for which to monitor CPU.
    :return: List of CPU usage percentages recorded.
    """  
    
    cpu_usage = array([])

    start_time = time.time()

    
    for _ in range(10): # wait max 10 second for main process to start running
        pid = util.check_main_process_activiation()
        if pid!=-1:
            break
        print("cpu monitor - pid is still -1")
        time.sleep(1)

    if pid == -1:
        # raise Exception("The main process failed to create. CPU monitoring stopped.")
        return []
    
    try:
        process = psutil.Process(pid)
        # while time.time() - start_time < duration:
        while process.is_running() and not process.status() == psutil.STATUS_ZOMBIE:
            # usage = psutil.cpu_percent(interval=interval)
            usage = process.cpu_percent(interval) # check the amount used dedicated by that process in given intervals - sleeps in between
            append(cpu_usage,usage)
            print(f"CPU Usage: {usage}%")
    except psutil.NoSuchProcess as e: # when the inspected process closes.
        print(e)
    except Exception as e: # catching generic different exception
        raise Exception(f"Unkown error has occured: {e}")
        
    return cpu_usage

def plot_cpu_usage(cpu_usage):
    """
    Plots the CPU usage data.
    
    :param cpu_usage: List of CPU usage percentages.
    """   
    plt.plot(cpu_usage)
    plt.title("CPU Usage Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("CPU Usage (%)")
    plt.show()


def start(cpu_config):
     
    # Monitor CPU usage
    usage_data = monitor_cpu(cpu_config['interval'], cpu_config['duration'], cpu_config['idle_value'])

    print("-------------------------------- CPU STATISTICS: --------------------------------")
    peak, average, minimum = util.calculate_statistics(usage_data)
    print(f"peak: {peak}, average: {average}, minimum: {minimum}")

    # Plot the usage data
    plot_cpu_usage(usage_data)

