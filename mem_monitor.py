import psutil
import time
import util




def monitor_memory(mem_config):
    
    for _ in range(10): # wait max 10 second for main process to start running
        pid = util.check_main_process_activiation()
        if pid!=-1:
            break
        print("mem monitor - pid is still -1")
        time.sleep(1)

    if pid == -1:
        raise Exception("The main process failed to create. MEM monitoring stopped.")
    
    memory_usage_history = []
    try:
        process = psutil.Process(pid)
        while True:
            memory_info = process.memory_info()
            memory_usage_history.append(memory_info.rss)  #rss is the Resident Set Size, in bytes

            time.sleep(1)  # Adjust the sleep time as needed
    except psutil.NoSuchProcess:
        pass  # Handle the case where the process ends
    except Exception as e: # catching generic different exception
        raise Exception(f"Unkown error has occured: {e}")

    return memory_usage_history




def start(mem_config):
    
    memory_usage_history = monitor_memory(mem_config)
    peak, average, minimum = util.calculate_statistics(memory_usage_history)
    print(f"Peak Memory Usage: {peak} bytes")
    print(f"Average Memory Usage: {average} bytes")
    print(f"Minimum Memory Usage: {minimum} bytes")
    