import cpu_monitor
import mem_monitor
import json
import multiprocessing
import Code
import util


# Read the JSON configuration file
def read_config():
    with open('config.json', 'r') as file:
        return json.load(file)
    
    
def start_monitoring(cpu_config, mem_config, time_config):
    print(cpu_config)
    
    cpu_monitor.start(cpu_config)
     
    
    # Create a monitoring processes
    cpu_monitor_process = multiprocessing.Process(target=cpu_monitor.start, args=(cpu_config))
    mem_monitor_process = multiprocessing.Process(target=mem_monitor.start, args=(mem_config))
    main_process = multiprocessing.Process(target=Code.run)
    
    main_process_pid = main_process.pid
    

    # Start the monitoring process
    cpu_monitor_process.start()
    print("start 1 done")
    mem_monitor_process.start()
    
    main_process.start()
    util.open_main_process_file(started = True, pid = main_process_pid)

    main_process.join()
    cpu_monitor_process.join()
    mem_monitor_process.join()
    
    
    

if __name__ == "__main__":
    print("moni")
    config = read_config()
    cpu_config = config["cpu_monitoring"]
    mem_config = config["memory_monitoring"]
    time_config = config["time_monitoring"]
    util.open_main_process_file(False)
    start_monitoring(cpu_config, mem_config, time_config)
    print("moni")
    