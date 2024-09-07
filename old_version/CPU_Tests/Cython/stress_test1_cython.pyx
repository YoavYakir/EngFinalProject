import numpy as np
import time
import psutil
import os

def stress_test():
    cpus = []
    mem = []
    # Get the current process ID
    pid = os.getpid()

    # Get the process using the process ID
    current_process = psutil.Process(pid)

    for i in range(50):
        cpu_percent = (current_process.cpu_percent(interval=1))/4
        memory_info = current_process.memory_percent()  
        cpus.append(cpu_percent)
        mem.append(memory_info) 
        if not i % 10:
            print("Still running calculations")

        # Create a NumPy array with 10 million elements
        data_array = np.zeros(10 ** 8)

        # Vectorized operation to modify each element
        data_array = np.arange(10 ** 8) * 2

        # Sum the elements of the array
        result = np.sum(data_array)
    print(f'avg cpu {sum(cpus)/len(cpus)}, avg mem {sum(mem)/len(mem)}')



# def measure_resource_utilization(process_id, interval=1):
#     cpus = []
#     mem = []

#     func_process = psutil.Process(process_id)

#     while True:
#         if not len(cpus)%10:
#             print("Still running monitoring")
#         try:
#             cpu_percent = (func_process.cpu_percent(interval=interval))/4
#             memory_info = func_process.memory_percent()
#             cpus.append(cpu_percent)
#             mem.append(memory_info) 
#         except:
#             print(f'avg cpu {sum(cpus)/len(cpus)}, avg mem {sum(mem)/len(mem)}')
#             break
        

#         # print(f"Function CPU Utilization: {cpu_percent/4}%, Memory Usage: {memory_info}%")
#         time.sleep(interval)

# if __name__ == "__main__":
startTime = time.time()

stress_test()

    # Create a multiprocessing.Process for your function
    # func_process = multiprocessing.Process(target=stress_test)
    # func_process.start()

    # # Get the process ID of the function
    # process_id = func_process.pid
    # print(f'PID - {process_id}')
    # Create a multiprocessing.Process for monitoring
    # resource_process = multiprocessing.Process(target=measure_resource_utilization, args=(process_id,))
    # resource_process.start()

    # Wait for both processes to finish
    # func_process.join()
    # resource_process.join()

print(f'It took {time.time() - startTime} seconds.')