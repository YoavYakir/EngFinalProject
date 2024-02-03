import multiprocessing
import time
import psutil
import random

def monte_carlo_pi(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples

def measure_resource_utilization(process_id, interval=1):
    cpus = []
    mem = []

    func_process = psutil.Process(process_id)

    while True:
        # if not len(cpus)%10:
        # print("Still running monitoring")
        try:
            cpu_percent = (func_process.cpu_percent(interval=interval))/4
            memory_info = func_process.memory_percent()
            cpus.append(cpu_percent)
            mem.append(memory_info) 
        except:
            print(f'avg cpu {sum(cpus)/len(cpus)}, 1 core avg - {(sum(cpus)/len(cpus))*4} avg mem {sum(mem)/len(mem)}')
            break
        

        # print(f"Function CPU Utilization: {cpu_percent/4}%, Memory Usage: {memory_info}%")
        #time.sleep(interval)

if __name__ == "__main__":
    startTime = time.time()

    # Create a multiprocessing.Process for your function
    func_process = multiprocessing.Process(target=monte_carlo_pi, args=(10**8,))
    func_process.start()

    # Get the process ID of the function
    process_id = func_process.pid
    print(f'PID - {process_id}')
    # Create a multiprocessing.Process for monitoring
    resource_process = multiprocessing.Process(target=measure_resource_utilization, args=(process_id,))
    resource_process.start()

    # Wait for both processes to finish
    func_process.join()
    resource_process.join()

    print(f'It took {time.time() - startTime} seconds.')