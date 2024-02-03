import multiprocessing
import time
import psutil
import random
import chaospy as cp
import numpy as np

# Define a simple model
def model(x, y):
    return np.sin(x) + np.cos(y)

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

    # Define the distribution for the inputs
    distribution = cp.J(cp.Normal(0, 1), cp.Uniform(-1, 1))

    # Generate samples
    samples = distribution.sample(10**8)

    # Evaluate the model with the samples
    x_samples, y_samples = samples
    evaluations = model(x_samples, y_samples)

    # Perform analysis (e.g., calculate the mean and variance of the output)
    expected_value = np.mean(evaluations)
    variance = np.var(evaluations)

    print(f"Expected Value: {expected_value}, Variance: {variance}")

    # Create a multiprocessing.Process for your function
    # func_process = multiprocessing.Process(target=monte_carlo_pi, args=(10**8,))
    # func_process.start()

    # Get the process ID of the function
    # process_id = func_process.pid
    # print(f'PID - {process_id}')
    # # Create a multiprocessing.Process for monitoring
    # resource_process = multiprocessing.Process(target=measure_resource_utilization, args=(process_id,))
    # resource_process.start()

    # # Wait for both processes to finish
    # func_process.join()
    # resource_process.join()

    print(f'It took {time.time() - startTime} seconds.')