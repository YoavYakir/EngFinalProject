import cython
import multiprocessing
import time
import psutil

def prime():
    nb_primes = 40000
    p = []
    n = 2
    while len(p) < nb_primes:
        # Is n prime?
        for i in p:
            if n % i == 0:
                break

        # If no break occurred in the loop
        else:
            p.append(n)
        n += 1
    return p


def measure_resource_utilization(process_id, interval=1):
    cpus = []
    mem = []

    func_process = psutil.Process(process_id)

    while True:
        if not len(cpus)%10:
            print("Still running monitoring")
        try:
            cpu_percent = (func_process.cpu_percent(interval=interval))/4
            memory_info = func_process.memory_percent()
            cpus.append(cpu_percent)
            mem.append(memory_info) 
        except:
            print(f'avg cpu {sum(cpus)/len(cpus)}, avg mem {sum(mem)/len(mem)}')
            break
        

        # print(f"Function CPU Utilization: {cpu_percent/4}%, Memory Usage: {memory_info}%")
        time.sleep(interval)

startTime = time.time()
prime()
# # Create a multiprocessing.Process for your function
# func_process = multiprocessing.Process(target=prime,)
# func_process.start()

# # Get the process ID of the function
# process_id = func_process.pid
# print(f'PID - {process_id}')
# # Create a multiprocessing.Process for monitoring
# resource_process = multiprocessing.Process(target=measure_resource_utilization, args=(process_id,))
# resource_process.start()

# # Wait for both processes to finish
# func_process.join()
# resource_process.join()

print(f'It took {time.time() - startTime} seconds.')