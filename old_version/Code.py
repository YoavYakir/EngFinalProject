import tensorflow
import psutil
from ai_benchmark import AIBenchmark
from util import timing_decorator

# Function to print CPU and memory usage
def print_system_usage():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    print(f"CPU Usage: {cpu_percent}%")
    print(f"Memory Usage: {memory.percent}%")

def cpu_heavy_func():
    i=0
    for j in range(400000):
        i+=1


# @timing_decorator
def run():
    print("START!")
    cpu_heavy_func()
    print("END!")
    # print_system_usage()
    # benchmark = AIBenchmark(use_CPU= True)
    # results = benchmark.run()
    exit(1)


