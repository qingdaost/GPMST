import random
import math

_lambda = 1

def generatetasks(seed, iots):
    random.seed(seed)
    arrival_time = 0
    task_list = []
    for i in range(1000):
        origin = random.randint(0, iots - 1)
        datasize = random.randint(200, 20000)
        workload = random.randint(5000, 10000)
        task = [origin, arrival_time, datasize, workload]
        task_list.append(task)
        p = random.random()
        inter_arrival_time = round(-math.log(1.0 - p)/_lambda, 2)
        arrival_time += inter_arrival_time
        #print(task)
    return task_list
