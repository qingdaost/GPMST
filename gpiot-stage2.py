import operator
import math
import random
import pickle
import numpy
import array

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap import cma

file = open('stage2-IV10.txt', mode='w')
sourcefile = open("gptrees.txt", 'r')
source_data = sourcefile.readlines()
targetfile = open("seed-IV10.txt", 'r')
target_data = targetfile.readlines()
trees = [source_data[22], source_data[5], source_data[25], source_data[31], source_data[38], source_data[43], target_data[264]]
#
source_num = len(trees) - 1

import Taskqueue
iots = 10
import Edgenetwork
#Decide servers number
fogs = 9
clouds = 7

#IoT energy consumption
pidel = 0.3
pbusy = 0.9
ptran = 1.3
tweight = 0.5
eweight = 0.5
iotcapacity = 500

runs = 50
gpruns = 30

def schedule(servers, iots, t, s):
    #time
    gt = t[1]
    if s == -1:
        pt = t[3] / iotcapacity
        start = max(iots[t[0]][-1], gt)
        ft = start + pt
        et = ft - gt
        iots[t[0]].append(ft)
    else:
        ot = servers[s][0]
        dt = t[2] / servers[s][1]
        pt = t[3] / servers[s][2]
        #sent = max(iots[t[0]][0], gt) consider trans queue
        arrive = gt + ot + dt
        start = max(servers[s][3][-1], arrive)
        ft = start + pt
        et = ft - gt
        #iots[t[0]][0] = arrive
        servers[s][3].append(ft)
    #energy
    if s == -1:
        pe = pt * pbusy
        if gt > iots[t[0]][-1]:
            ie = (gt - iots[t[0]][-1]) * pidel
            ec = ie + pe
        else:
            ec = pe
    else:
        ec = dt * ptran
    return et, ec

def getidle(servers, task):
    count = 0
    for s in servers:
        if task[1] > s[3][-1]:
            count += 1
    return count

def getwaits(servers, iot, s, task):
    count = 0
    if s == -1:
        for t in iot:
            if t > task[1]:
                count += 1
    else:
        for t in servers[s][3][::-1]:
            if t > task[1]:
                count += 1
            else:
                break
    return count

def evaluatetree(individual):
    func = toolbox.compile(expr=individual)
    totalresult = 0
    for run in range(runs):
        servers = Edgenetwork.generateservers(fogs, clouds)
        iotservers = Edgenetwork.generateiotservers(iots)
        task_list = Taskqueue.generatetasks(run, iots)
        oet = 0
        oec = 0
        last_time = 0
        for t in task_list:
            f_list = []
            for s in range(-1, fogs + clouds):
                gt = t[1]
                if s == -1:
                    tt = 0
                    pt = t[3] / iotcapacity
                    start = max(iotservers[t[0]][-1], gt)
                    wt = start - gt
                else:
                    ot = servers[s][0]
                    dt = t[2] / servers[s][1]
                    tt = ot + dt
                    pt = t[3] / servers[s][2]
                    arrive = gt + ot + dt
                    start = max(servers[s][3][-1], arrive)
                    wt = start - gt
                # energy
                if s == -1:
                    pe = pt * pbusy
                    te = 0
                else:
                    pe = 0
                    te = dt * ptran
                it = t[1] - last_time
                last_time = it
                ns = getidle(servers, t)
                na = getwaits(servers, iotservers[t[0]], s, t)
                f_value = func(tt, wt, pt, it, te, pe, ns, na)
                f_list.append(f_value)
            id = numpy.argmin(f_list)
            #print(t, "is served by server", id)
            et, ec = schedule(servers, iotservers, t, id - 1)
            oet += et
            oec += ec
        result = tweight * oet + eweight * oec
        totalresult += result
    #print("One evaluation")
    return totalresult / runs

# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

pset = gp.PrimitiveSet("MAIN", 8)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
#pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(max, 2)
pset.addPrimitive(min, 2)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset.addEphemeralConstant("rand100", lambda: random.random() * 100)
#pset.renameArguments(ARG0='dt', ARG1='wt', ARG2='pt', ARG3='it', ARG4='de', ARG5='pe', ARG6='ns', ARG7='na')
pset.renameArguments(ARG0='tt', ARG1='wt', ARG2='pt', ARG3='it', ARG4='te', ARG5='pe', ARG6='ns', ARG7='na')
#creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
#toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
#toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

#def evaluate(individual, task_list):
def evaluate(individual):
    reward = weighting(individual, trees)
    return reward,

def testfitness(weights, trees):
    weights = softmax(weights)
    functions = []
    for tree in trees:
        functions.append(toolbox.compile(expr=tree))
    totalresult = 0
    for instance in range(runs):
        run = instance + runs
        servers = Edgenetwork.generateservers(fogs, clouds)
        iotservers = Edgenetwork.generateiotservers(iots)
        task_list = Taskqueue.generatetasks(run, iots)
        oet = 0
        oec = 0
        last_time = 0
        for t in task_list:
            f_list = []
            for s in range(-1, fogs + clouds):
                gt = t[1]
                if s == -1:
                    tt = 0
                    pt = t[3] / iotcapacity
                    start = max(iotservers[t[0]][-1], gt)
                    wt = start - gt
                else:
                    ot = servers[s][0]
                    dt = t[2] / servers[s][1]
                    tt = ot + dt
                    pt = t[3] / servers[s][2]
                    arrive = gt + ot + dt
                    start = max(servers[s][3][-1], arrive)
                    wt = start - gt
                # energy
                if s == -1:
                    pe = pt * pbusy
                    te = 0
                else:
                    pe = 0
                    te = dt * ptran
                it = t[1] - last_time
                last_time = it
                ns = getidle(servers, t)
                na = getwaits(servers, iotservers[t[0]], s, t)
                f_value = 0
                for i in range(source_num + 1):
                    f_value += functions[i](tt, wt, pt, it, te, pe, ns, na) * weights[i]
                f_list.append(f_value)
            id = numpy.argmin(f_list)
            # print(t, "is served by server", id)
            et, ec = schedule(servers, iotservers, t, id - 1)
            oet += et
            oec += ec
        result = tweight * oet + eweight * oec
        totalresult += result
    # print("One evaluation")
    return totalresult / runs

def weighting(weights, trees):
    weights = softmax(weights)
    functions = []
    for tree in trees:
        functions.append(toolbox.compile(expr=tree))
    totalresult = 0
    for run in range(runs):
        servers = Edgenetwork.generateservers(fogs, clouds)
        iotservers = Edgenetwork.generateiotservers(iots)
        task_list = Taskqueue.generatetasks(run, iots)
        oet = 0
        oec = 0
        last_time = 0
        for t in task_list:
            f_list = []
            for s in range(-1, fogs + clouds):
                gt = t[1]
                if s == -1:
                    tt = 0
                    pt = t[3] / iotcapacity
                    start = max(iotservers[t[0]][-1], gt)
                    wt = start - gt
                else:
                    ot = servers[s][0]
                    dt = t[2] / servers[s][1]
                    tt = ot + dt
                    pt = t[3] / servers[s][2]
                    arrive = gt + ot + dt
                    start = max(servers[s][3][-1], arrive)
                    wt = start - gt
                # energy
                if s == -1:
                    pe = pt * pbusy
                    te = 0
                else:
                    pe = 0
                    te = dt * ptran
                it = t[1] - last_time
                last_time = it
                ns = getidle(servers, t)
                na = getwaits(servers, iotservers[t[0]], s, t)
                f_value = 0
                for i in range(source_num + 1):
                    f_value += functions[i](tt, wt, pt, it, te, pe, ns, na) * weights[i]
                f_list.append(f_value)
            id = numpy.argmin(f_list)
            et, ec = schedule(servers, iotservers, t, id - 1)
            oet += et
            oec += ec
        result = tweight * oet + eweight * oec
        totalresult += result
    return totalresult / runs

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = numpy.exp(x - numpy.max(x))  # subtract max to stabilize
    return e_x / e_x.sum(axis=0)  # only difference

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox.register("evaluate", evaluate)

# hyperparameters
N = source_num + 1

# Init
treefitness = []
for tree in trees:
    treefitness.append(evaluatetree(tree))

best = min(treefitness)
bestindex = treefitness.index(best)
w = [-10.0] * N
w[bestindex] = 10.0

def main():

    for gprun in range(gpruns):
        file.write("RUN" + str(gprun) + '\n')
        random.seed(gprun)

        lambda_ = 4 + math.floor(3 * math.log(N))
        strategy = cma.Strategy(centroid=w, sigma=3.0, lambda_=lambda_)
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)

        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)

        # The CMA-ES algorithm converge with good probability with those settings
        algorithms.eaGenerateUpdate(file, toolbox, ngen=50, stats=stats, halloffame=hof)

        file.write(str(testfitness(hof[0], trees)) + '\n')


if __name__ == "__main__":
    main()


