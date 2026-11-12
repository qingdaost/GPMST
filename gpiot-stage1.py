import operator
import random
import pickle
import numpy

from deap import base
from deap import creator
from deap import tools
from deap import gp

import evalGP_main as evalGP

tweight = 0.5
eweight = 0.5

import Taskqueue
iots = 10
iotcapacity = 500

import Edgenetwork
fogs = 9
clouds = 7
filename = 'seed-IV10.txt'

#IoT energy consumption
pidel = 0.3
pbusy = 0.9
ptran = 1.3

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
        arrive = gt + ot + dt
        start = max(servers[s][3][-1], arrive)
        ft = start + pt
        et = ft - gt
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
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evaluate(individual):
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
            et, ec = schedule(servers, iotservers, t, id - 1)
            oet += et
            oec += ec
        result = tweight * oet + eweight * oec
        totalresult += result
    return totalresult / runs,

toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("selectWorst", tools.selWorst)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))

def main():
    file = open(filename, mode='w')
    for gprun in range(gpruns):
        file.write("RUN" + str(gprun) + '\n')
        random.seed(gprun)
        pop = toolbox.population(n=506)
        sourcefile1 = open("d4f6c3best.pickle", 'rb')
        sourcepop1 = pickle.load(sourcefile1)
        sourcefile2 = open("d5f6c4best.pickle", 'rb')
        sourcepop2 = pickle.load(sourcefile2)
        sourcefile3 = open("d6f6c5best.pickle", 'rb')
        sourcepop3 = pickle.load(sourcefile3)
        sourcefile4 = open("d7f7c3best.pickle", 'rb')
        sourcepop4 = pickle.load(sourcefile4)
        sourcefile5 = open("d8f7c4best.pickle", 'rb')
        sourcepop5 = pickle.load(sourcefile5)
        sourcefile6 = open("d9f7c5best.pickle", 'rb')
        sourcepop6 = pickle.load(sourcefile6)
        pop1 = pop + sourcepop1 + sourcepop2 + sourcepop3 + sourcepop4 + sourcepop5 + sourcepop6
        hof = tools.HallOfFame(1)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", numpy.mean)
        mstats.register("std", numpy.std)
        mstats.register("min", numpy.min)
        mstats.register("max", numpy.max)

        pop1, log = evalGP.eaSimple(file, pop1, toolbox, 0.8, 0.15, 0.0195, 50, stats=mstats,
                                   halloffame=hof, verbose=True)


    return pop1, log, hof


if __name__ == "__main__":
    main()