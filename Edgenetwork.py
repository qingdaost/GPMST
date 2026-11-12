import random
import copy

def generateservers(fogs, clouds):
    random.seed(9)
    servers = []
    fog = [[0.0005, 2000, 12000], [0.0005, 4000, 9000]]
    cloud = [[0.03, 500, 36000], [0.03, 1000, 30000]]
    for i in range(fogs):
        server = copy.deepcopy(random.choice(fog))
        servers.append(server)
    for i in range(clouds):
        server = copy.deepcopy(random.choice(cloud))
        servers.append(server)
    for server in servers:
        server.append([0])
    return servers


def generateiotservers(iots):
    iotservers = []
    for i in range(iots):
        iotservers.append([0])
    return iotservers
