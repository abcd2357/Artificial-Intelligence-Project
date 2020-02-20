# CSCI561 HW 1
# XU KANGYAN
# 2020.2.12

# import time
# start = time.time()

f = open("input.txt", 'r')
lines = f.readlines()
f_out = open("output.txt", 'w')

algorithm = lines[0].strip()
width, height = map(int, lines[1].split())
Initial = list(map(int, lines[2].split()))
Target = list(map(int, lines[3].split()))
channel_num = int(lines[4])
channel = []
for i in range(0, channel_num):
    channel.append(list(map(int, lines[i+5].split())))
# Bi-directionality
for i in range(0, channel_num):
    channel.append([channel[i][3], channel[i][1], channel[i][2], channel[i][0]])

def BFS():

    # Graph Search -- Breadth First Search
    frontier = list()
    frontier.append(Initial)
    explored_set = list()
    parent = list()
    child = list()
    path = list()
    cost = 1

    while 1:
        if len(frontier) == 0:
            f_out.write("FAIL")
            f.close()
            break

        node = frontier.pop(0)

        if node == Target:
            path.append(node)

            while 1:
                if node == Initial:
                    break
                # lists child & parent are bijective
                node = parent[child.index(node)]
                path.append(node)

            step = len(path)
            total_cost = (step-1) * cost
            # cost & step
            f_out.write(str(total_cost) + '\n' + str(step) + '\n')
            # Initial with cost 0
            f_out.write(str(path[step-1][0])+" "+str(path[step-1][1])+" "+str(path[step-1][2])+" "+str(0)+'\n')
            # Steps with cost 1
            for line in range(step-2, -1, -1):
                f_out.write(str(path[line][0])+" "+str(path[line][1])+" "+str(path[line][2])+" "+str(cost)+'\n')
            f.close()
            break

        explored_set.append(node)

        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                # N NE E SE S SW W NW Move
                if (node[1]+x == 0) or (node[1]+x == width-1) or (node[2]+y == 0) or (node[2]+y == height-1) or (x == y == 0):
                    node_child = node
                else:
                    node_child = [node[0], node[1] + x, node[2] + y]
                if (node_child not in frontier) and (node_child not in explored_set):
                    frontier.append(node_child)
                    child.append(node_child)
                    parent.append(node)
                # Jaunt
                for j in range(0, channel_num):
                    if channel[j][0:3] == node[0:3]:
                        node_child = [channel[j][3], node[1], node[2]]
                        if (node_child not in frontier) and (node_child not in explored_set):
                            frontier.append(node_child)
                            child.append(node_child)
                            parent.append(node)

def UCS():

    # Graph Search -- Uniform Cost Search
    frontier = list()
    # node[3] as pathcost
    Initial.append(0)
    frontier.append(Initial)
    explored_set = list()
    children = list()
    child = list()
    parent = list()
    path = list()
    flag = 0

    while 1:
        if len(frontier) == 0:
            f_out.write("FAIL")
            f.close()
            break

        # Pop the lowest-pathcost node in frontier. CAUTION!
        node = min(frontier, key=lambda x: x[3])
        del frontier[frontier.index(node)]

        if node[0:3] == Target:
            path.append(node)

            while 1:
                if node[0:3] == Initial[0:3]:
                    break
                # lists child & parent are bijective
                node = parent[child.index(node)]
                path.append(node)

            step = len(path)
            total_cost = path[0][3]
            # cost & step
            f_out.write('\n'.join(map(str, [total_cost, step]))+'\n')
            # Initial with cost 0
            f_out.write(" ".join(map(str, [path[step-1][0], path[step-1][1], path[step-1][2], path[step-1][3]]))+'\n')
            for line in range(step-2, -1, -1):
                f_out.write(" ".join(map(str, [path[line][0], path[line][1], path[line][2], path[line][3]-path[line+1][3]]))+'\n')
            f.close()
            break

        explored_set.append(node)

        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                # N NE E SE S SW W NW Move
                if (node[1]+x == 0) or (node[1]+x == width-1) or (node[2]+y == 0) or (node[2]+y == height-1) or (x == y == 0):
                    node_child = node
                else:
                    node_child = [node[0], node[1]+x, node[2]+y, node[3]+6+4*(abs(x)+abs(y))]
                if (node_child not in frontier) and (node_child not in explored_set):
                    children.append(node_child)
                # Jaunt
                for tunnel in channel:
                    if tunnel[0:3] == node[0:3]:
                        cost = abs(tunnel[3] - node[0])
                        node_child = [tunnel[3], node[1], node[2], node[3]+cost]
                        if (node_child not in frontier) and (node_child not in explored_set):
                            children.append(node_child)

        while len(children) > 0:
            node_current = children.pop(0)

            f_length = len(frontier)
            e_length = len(explored_set)

            if (node_current[0:3] not in [frontier[m][0:3] for m in range(0, f_length)]) and (node_current[0:3] not in [explored_set[n][0:3] for n in range(0, e_length)]):
                frontier.append(node_current)
                child.append(node_current)
                parent.append(node)
                continue

            if f_length > 0:
                for m in range(f_length - 1, -1, -1):
                    # pick small pathcost
                    if (node_current[0:3] == frontier[m][0:3]) and (node_current[3] < frontier[m][3]):
                        del frontier[m]
                        flag = 1

            if flag == 1:
                frontier.append(node_current)
                child.append(node_current)
                parent.append(node)
                flag = 0

def ASTAR():

    # Graph Search -- A* Search
    frontier = list()
    # node[3] as estimated total cost
    year_start = abs(Target[0]-Initial[0])
    dx_start = abs(Target[1]-Initial[1])
    dy_start = abs(Target[2]-Initial[2])
    heuristic_start = year_start+min(dx_start, dy_start)*14+abs(dx_start-dy_start)*10
    Initial.append(heuristic_start)
    # node[4] as cost so far
    Initial.append(0)
    frontier.append(Initial)
    explored_set = list()
    children = list()
    child = list()
    parent = list()
    path = list()
    flag = 0

    while 1:
        if len(frontier) == 0:
            f_out.write("FAIL")
            f.close()
            break

        # Pop the lowest-total_cost node in frontier. CAUTION!
        node = min(frontier, key=lambda x: x[3])
        del frontier[frontier.index(node)]

        if node[0:3] == Target:
            path.append(node)

            while 1:
                if node[0:3] == Initial[0:3]:
                    break
                # lists child & parent are bijective
                node = parent[child.index(node)]
                path.append(node)

            step = len(path)
            total_cost = path[0][4]
            # cost & step
            f_out.write('\n'.join(map(str, [total_cost, step]))+'\n')
            # Initial with cost 0
            f_out.write(" ".join(map(str, [path[step-1][0], path[step-1][1], path[step-1][2], path[step-1][4]]))+'\n')
            for line in range(step-2, -1, -1):
                f_out.write(" ".join(map(str, [path[line][0], path[line][1], path[line][2], path[line][4]-path[line+1][4]]))+'\n')
            f.close()
            break

        explored_set.append(node)

        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                # N NE E SE S SW W NW Move
                if (node[1]+x == 0) or (node[1]+x == width-1) or (node[2]+y == 0) or (node[2]+y == height-1) or (x == y == 0):
                    node_child = node
                else:
                    # cost so far to reach node_child
                    g = node[4] + 6 + 4*(abs(x)+abs(y))

                    # estimated cost to goal from node_child
                    year = abs(Target[0]-node[0])
                    dx = abs(Target[1]-node[1]-x)
                    dy = abs(Target[2]-node[2]-y)
                    heuristic = year + min(dx, dy)*14 + abs(dx-dy)*10

                    # estimated total cost of path through node_child to goal
                    func_eval = g + heuristic

                    node_child = [node[0], node[1]+x, node[2]+y, func_eval, g]
                if (node_child not in frontier) and (node_child not in explored_set):
                    children.append(node_child)
                # Jaunt
                for tunnel in channel:
                    if tunnel[0:3] == node[0:3]:
                        # cost so far to reach node_child
                        g = node[4] + abs(tunnel[3] - node[0])

                        # estimated cost to goal from node_child
                        year = abs(Target[0] - tunnel[3])
                        dx = abs(Target[1] - node[1])
                        dy = abs(Target[2] - node[2])
                        heuristic = year + min(dx, dy)*14 + abs(dx-dy)*10

                        # estimated total cost of path through node_child to goal
                        func_eval = g + heuristic

                        node_child = [tunnel[3], node[1], node[2], func_eval, g]
                        if (node_child not in frontier) and (node_child not in explored_set):
                            children.append(node_child)

        while len(children) > 0:
            node_current = children.pop(0)

            f_length = len(frontier)
            e_length = len(explored_set)

            if (node_current[0:3] not in [frontier[m][0:3] for m in range(0, f_length)]) and (node_current[0:3] not in [explored_set[n][0:3] for n in range(0, e_length)]):
                frontier.append(node_current)
                child.append(node_current)
                parent.append(node)
                continue

            if f_length > 0:
                for m in range(f_length-1, -1, -1):
                    # pick small estimated total cost. CAUTION!
                    if (node_current[0:3] == frontier[m][0:3]) and (node_current[3] < frontier[m][3]):
                        del frontier[m]
                        flag = 1

            if flag == 1:
                frontier.append(node_current)
                child.append(node_current)
                parent.append(node)
                flag = 0

if algorithm == 'BFS':
    BFS()
    # print(round((time.time() - start), 2))
elif algorithm == 'UCS':
    UCS()
    # print(round((time.time() - start), 2))
elif algorithm == 'A*':
    ASTAR()
    # print(round((time.time() - start), 2))