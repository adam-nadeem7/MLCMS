import json

json_file = 'task5_3.json'
with open(json_file, 'r') as file:
    data = json.load(file)

paths = []

size = data['size']
#making the grid 
rows = size['x']
cols = size['y']
pedestrians = data['pedestrians']
for pedestrian in pedestrians:
    start = [ pedestrian['x'] , pedestrian['y']]
    #start is the starting point cordinates
    #curr is the current point cordinates
    curr = start
    curr_val = 0
    obs_val = rows + cols #value of obstacle 

    targets = data['target']
    target = [targets[0]['x'] , targets[0]['y']  ]
    #queue to store the current point and the neighbours
    queue = []
    queue.append(curr)


    #initialziing grid with positive infiintiy
    grid = [[int(0) for _ in range(cols)] for _ in range(rows)]

    #list of all obstacle points
    obstacles = []
    # obstacles = [[0,1] , [3,3] , [1, 1]  ,  [2, 4]]
    for obstacle in data['obstacles']:
        obstacles.append([obstacle['x'] , obstacle['y']])


    #penalizing all obstacles in the grid 
    for obs in obstacles:
        x = obs[0]
        y = obs[1]
        obstacle_penalty = rows + cols #setting grid values of obstacle
        grid[x][y] = obstacle_penalty

    while(  queue):
        curr = queue.pop(0)
        curr_val = grid[curr[0]][curr[1]]
        for i in range(curr[0]-1, curr[0]+ 2):
            for j in range(curr[1]-1 , curr[1]+2):
                if 0<=i< rows and 0<=j<cols: #checking for the boundries of grid
                        if grid[i][j] == 0: #add all yunvisited popoints , obtsacles would be ignored as they have non zero grid values
                            if i== start[0] and j==start[1]: #skip the start points
                                continue
                            grid[i ][j] = round(curr_val + ((((i-curr[0]) **2) + (j-curr[1]) ** 2) ** 0.5) , 3) 
                            queue.append([i , j])              

                    

    #finding shortest path after implementing the grid
    #starting from the target

    path = [[target[0] , target[1]]]
    distance = 0
    curr = target
    while ( curr!=start):
        neighbours = []
        
        for i in range(curr[0]-1, curr[0]+ 2):
            for j in range(curr[1]-1 , curr[1]+2):
                if 0<=i< rows and 0<=j<cols:
                    neighbours.append([ grid[i][j] , i , j])
        
        next_point = min(neighbours)
        curr = [next_point[1] , next_point[2]]

        path.append(curr)

    path = path[::-1] #reversing the path
    paths.append(path)
    distance = grid[target[0]][target[1]]
with open('path.json', 'w') as file:
    path = json.dump(path,file)
