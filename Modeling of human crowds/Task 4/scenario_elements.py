import scipy.spatial.distance
from PIL import Image, ImageTk
import numpy as np
import math
import json

class Pedestrian:
    """
    Defines a single pedestrian.
    """

    def __init__(self, position, desired_speed,obstacles,targets):
        self._position = position
        self._desired_speed = desired_speed
        self._obstacles = obstacles
        self._targets = targets
    # task3 :change speed
    def change_speed(self, ratios):
        if ratios is not None:
            self._desired_speed = self._desired_speed*ratios
        
    # add obstacle
    def obstacles(self):
        return self._obstacles
    @property
    def position(self):
        return self._position

    @property
    def desired_speed(self):
        return self._desired_speed

    def get_neighbors(self, scenario,desired_speed):
        """
        Compute all neighbors in a 9 cell neighborhood of the current position.
        :param scenario: The scenario instance.
        :return: A list of neighbor cell indices (x,y) around the current position.
        """
        # a part of task 3 modified by Jun take speed in to account 
        # pedestrains can move if distance from those centers of cells to center of current cell are less than speed,in other words,in the circle with speed as radius and current position is center 
        radius = desired_speed
        cover_cells=[]
        for x in range(-int(radius) -1,int(radius) + 1):
            for y in range(-int(radius) -1,int(radius) + 1):
                if x**2 + y**2 <= radius**2:
                    cover_cells.append((x, y))

        return [
            (int(cell[0] + self._position[0]), int(cell[1] + self._position[1]))
            for cell in cover_cells

            if 0 <= cell[0] + self._position[0] < scenario.width and 0 <= cell[1]  + self._position[1] < scenario.height and np.abs(x) + np.abs(y) > 0
        ]
    



    def update_step(self,scenario,all_ped_position,passed_steps):
        """
        Moves to the cell with the lowest distance to the target.
        This does not take obstacles or other pedestrians into account.
        Pedestrians can occupy the same cell.

        :param scenario: The current scenario instance.
        """
        # task 4: to use fast marshing to plan path, main function is in task_4.py
        with open('path.json','r') as file:
            path = json.load(file)
        passed_steps = int(passed_steps)
        # update position using the path created by task_4.py,and using passed_step as index
        if passed_steps <= len(path)-1:
            self._position = (path[passed_steps][0],path[passed_steps][1])



class Scenario:
    """
    A scenario for a cellular automaton.
    """
    GRID_SIZE = (500, 500)
    ID2NAME = {
        0: 'EMPTY',
        1: 'TARGET',
        2: 'OBSTACLE',
        3: 'PEDESTRIAN'
    }
    NAME2COLOR = {
        'EMPTY': (255, 255, 255),
        'PEDESTRIAN': (255, 0, 0),
        'TARGET': (0, 0, 255),
        'OBSTACLE': (255, 0, 255)
    }
    NAME2ID = {
        ID2NAME[0]: 0,
        ID2NAME[1]: 1,
        ID2NAME[2]: 2,
        ID2NAME[3]: 3
    }

    def __init__(self, width, height):
        if width < 1 or width > 1024:
            raise ValueError(f"Width {width} must be in [1, 1024].")
        if height < 1 or height > 1024:
            raise ValueError(f"Height {height} must be in [1, 1024].")

        self.width = width
        self.height = height
        self.grid_image = None
        self.grid = np.zeros((width, height))
        self.pedestrians = []
        self.target_distance_grids = self.recompute_target_distances()

    def recompute_target_distances(self):
        self.target_distance_grids = self.update_target_grid()
        return self.target_distance_grids

    def update_target_grid(self):
        """
        Computes the shortest distance from every grid point to the nearest target cell.
        This does not take obstacles into account.
        :returns: The distance for every grid cell, as a np.ndarray.
        """
        targets = []
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x, y] == Scenario.NAME2ID['TARGET']:
                    targets.append([y, x])  # y and x are flipped because they are in image space.
        if len(targets) == 0:
            return np.zeros((self.width, self.height))

        targets = np.row_stack(targets)
        x_space = np.arange(0, self.width)
        y_space = np.arange(0, self.height)
        xx, yy = np.meshgrid(x_space, y_space)
        positions = np.column_stack([xx.ravel(), yy.ravel()])

        # after the target positions and all grid cell positions are stored,
        # compute the pair-wise distances in one step with scipy.
        distances = scipy.spatial.distance.cdist(targets, positions)

        # now, compute the minimum over all distances to all targets.
        distances = np.min(distances, axis=0)

        return distances.reshape((self.width, self.height))

    def update_step(self,passed_steps):
        """
        Updates the position of all pedestrians.
        This does not take obstacles or other pedestrians into account.
        Pedestrians can occupy the same cell.
        """
        # modified by Jun to adjust speed 
        all_ped_position=[]
        ratios = 1
        # save position of all pedestrains
        for pedestrian in self.pedestrians:
            all_ped_position.append(pedestrian.position) 
        # store move distance of one update for all pedestrains
        move_list = []

        for pedestrian in self.pedestrians:
            # print(pedestrian.desired_speed)
            move_dist = pedestrian.update_step(self,all_ped_position,passed_steps)
            move_list.append(move_dist)
        # arrived pedestrain's move_distance is none, remove those none so that we can calculate ratios
        no_none_list = [x for x in move_list if x is not None]
        # use average move distance to adjust the speed to be average speed 
        # mean(v)= v*mean(distance)/ distance
        if len(no_none_list) != 0:
            average = sum(no_none_list) / len(no_none_list)
            ratios = []
            for num in move_list:
                if num is not None:
                    ra = average/num 
                    ratios.append(ra)
                else:
                    ratios.append(None) 
            i=0
        # adjust the speed after every update of position for all pedestrains
            for pedestrian in self.pedestrians:
                pedestrian.change_speed(ratios[i])
                i+=1

        # end by jun 
        


    @staticmethod
    def cell_to_color(_id):
        return Scenario.NAME2COLOR[Scenario.ID2NAME[_id]]

    def target_grid_to_image(self, canvas, old_image_id):
        """
        Creates a colored image based on the distance to the target stored in
        self.target_distance_gids.
        :param canvas: the canvas that holds the image.
        :param old_image_id: the id of the old grid image.
        """
        im = Image.new(mode="RGB", size=(self.width, self.height))
        pix = im.load()
        for x in range(self.width):
            for y in range(self.height):
                target_distance = self.target_distance_grids[x][y]
                pix[x, y] = (max(0, min(255, int(10 * target_distance) - 0 * 255)),
                             max(0, min(255, int(10 * target_distance) - 1 * 255)),
                             max(0, min(255, int(10 * target_distance) - 2 * 255)))
        im = im.resize(Scenario.GRID_SIZE, Image.NONE)
        self.grid_image = ImageTk.PhotoImage(im)
        canvas.itemconfigure(old_image_id, image=self.grid_image)

    def to_image(self, canvas, old_image_id):
        """
        Creates a colored image based on the ids stored in self.grid.
        Pedestrians are drawn afterwards, separately.
        :param canvas: the canvas that holds the image.
        :param old_image_id: the id of the old grid image.
        """
        im = Image.new(mode="RGB", size=(self.width, self.height))
        pix = im.load()
        for x in range(self.width):
            for y in range(self.height):
                pix[x, y] = self.cell_to_color(self.grid[x, y])
        for pedestrian in self.pedestrians:
            x, y = pedestrian.position
            pix[x, y] = Scenario.NAME2COLOR['PEDESTRIAN']
        im = im.resize(Scenario.GRID_SIZE, Image.NONE)
        self.grid_image = ImageTk.PhotoImage(im)
        canvas.itemconfigure(old_image_id, image=self.grid_image)
