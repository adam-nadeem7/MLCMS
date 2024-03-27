import scipy.spatial.distance
from PIL import Image, ImageTk
import numpy as np
import math


class Pedestrian:
    """
    Defines a single pedestrian.
    """

    def __init__(self, position, desired_speed,obstacles,targets):
        self._position = position
        self._desired_speed = desired_speed
        self._obstacles = obstacles
        self._targets = targets

    @property
    def position(self):
        return self._position

    # def change_speed(self, ratios):
    #     if ratios is not None:
    #         self._desired_speed = self._desired_speed*ratios


    @property
    def desired_speed(self):
        return self._desired_speed

    def get_neighbors(self, scenario, desired_speed):
        """
        Compute all neighbors in a 9 cell neighborhood of the current position.
        :param scenario: The scenario instance.
        :return: A list of neighbor cell indices (x,y) around the current position.
        """
        # return [
        #     (int(x + self._position[0]), int(y + self._position[1]))
        #     for x in [-1, 0, 1]
        #     for y in [-1, 0, 1]
        #     if 0 <= x + self._position[0] < scenario.width and 0 <= y + self._position[1] < scenario.height and np.abs(x) + np.abs(y) > 0
        # ]
    
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
        # neighbors = self.get_neighbors(scenario)
        # next_cell_distance = scenario.target_distance_grids[self._position[0]][self._position[1]]
        # next_pos = self._position
        # for (n_x, n_y) in neighbors:
        #     if next_cell_distance > scenario.target_distance_grids[n_x, n_y]:
        #         next_pos = (n_x, n_y)
        #         next_cell_distance = scenario.target_distance_grids[n_x, n_y]
        # self._position = next_pos            

        neighbors = self.get_neighbors(scenario, self._desired_speed)
        # check if there is obstacle or other pedestains
        # distance to target from current position
        orin_position = self._position
        current_cell_distance = scenario.target_distance_grids[self._position[0]][self._position[1]]
        dic_dist_pos = {current_cell_distance:self._position}
        # build a dict 
        # key: current and neighbors' distances to target
        # value: current and neighbors' positions
        #  take current position in to account is because it maybe arrived target
        for (n_x, n_y) in neighbors:  
            dist = scenario.target_distance_grids[n_x, n_y]
            dic_dist_pos[dist] = (n_x, n_y)
        # sort the dictionary with distances,find the nearest one to target)
        sorted_dict = dict(sorted(dic_dist_pos.items(), key=lambda item: item[0]))

        # all_ped_position is unaccessable positiion,because pedestrain should not overlap
        # make current position and target position accessable, in case some padestrain arrived and some not
        if (self._position[0],self._position[1]) in all_ped_position:
            all_ped_position.remove((self._position[0],self._position[1]))
        for tar in self._targets:
            while tar in all_ped_position:
                all_ped_position.remove(tar)
        # remove the next_position if it is obstacle or pedestrains, then build no_ob_sorted_dict
        no_ob_sorted_dict= {key: value for key, value in sorted_dict.items() if value not in self._obstacles+all_ped_position}
        # update position
        self._position = (list(no_ob_sorted_dict.values())[0])
        # move_distance is to calculate ratio to adjust the speed
        move_distance = math.sqrt(((self._position[0]-orin_position[0])**2+(self._position[1]-orin_position[1])**2))
        # add new position of one pedestrain, make it unaccessable to other pedestrains
        all_ped_position.append(self._position)
        # saving distance that the pedestrains move aimed to ajust speed
        if move_distance != 0:
            return move_distance
        else:
            return None
        # end bu Jun


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
            move_dist = pedestrian.update_step(self,all_ped_position,passed_steps)
            move_list.append(move_dist)

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

