import sys
import json
import tkinter
from tkinter import simpledialog
from tkinter import Button, Canvas, Menu
from scenario_elements import Scenario, Pedestrian
import json
import time

global passed_steps
passed_steps = 0

class ScenarioCreator(tkinter.Toplevel):
    def __init__(self, master, main_gui):
        super().__init__(master)
        self.main_gui = main_gui

        self.grid_size = 20
        self.element_type = tkinter.StringVar(value='Pedestrian')
        self.scenario_data = {
            "pedestrians": [],
            "objects": [],
            "target": []
        }

        self.init_ui()

    def init_ui(self):
        # Create grid buttons
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                btn = tkinter.Button(self, text=f'{row},{col}', command=lambda r=row, c=col: self.add_element(r, c))
                btn.grid(row=row, column=col)

        # Create element type radio buttons
        tkinter.Radiobutton(self, text='Pedestrian', variable=self.element_type, value='Pedestrian').grid(row=self.grid_size, column=0, columnspan=3)
        tkinter.Radiobutton(self, text='Target', variable=self.element_type, value='Target').grid(row=self.grid_size, column=3, columnspan=3)
        tkinter.Radiobutton(self, text='Obstacle', variable=self.element_type, value='Obstacle').grid(row=self.grid_size, column=6, columnspan=3)

        # Save and continue button
        tkinter.Button(self, text='Save and Continue', command=self.save_and_close).grid(row=self.grid_size + 1, column=0, columnspan=self.grid_size)

    def add_element(self, row, col):
        element_type = self.element_type.get()
        if element_type == "Pedestrian":
            speed = simpledialog.askfloat("Input", "Enter pedestrian speed:", parent=self)
            if speed is not None:
                self.scenario_data["pedestrians"].append({'x': col, 'y': row, 'speed': speed})
        elif element_type == "Target":
            self.scenario_data["target"].append({'x': col, 'y': row})
        elif element_type == "Obstacle":
            self.scenario_data["objects"].append({'x': col, 'y': row})
        print('Added:', element_type, 'at', col, row)

    def save_scenario(self):
        with open("Trial1.json", "w") as outfile:
            json.dump(self.scenario_data, outfile)

    def save_and_close(self):
        self.save_scenario()
        if hasattr(self.main_gui, 'load_scenario_data'):
            self.main_gui.scenario_data = self.scenario_data
            self.main_gui.load_scenario_data()
            self.destroy()
        else:
            print("Main GUI does not have a load_scenario_data method")
            self.destroy()




class MainGUI():
    """
    Defines a simple graphical user interface.
    To start, use the `start_gui` method.
    """

    def __init__(self):
        self.initial_data = None
        self.scenario = None
        self.canvas = None
        self.canvas_image = None
        self.win = None

    def load_scenario_data(self):
        if hasattr(self, 'scenario_data'):
            print('Loading scenario data:', self.scenario_data)

            # Clear previous pedestrians, targets, and objects
            self.scenario.pedestrians = []
            self.scenario.grid[:] = Scenario.NAME2ID['EMPTY']

            obstacles = [(obj['x'], obj['y']) for obj in self.scenario_data.get('objects', [])]
            targets = [(tar['x'], tar['y']) for tar in self.scenario_data.get('target', [])]
                
            # Add new targets and objects
            for target in self.scenario_data.get('target', []):
                x, y = target['x'], target['y']
                self.scenario.grid[x, y] = Scenario.NAME2ID['TARGET']
                
            for obj in self.scenario_data.get('objects', []):
                x, y = obj['x'], obj['y']
                self.scenario.grid[x, y] = Scenario.NAME2ID['OBSTACLE']

            # Add new pedestrians
            for ped in self.scenario_data.get('pedestrians', []):
                x, y, speed = ped['x'], ped['y'], ped['speed']
                self.scenario.pedestrians.append(Pedestrian((x, y), speed, obstacles, targets))
                
            # Recompute distances to targets
            self.scenario.recompute_target_distances()
            
            # Visualize the updated scenario
            self.scenario.to_image(self.canvas, self.canvas_image)
            print('Scenario data loaded')
        else:
            print('No scenario data to load')


    def create_scenario(self):
        creator = ScenarioCreator(self.win, self)
        self.win.wait_window(creator)
        if hasattr(self, 'scenario_data'):
            # print('Scenario created:', self.scenario_data)
            self.load_scenario_data()
            self.scenario.to_image(self.canvas, self.canvas_image)
        else:
            print('No scenario data to load')
            


    def restart_scenario(self, ):
        if self.scenario is not None and self.canvas is not None and self.canvas_image is not None:
            self.scenario = self.create_initial_scenario()
            self.scenario.to_image(self.canvas, self.canvas_image)
            print('Scenario restarted')
        else:
            print('Cannot restart, scenario or canvas not initialized')

    def create_initial_scenario(self):
        with open('coordinates2.json', 'r') as file:
            data = json.load(file)

        pedestrians_data = data.get('pedestrians', [])
        objects_data = data.get('obstacles', [])
        targets_data = data.get('target', {})
        grid_data = data.get('size', {})
        sc = Scenario(grid_data['x'], grid_data['y'])

        pedestrians = [(ped['x'], ped['y'], ped['speed']) for ped in pedestrians_data]
        obstacle = [(obj['x'], obj['y']) for obj in objects_data]
        target = [(tar['x'], tar['y']) for tar in targets_data]

        for tar in target:
            sc.grid[tar[0], tar[1]] = Scenario.NAME2ID['TARGET']

        for obj in obstacle:
            sc.grid[obj[0], obj[1]] = Scenario.NAME2ID['OBSTACLE']

        sc.recompute_target_distances()
        sc.pedestrians = []

        for ped in pedestrians:
            sc.pedestrians.append(Pedestrian((ped[0], ped[1]), ped[2], obstacle, target))

        return sc


    def step_scenario(self, scenario, canvas, canvas_image, win):
        """
        Moves the simulation forward by one step, and visualizes the result.

        Args:
            scenario (scenario_elements.Scenario): Add _description_
            canvas (tkinter.Canvas): Add _description_
            canvas_image (missing _type_): Add _description_
        """
        # scenario.update_step()
        # scenario.to_image(canvas, canvas_image)

        global passed_steps
        # passed_steps = 0
        passed_steps += 1
        show_steps = tkinter.Label(win, text='pass steps is %d'%passed_steps)
        show_steps.place(x=20, y=100)

        scenario.update_step(passed_steps)
        scenario.to_image(canvas, canvas_image)
        return passed_steps


    def exit_gui(self, ):
        """
        Close the GUI.
        """
        sys.exit()


    def start_gui(self, ):
        """
        Creates and shows a simple user interface with a menu and multiple buttons.
        Only one button works at the moment: "step simulation".
        Also creates a rudimentary, fixed Scenario instance with three Pedestrian instances and multiple targets.
        """
        win = tkinter.Tk()
        win.geometry('500x500')  # setting the size of the window
        win.title('Cellular Automata GUI')

        menu = Menu(win)
        win.config(menu=menu)
        file_menu = Menu(menu)
        menu.add_cascade(label='Simulation', menu=file_menu)
        file_menu.add_command(label='New', command=self.create_scenario)
        file_menu.add_command(label='Restart', command=self.restart_scenario)
        file_menu.add_command(label='Close', command=self.exit_gui)

        self.canvas = Canvas(win, width=Scenario.GRID_SIZE[0], height=Scenario.GRID_SIZE[1])
        self.canvas_image = self.canvas.create_image(5, 50, image=None, anchor=tkinter.NW)
        self.canvas.pack()

        self.scenario = self.create_initial_scenario()
        self.scenario.to_image(self.canvas, self.canvas_image)

        def helper_autostep(i, step):
            if i >= int(step):
                return
            self.step_scenario(self.scenario, self.canvas, self.canvas_image, win)
            time.sleep(0.5)
            helper_autostep(i+1, step)

        # automatic simulation from Jun
        def auto_step(i):
            steps = entry.get()
            helper_autostep(i, steps)

        entry = tkinter.Entry(win, width=20)
        entry.place(x=20, y=70) 
        global passed_steps
        passed_steps=0

        
        btn = Button(win, text='automatic simulation:', command=lambda : auto_step(0))
        btn.place(x=20, y=40)

        btn = Button(win, text='Step simulation', command=lambda: self.step_scenario(self.scenario, self.canvas, self.canvas_image, win))
        btn.place(x=20, y=10)
        btn = Button(win, text='Restart simulation', command=self.restart_scenario)
        btn.place(x=200, y=10)
        btn = Button(win, text='Create simulation', command=self.create_scenario)
        btn.place(x=380, y=10)

        self.win = win
        win.mainloop()
