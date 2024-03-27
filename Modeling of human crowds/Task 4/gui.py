import sys
import tkinter
from tkinter import Button, Canvas, Menu
from scenario_elements import Scenario, Pedestrian
import json



class MainGUI():
    """
    Defines a simple graphical user interface.
    To start, use the `start_gui` method.
    """

    def create_scenario(self, ):
        print('create not implemented yet')


    def restart_scenario(self, ):
        print('restart not implemented yet')


    def step_scenario(self, scenario, canvas, canvas_image,win):
        """
        Moves the simulation forward by one step, and visualizes the result.

        Args:
            scenario (scenario_elements.Scenario): Add _description_
            canvas (tkinter.Canvas): Add _description_
            canvas_image (missing _type_): Add _description_
        """
        global passed_steps
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
        win.geometry('1000x500')  # setting the size of the window
        win.title('Cellular Automata GUI')

        menu = Menu(win)
        win.config(menu=menu)
        file_menu = Menu(menu)
        menu.add_cascade(label='Simulation', menu=file_menu)
        file_menu.add_command(label='New', command=self.create_scenario)
        file_menu.add_command(label='Restart', command=self.restart_scenario)
        file_menu.add_command(label='Close', command=self.exit_gui)

        canvas = Canvas(win, width=Scenario.GRID_SIZE[0], height=Scenario.GRID_SIZE[1])  # creating the canvas
        canvas_image = canvas.create_image(5, 50, image=None, anchor=tkinter.NW)
        canvas.pack()

        # start from Aadam
        # Read JSON data from file
        with open('task5_3.json', 'r') as file:
            data = json.load(file)

        # Extract grid size, pedestrians, and target from the JSON data
        # size of grids
        sc = Scenario(data.get('size')['x'],data.get('size')['y'])
        pedestrians_data = data.get('pedestrians', [])
        obstacles_data = data.get('obstacles', [])
        targets_data = data.get('target', {})

        pedestrians = [(ped['x'], ped['y'], ped['speed']) for ped in pedestrians_data]
        obstacles = [(obj['x'], obj['y']) for obj in obstacles_data]
        targets = [(tar['x'], tar['y']) for tar in targets_data]

        for tar in targets:
            sc.grid[tar[0], tar[1]] = Scenario.NAME2ID['TARGET']

        for obj in obstacles:
            sc.grid[obj[0], obj[1]] = Scenario.NAME2ID['OBSTACLE']

        sc.recompute_target_distances()

        sc.pedestrians = []

        for ped in pedestrians:
            sc.pedestrians.append(Pedestrian((ped[0], ped[1]),ped[2],obstacles, targets))
        # end of Aadam
        

        # can be used to show pedestrians and targets
        sc.to_image(canvas, canvas_image)

        # can be used to show the target grid instead
        # sc.target_grid_to_image(canvas, canvas_image)

        # automatic simulation from Jun
        def auto_step():
            steps = entry.get()
            i = 0
            while i < int(steps): 
                self.step_scenario(sc, canvas, canvas_image,win)
                i += 1

        entry = tkinter.Entry(win, width=20)
        entry.place(x=20, y=70) 
        global passed_steps
        passed_steps=0


        btn = Button(win, text='automatic simulation:', command=lambda : auto_step())
        btn.place(x=20, y=40)

        btn = Button(win, text='Step simulation', command=lambda: self.step_scenario(sc, canvas, canvas_image,win))
        btn.place(x=20, y=10)      
        btn = Button(win, text='Restart simulation', command=self.restart_scenario)
        btn.place(x=200, y=10)
        btn = Button(win, text='Create simulation', command=self.create_scenario)
        btn.place(x=380, y=10)

        win.mainloop()
