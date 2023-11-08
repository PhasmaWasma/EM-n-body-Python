import math
import numpy as np 
import sys
import matplotlib.pyplot as plt 
import matplotlib.animation as anim 
import matplotlib.colors as mcolors

class Particle:
    ''' 
    Class that stores the mass and charge of the point particle as well as
    the position, velocity, and acceleration histories of it. These are important
    for using the retarded times at distances and speeds where the propogation
    time of the speed of light is significant
    '''
    __slots__ = ("_mass", "_charge", "_position", "_velocity", "_acceleration", "_max_distance", "_num_dimensions", "_dimensions")

    def __init__(self, position: list, velocity: list, mass: "int | float", charge: "int | float") -> None:
        ''' 
        Initializes the Particle object
        Parameters:
            position: position vector as a list of ints or floats
            velocity: velocity vector as a list of ints or floats
            mass:     mass of the Particle in kg
            charge:   charge of the Particle in Coulombs 
        Returns:
            None
        '''
        for i in range(2):
            if isinstance(position[i], int):
                position[i] = float(position[i])
            if isinstance(velocity[i], int):
                velocity[i] = float(velocity[i])

        self._mass         = mass
        self._charge       = charge 
        self._position     = [np.array(position)]
        self._velocity     = [np.array(velocity)]
        self._acceleration = []
        self._max_distance = np.linalg.norm(self._position)
        self._dimensions = [None, None, None]
        if 0.0 in position:
            self._num_dimensions = 2
            for i in range(3):
                if position[i] != 0.0:
                    self._dimensions[i] = 1
        else:
            self._num_dimensions = 3

    def getMass(self) -> float:
        ''' 
        Returns the mass (kg) of the particle
        Parameters:
            None
        Returns:
            float of the mass in kg
        '''
        return self._mass

    def getCharge(self) -> float:
        ''' 
        Returns the charge (C) of the particle
        Parameters:
            None
        Returns:
            float of the charge in C
        '''
        return self._charge

    def getPosition(self, time: int) -> "array":
        ''' 
        Returns the position of the particle at the specified sim time.
        If the time given is negative, it returns the first value instead
        Parameters:
            time: int of the instantaneous position at the simulation time
        Returns:
            numpy array of the position vector
        Raises:
            IndexError if the index is greater than the length of the total array
        '''
        if time < -1:
            return np.copy(self._position[0])
        elif time < len(self._position):
            return np.copy(self._position[time])
        else:
            raise IndexError("Given time is greater than the length of the array")

    def getVelocity(self, time: int) -> "array":
        ''' 
        Returns the velocty of the particle at the specified sim time.
        If the time given is negative, it returns the first value instead
        Parameters:
            time: int of the instantaneous velocity at the simulation time
        Returns:
            numpy array of the velocity vector
        Raises:
            IndexError if the index is greater than the length of the total array
        '''
        if time < -1:
            return np.copy(self._velocity[0])
        elif time < len(self._velocity):
            return np.copy(self._velocity[time])
        else:
            raise IndexError("Given time is greater than the length of the array")

    def getAccel(self, time: int) -> "array":
        ''' 
        Returns the acceleration of the particle at the specified sim time.
        If the time given is negative, it returns the first value instead
        Parameters:
            time: int of the instantaneous acceleration at the simulation time
        Returns:
            numpy array of the acceleration vector
        Raises:
            IndexError if the index is greater than the length of the total array
        '''
        if time < -1:
            return np.copy(self._acceleration[0])
        elif time < len(self._acceleration):
            return np.copy(self._acceleration[time])
        elif time == 0 and len(self._acceleration) == 0:
            return np.array([0.0, 0.0, 0.0])
        else:
            raise IndexError("Given time is greater than the length of the array")

    def getMaxDist(self) -> float:
        ''' 
        Returns the max distance from the origin the particle reached
        Parameters:
            None
        Returns:
            float of the furthest distance it was from the origin
        '''
        return self._max_distance

    def getDims(self) -> int:
        return self._num_dimensions

    def addPosition(self, new_position: "array") -> None:
        ''' 
        Appends the given position vector to the position history.
        Also updates the _max_distance and dimensions if appropriate
        Parameters:
            new_position: numpy array of the position to append
        Returns:
            None
        '''
        self._position.append(new_position)
        total_dist = np.linalg.norm(new_position)
        if total_dist > self._max_distance:
            self._max_distance = total_dist

        if self._num_dimensions == 3:
            pass 
        elif 0.0 not in new_position:
            self._num_dimensions = 3
            self._dimensions = [1, 1, 1]

    def addVelocity(self, new_velocity: "array") -> None:
        ''' 
        Appends the given velocity vector to the velocity history
        Parameters:
            new_velocity: numpy array of the velocity to append
        Returns:
            None
        '''
        self._velocity.append(new_velocity)
    
    def addAccel(self, new_accel: "array") -> None:
        ''' 
        Appends the given acceleration vector to the acceleration history
        Parameters:
            new_acceleration: numpy array of the acceleration to append
        Returns:
            None
        '''
        self._acceleration.append(new_accel)

    def getAllPositions(self) -> "list[array]":
        ''' 
        Returns a copy of the entire position history of the particle
        Parameters:
            None
        Returns:
            list of each numpy array position vector
        '''
        return self._position.copy()

    def getDimsShape(self) -> list:
        ''' 
        Returns the shape of the dimensions for the particle.
        1 means the dimension has values, None means no nonzero values
        Parameters:
            None
        Returns:
            a list of the shape for the dimensions
        '''
        return self._dimensions.copy()

    def getAllComponents(self) -> "array":
        ''' 
        Returns a list of the x, y, and z component histories
        If there are only two dimensions with values, the list 
        will have None in the dimension with no values
        Parameters:
            None
        Returns:   
            list of numpy arrays with dimension position data
        '''
        length = len(self._position)
        x_pos    = np.zeros(length)
        y_pos    = np.zeros(length)
        z_pos    = np.zeros(length)
        all_dims = [x_pos, y_pos, z_pos]

        for i in range(length):
            x_pos[i] = self._position[i][0]
            y_pos[i] = self._position[i][1]
            z_pos[i] = self._position[i][2]

        if self._num_dimensions == 3:
            return [x_pos, y_pos, z_pos]
        else:
            out = self._dimensions.copy()
            for i in range(3):
                if out[i] != None:
                    out[i] = all_dims[i]
            
            return out

def lorentzForce(p_test: Particle, p_source: Particle, t_curr: int, t_step: float, rk4_pos: "array" = 0, rk4_vel: "array" = 0) -> "array":
    ''' 
    Calculates and returns the force vector on p_test from p_source
    Parameters:
        p_test:   the particle the force is being calculated for
        p_source: the particle creating the field affecing p_test
        t_curr:   the current time in the simulation
        t_step:   the width of each time step
        rk4_pos:  the change to the position from the rk4 func
        rk4_vel:  the change to the velocity from the rk4 func
    Returns:
        force vector affecting p_test in numpy array
    '''
    c   = 3e8
    e_0 = 8.854187e-12
    r = (p_test.getPosition(-1) + rk4_pos) - p_source.getPosition(t_curr) # this is 𝓇
    t_r = 0
    if np.linalg.norm(r) / c <= t_step and np.linalg.norm(p_source.getVelocity(t_curr)) < 0.05 * c:
        t_r = t_curr
    else:
        #have to compute the t_r somehow 
        pass

    r_dir = r / (np.linalg.norm(r))
    u     = (c * r_dir) - p_source.getVelocity(t_r)
    a     = 0
    if t_curr == 0:
        a = np.array([0.0, 0.0, 0.0])
    else:
        a = p_source.getAccel(t_r)

    v_field   = (c**2 * np.linalg.norm(p_source.getVelocity(t_r))**2) * u 
    rad_field = np.cross(r, np.cross(u, a))

    e_field = v_field + rad_field
    b_field = np.cross(((p_test.getVelocity(t_curr) + rk4_vel) / c), np.cross(r_dir, e_field))
    scaling = (np.dot(r, u))**3
    scaling = np.linalg.norm(r) / scaling

    F_lorentz  = (p_test.getCharge() * p_source.getCharge()) / (4 * np.pi * e_0)
    F_lorentz *= scaling * (e_field + b_field)

    return F_lorentz 

def getAccel(p_test: Particle, sources: "list[Particle]", t_curr: int, t_step: float, rk4_pos: "array" = 0, rk4_vel: "array" = 0) -> "array":
    ''' 
    Calculates and returns the net acceleration vector on p_test 
    Parameters:
        p_test:  the Particle the net acceleration is being calculated for
        sources: list of all the total Particle objects in the simulation
        t_curr:  int of which time step the simulation is currently at
        t_step:  float of the length of each time step
        rk4_pos: the change to the position from the rk4 func
        rk4_vel: the change to the velocity from the rk4 func
    Returns:
        numpy array vector of the net acceleration
    '''
    a_net = 0
    for p in sources:
        if p is p_test:
            continue
        else:
            a_net += lorentzForce(p_test, p, t_curr, t_step, rk4_pos, rk4_vel)

    return a_net / p_test.getMass()

def rk4(p_test: Particle, sources: "list[Particle]", t_curr: int, t_step: float) -> None:
    ''' 
    Fourth order Runge-Kutta solver that finds the new position and velocity vectors
    for p_test, then appends it to the position and velocity histories
    Parameters:
        p_test: the Particle the net acceleration is being calculated for
        sources: list of all the total Particle objects in the simulation
        t_curr: int of which time step the simulation is currently at
        t_step: float of the length of each time step
    Returns:
        None
    '''
    test_pos = p_test.getPosition(t_curr)
    test_vel = p_test.getVelocity(t_curr)

    kv_0 = test_vel
    ka_0 = getAccel(p_test, sources, t_curr, t_step)

    kv_1 = test_vel + (ka_0 * 0.5 * t_step)
    ka_1 = getAccel(p_test, sources, t_curr, t_step, (kv_0 * 0.5 * t_step), kv_0) #uses the additional change to position and velocity to calc next accel 

    kv_2 = test_vel + (ka_1 * 0.5 * t_step)
    ka_2 = getAccel(p_test, sources, t_curr, t_step, (kv_1 * 0.5 * t_step), kv_1)

    kv_3 = test_vel + (ka_2 * 0.5 * t_step)
    ka_3 = getAccel(p_test, sources, t_curr, t_step, (kv_2 * 0.5 * t_step), kv_2)

    kv_4 = test_vel + (ka_3 * 0.5 * t_step)
    ka_4 = getAccel(p_test, sources, t_curr, t_step, (kv_3 * 0.5 * t_step), kv_3)

    test_pos += (kv_1 + (kv_2 * 2.0) + (kv_3 * 2.0) + kv_4) * t_step / 6.0 #weighted sum of the 4 stages
    test_vel += (ka_1 + (ka_2 * 2.0) + (ka_3 * 2.0) + ka_4) * t_step / 6.0

    p_test.addPosition(test_pos)
    p_test.addVelocity(test_vel)
    p_test.addAccel(ka_0)

def runSimulation(filename: str, sources: "list[Particle]", t_total: float, t_step: float = 0.01) -> None:
    ''' 
    Runs the electrodynamics sim for t_total seconds
    Parameters:
        sources:  a list of every Particle that will be in the simulation
        filename: the name of the output video
        t_total:  the length of the simulation in seconds
        t_step:   the duration of each step in the simulation
    Returns:
        None
    '''
    total_steps = int(t_total / t_step) + 1

    for charge in sources:
        charge.addAccel(getAccel(charge, sources, 0, t_step))

    for i in range(0, total_steps):
        for charge in sources:
            rk4(charge, sources, i, t_step)
        
        if i % 10 == 0 and i != 0:
            print(f"Steps {i} of {total_steps-1} completed")
    
    print("Simulation complete. Exporting to video now")

def dataToVideo2D(filename: str, source_x: "list[array]", source_y: "list[array]", x_name: str, y_name: str, t_step: float, max_size: float, with_paths: bool, auto_zoom: bool) -> None:
    ''' 
    Takes the lists of the source x and y positions and graphs each frame
    and makes a movie of the resulting frames
    Parameters:
        filename:   name of the output file as a str
        source_x:   list of the x position histories of each particle
        source_y:   list of the y position histories of each particle
        x_name:     the dimension of the first axis with units
        y_name:     the dimension of the second axis with units
        t_step:     duration of the time step
        max_size:   largest size of the graph
        vid_specs:  dict containing all of the specific parameters for the video
        with_paths: bool that determines if particle paths will be shown
        auto_zoom:  bool that determines if the plot scale will be fixed or scale automatically
    Returns:
        None, saves the video to file
    '''
    colors = []
    if len(source_x) < 18:
        for color in mcolors.BASE_COLORS:
            if color != "w":
                colors.append(color)
        
        for color in mcolors.TABLEAU_COLORS:
            colors.append(color)
    else:
        for color in mcolors.CSS4_COLORS:
            if "white" not in color:
                colors.append(color)

    metadata = dict(title="Movie", artist="Opossum")
    writer = anim.FFMpegWriter(fps=24, metadata=metadata)

    fig = plt.figure()
    fig.set_size_inches(5, 5)

    with writer.saving(fig, filename, 100):

        for t in range(len(source_x[0])):
            t_curr = t * t_step
            plt.title(f"t = {t_curr} (s)")

            if auto_zoom == False:
                plt.xlim(-max_size, max_size)
                plt.ylim(-max_size, max_size)
        
            plt.xlabel(x_name)
            plt.ylabel(y_name)

            for source in range(len(source_x)):
                plt.plot(source_x[source][t], source_y[source][t], marker=".", color=colors[source], label=f"p{source}")

                if with_paths:
                    plt.plot(source_x[source][0:t+1], source_y[source][0:t+1], marker="None", color=colors[source])
            
            writer.grab_frame()
            plt.clf()

def plot2D(sources: "list[array]", t_step: float, filename: str, with_paths: bool = True, auto_zoom: bool = True) -> None:
    ''' 
    Plots the position data of all the charges in 2D
    '''
    max_size      = 0
    max_size      = math.ceil(max_size) #the max size determines the final image size if auto_zoom is False
    source_shapes = []
    
    for p in sources:
        dist = p.getMaxDist()
        if dist > max_size:
            max_size = dist
        
        if p.getDims() > 2:
            response = input("The given sim data is 3D. Would you prefer to plot as 3D instead? (y/n)")
            if response == "y" or response == "yes":
                #send to 3D plot func with the same args
                pass
            else:
                sys.exit()
        else:
            source_shapes.append(p.getDimsShape())
    
    source_all = []
    source_x   = []
    source_y   = []
    x_name     = ""
    y_name     = ""
    
    if source_shapes.count(source_shapes[0]) != len(source_shapes):
        response = input("The given sim data is 3D. Would you prefer to plot as 3D instead? (y/n)")
        if response == "y" or response == "yes":
            #send to 3D plot func with the same args
            pass
        else:
            sys.exit()
    else:
        for source in sources:
            source_all.append(source.getAllComponents())

        for i in range(3):
            if source_shapes[0][i] != None and x_name == "":
                x_name = f"{chr(ord('x') + i)} (m)"
                for source in source_all:
                    source_x.append(source[i])

            elif source_shapes[0][i] != None and x_name != "":
                y_name = f"{chr(ord('x') + i)} (m)"
                for source in source_all:
                    source_y.append(source[i])
                
    dataToVideo2D(filename, source_x, source_y, x_name, y_name, t_step, max_size, with_paths, auto_zoom)
    

def main() -> None:
    p1 = Particle([1, 1, 0], [0.5, 0, 0], 0.1, 1e-5)
    p2 = Particle([1, -1, 0], [0, -0.5, 0], 0.1, 1e-5)
    p3 = Particle([-1, 1, 0], [-0.5, 0, 0], 0.1, 1e-5)
    p4 = Particle([-1, -1, 0], [0, 0.5, 0], 0.1, 1e-5)

    sources = [p1, p2, p3, p4]

    t_step   = 0.05
    t_total  = 1.5
    filename = "test_1.mp4"

    with_paths = True
    auto_zoom  = True

    runSimulation(filename, sources, t_total, t_step)

    plot2D(sources, t_step, with_paths, auto_zoom)

if __name__ == "__main__":
    main()