import numpy as np
import csv      
def load_planets(planets_file):
    """
    Loads planets data set and returns it in the variable planets
    """
    with open('planets.csv', newline='') as planets_file:
        csv_reader = csv.reader(planets_file)
        planets = []
        for i in csv_reader:
            planets.append(i)
   
        return planets

# We transform planets into a numpy array so it's easier to do operations with it, and we convert to radians the phi and omega values
planets = np.array(load_planets("planets.csv"),dtype=np.float64)
planets[:, 2] = np.radians(planets[:, 2])  
planets[:, 3] = np.radians(planets[:, 3])  
planets_copy = np.copy(planets)
planets_copy2 = np.copy(planets)


def get_planet_r(planets):    
    """
    This function returns a single array containing the radius from all of the planets to the sun.
    We substract all of the data from the planet data set and we plug it in the radius formula
    """
      
    a = planets[:, 0]  # First column
    e = planets[:, 1]  # Second column
    w = planets[:, 2]  # Third column
    phi = planets[:, 3]  # Fourth column
    r = (a*(1 - e**2))/(1 + e * np.cos(phi - w))
    return r


def get_planet_coords(planets):
    """
    Returns a single 3d array containing the coordinates of each planet
    """
    r = get_planet_r(planets) #current distance form planet to the sun
    phi = planets[:, 3]
    x = r * np.cos(phi)  # x = r * cos(θ)
    y = r * np.sin(phi)  # y = r * sin(θ)
    z = np.zeros_like(phi)   # z-coordinate is 0 if planets are assumed to be in the same plane
    
    # Combine x, y, and z into a single array of coordinates
    coordinates = np.array([x, y, z]).T #.T for transpose the array and better visualization
    return coordinates

def get_planet_orbits(planets, phi_linspace):
 
    """
    Given an array of numbers phi_linspace it will return the coordinates of each planet at each point of the array.
    Formatted so the array data is divided by planet.
    """
    a = planets[:, 0]  # First column
    e = planets[:, 1]  # Second column
    w = planets[:, 2]  # Third column
    
    r = [] #radius list
    orbits = [] #orbits list
    for j in range(len(a)): #iterating over the length of planets array, from 0 to 7
        orbit = [] #orbit list to store each value x and y
        
        for i in phi_linspace:
            r = ((a[j] * (1 - e[j]**2)) / (1 + e[j] * np.cos(i - w[j]))) #iterates for 1 planet in all of phi_linspace points before moving on to iterating over the next planet
            
            x = r * np.cos(i)
            y = r * np.sin(i)
            orbit.append((x, y))
        orbits.append(orbit)
    
    return orbits

def update_all_planets(planets, dt):
    """
    Updates the planets array in the global scope for a timestep dt
    """
    G_M_sun = 3.964 * 10 ** (-14)  # Gravitational parameter (simplified)

    for planet in planets:
        a = planet[0]
        e = planet[1] 
        w = planet[2]  
        phi = planet[3]        
        r = (a * (1 - e**2)) / (1 + e * np.cos((phi - w)))
        planet[3] = phi + (np.sqrt(G_M_sun * a * (1 - e **2)) / r ** 2) * dt #we change phi value in planets array

        
        
def accel_g_sun(vec_r):
    M_sun_G = -3.9639224e-14    
    r = (np.sqrt(np.sum(vec_r**2)))
    acceleration = M_sun_G / r**3 * vec_r
    return acceleration

    

def accel_g_jupiter(vec_r, planets):
    M_jupiter_G = 1.1904*10**(-19)  
    jupiter_pos = get_planet_coords(planets)[4]
    relative_pos = vec_r - jupiter_pos
    r = np.sqrt(relative_pos[0]**2 + relative_pos[1]**2 + relative_pos[2]**2)
    acceleration = -M_jupiter_G / r**3 * relative_pos  
    return acceleration



def get_planet_distances(vec_r, planets):
    coordinates = get_planet_coords(planets)
    
    distance = coordinates - vec_r

    distances = np.sqrt(np.sum(distance**2, axis=1))
    
    return distances

    
    
def accel_total(vec_r, planets, sun_only=False):
    """
    Helper function for find_trajectory below.
    """
    if sun_only:
        return accel_g_sun(vec_r)
    else:
        return accel_g_sun(vec_r) + accel_g_jupiter(vec_r, planets)


def find_trajectory(vec_r0, vec_v0, planets, t_steps, sun_only=True):
    """
    Main loop for solar system gravitation project.

    Arguments:
    =====
    * vec_r0: Initial 3-vector position of the small mass (Cartesian coordinates.)  
    * vec_v0: Initial 3-vector velocity of the small mass (Cartesian coordinates.)
    * planets: an (Np x 4) planet array, at their initial positions - see API for description.
        Each row in a planet array contains the values:
            (a, eps, omega, phi)
        where phi is the planet's angular position, 
        and a, eps, omega are orbital parameters.
    * t_steps: NumPy array (linspace or arange) specifying the range of times to simulate
        the trajectory over, regularly spaced by timestep dt.
    * sun_only: binary flag.  If set to True, only the Sun's gravity is included in the simulation.
        If False, then Jupiter's acceleration is included as well.

    Returns:
    =====
    A tuple of the form (r, planet_distance, v).

    "r" contains the coordinates (x,y,z) of the test mass at each 
    corresponding time in t_steps, as a (3 x Nt) array.
    "planet_distance" contains the distances from the small mass
    to each planet in planets, in order, as a function of time - this is a
    (Np x Nt) array.
    "v" contains the velocity vector of the test mass at each time
    in t_steps, as a (3 x Nt) array.

    """

    dt = t_steps[1] - t_steps[0]
    Nt = len(t_steps)

    r = np.zeros((3, Nt))
    r[:,0] = vec_r0

    v = np.zeros((3, Nt))
    v[:,0] = vec_v0

    planet_distance = np.zeros((8, Nt)) 
    planet_distance[:,0] = get_planet_distances(vec_r0, planets)    

    # Copy the planets array so we don't change the original!
    local_planets = planets.copy()

    for i in range(Nt-1):
        ## Omelyan SST update

        ## V dt/6
        update_all_planets(local_planets, dt/6)
        vec_v = v[:,i] + (dt/6) * accel_total(r[:,i], local_planets, sun_only=sun_only)

        ## T dt/2
        vec_r = r[:,i] + vec_v * dt/2 

        ## V 2dt/3
        update_all_planets(local_planets, 2*dt/3)
        vec_v = vec_v + (2*dt/3) * accel_total(vec_r, local_planets, sun_only=sun_only)

        ## T dt/2; final position update
        r[:,i+1] = vec_r + vec_v * dt/2

        ## V dt/6; final velocity update
        update_all_planets(local_planets, dt/6)
        v[:,i+1] = vec_v + (dt/6) * accel_total(r[:,i+1], local_planets, sun_only=sun_only)

        planet_distance[:,i+1] = get_planet_distances(r[:,i+1], local_planets)
        
    return (r, planet_distance, v)



def run_API_tests():
    # Test 1: Load planets and ensure correct type and basic structure
    test_planets = load_planets("planets.csv")
    assert isinstance(test_planets, list), "load_planets should return a list."
    assert len(test_planets) > 0 and len(test_planets[0]) == 4, "Each planet entry should have exactly four elements."

    # Convert test_planets to NumPy array for further testing
    test_planets = np.array(test_planets, dtype=np.float64)
    
    # Test 2: Conversion to radians for phi and omega values
    original_phi = test_planets[:, 3].copy()  # Storing original phi values for comparison
    test_planets[:, 2:4] = np.radians(test_planets[:, 2:4])
    assert not np.allclose(test_planets[:, 3], original_phi), "Phi values should be converted from degrees to radians."
    
    # Test 3: Get planet radii using get_planet_r
    radii = get_planet_r(test_planets)
    assert isinstance(radii, np.ndarray), "get_planet_r should return a numpy array."
    assert radii.shape[0] == test_planets.shape[0], "Output radii should match number of planets."

    # Test 4: Get planet coordinates and check dimensions
    coords = get_planet_coords(test_planets)
    assert coords.shape == (test_planets.shape[0], 3), "Should output 3D coordinates for each planet."
    
    # Test 5: Update all planets and check for actual update in phi
    original_phi_values = test_planets[:, 3].copy()
    update_all_planets(test_planets, 1000)  # Apply a time step large enough to see an effect
    assert not np.array_equal(test_planets[:, 3], original_phi_values), "Phi values should update with time step."

    # Test 6: Verify trajectory calculation for a single step
    t_steps = np.array([0, 86400])  # One day step to ensure noticeable movement
    trajectory_data = find_trajectory(test_planets[0, :3], np.zeros(3), test_planets, t_steps, sun_only=True)
    assert isinstance(trajectory_data, tuple) and len(trajectory_data) == 3, "find_trajectory should return a tuple of three elements."
    assert trajectory_data[0].shape[1] == t_steps.shape[0], "Trajectory data should match the number of time steps."

    

run_API_tests()