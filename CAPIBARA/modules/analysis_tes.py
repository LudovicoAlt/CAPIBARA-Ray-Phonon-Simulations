import numpy as np
from scipy import signal

def box_intersect_optim(p0, n_box, y0, dir):

    t = (np.einsum('ik,ijk->ji', n_box, -y0+p0[:, None])/(dir@n_box.T) ).T  

    return t

def reflected_dir_optim(dir, n):
  
  return dir - 2*np.einsum('ij,ij,ia->ia', dir, n, n)

def ray(p_0, dir, t):
  return (p_0 + t * dir)

def response(t, tau_rise=10, tau_fall=200):
    return np.exp(-t/tau_fall)-np.exp(-t/tau_rise)

n_t_bins = 500 #Also sets max time
pad_dim = 4_000
dt = 0.5 #time step of simulation (mus)
t_conv = np.arange(n_t_bins+pad_dim)*dt
convolution_data = response(t_conv)

class Geometry_TES():
    detector_positions = []
    labels = []
    normals_KIDs = []

    def __init__(self, wall_positions, detector_positions, wall_normals, detector_normals, detector_labels, detector_radius):

        self.wall_positions = wall_positions
        self.detector_positions = detector_positions
        self.wall_normals = wall_normals
        self.detector_normals = detector_normals
        self.detector_labels = detector_labels
        self.detector_radius = detector_radius

class Analysis_TES():

    def __init__(self, crystal_geometry : Geometry_TES , v_fast = 1, v_slow = 0.5, p_abs = 0.0064, n_sims=10_000, lifetime=1000, steps=1000, epsilon=0.0008):
        '''
        v_fast : float : velocity of the phonons in the fast direction
        v_slow : float : velocity of the phonons in the slow direction
        p_abs : float : probability of being absorbed
        crystal_geometry : Geometry : geometry of the crystal
        n_sims : int : number of phonons to simulate (default: 10_000)
        lifetime : float : lifetime of the phonons (default: 1000)
        steps : int : number of simulation steps (default: 1000)
        epsilon : float : small offset to avoid degenerate positions (default: 0.0008)
        '''

        self.v_fast     = v_fast
        self.v_slow     = v_slow
        self.p_abs      = p_abs
        self.n_sims     = n_sims
        self.lifetime   = lifetime
        self.steps      = steps
        self.epsilon    = epsilon

        self.crystal_geometry   = crystal_geometry
        self.w                  = crystal_geometry.detector_radius

    #v_slow is on a plane, while v_fast is on z direction
    def initialize(self, starting_position):

        a = np.random.randn( 3, self.n_sims ) #dim (3, n_sims)
        a = a/sum(a*a)**0.5 # normalize on a sphere r/|r|

        v_s = self.v_slow/2**0.5 #divide by sqrt of 2 as it is the diagonal speed might have to double check for TES
        v_f = self.v_fast
        
        speeds = np.array( (v_s, v_s, v_f) )

        directions_t_0 = a.T * speeds  #dim (n_sims, 3) give direction of the phonons scaled by the speeds

        positions_0 = np.zeros_like(directions_t_0)+starting_position+directions_t_0*self.epsilon/(speeds/np.sum(speeds**2)**0.5) #Start them all at starting_position + direction * epsilon is necessary otherwise degenerate

        return positions_0, directions_t_0

    def get_intersections(self, positions, directions):
        wall_intersections = box_intersect_optim(self.crystal_geometry.wall_positions, self.crystal_geometry.wall_normals, positions, directions)
        detector_intersections = box_intersect_optim(self.crystal_geometry.detector_positions, self.crystal_geometry.detector_normals, positions, directions)
        return wall_intersections, detector_intersections

    def simulation(self, starting_position):

        positions_0, directions_t_0 = self.initialize(starting_position)

        mean_free_paths = -self.lifetime * np.log(np.random.uniform(0, 1, self.n_sims)) #TODO figure out proper lifetime to be 100 times box normalize mean free path to d/2 ????
        phonon_total_path = np.zeros(self.n_sims)

        absorbed_phonons = [] #(detector, time)
        total_times = np.zeros( self.n_sims )

        positions = positions_0
        directions = directions_t_0

        #Array of all intersected positions, step 0 is the initial state so add 1 to the steps
        intersected_pos_array = np.zeros( (self.steps+1, self.n_sims, 3) )
        intersected_pos_array[0] = positions

        for i in range(self.steps):
            wall_intersections_all, detector_intersections_all = self.get_intersections(positions, directions)

            closest_hit = np.where( (wall_intersections_all > 0), wall_intersections_all, np.inf).argmin(0) 

            not_dead = (positions.sum(1) != 0) # phonons @ (0,0,0) are dead, probably better to put them elsewhere
            
            time_evolve = wall_intersections_all.T[np.arange(self.n_sims), closest_hit]
            
            intersected_pos = ray(positions, directions, time_evolve[:,None]) #ray to the plane, agnostic of the disc

            inter_dist = ( intersected_pos[:,:,None]-self.crystal_geometry.detector_positions.T ) # (n_sims, dr(3), n_det)
            distance = np.einsum('ijk,ijk->ik', inter_dist, inter_dist) # (n_sims, n_det) with distance to detectors
            minimum_hit = np.argmin(distance, axis=1) # (n_sims) with min detector hit
            minimum_distance = distance[np.arange(self.n_sims), minimum_hit] # (n_sims) with min distance to detector
            absorbed_in_kid = (minimum_distance < self.w)

            # check distance to detectors vs intersection_pos
            
            # absorbed_in_kid = np.sqrt( np.einsum('ij,ij->i', inter_dist, inter_dist) ) < self.w #Check intersection with the KID

            total_times += time_evolve 
            
            diff = intersected_pos-positions
            phonon_total_path += (diff**2).sum(1)**0.5 #To check if beyond mean free path

            positions[not_dead] = (intersected_pos + self.epsilon*self.crystal_geometry.wall_normals[closest_hit])[not_dead]
            
            directions = reflected_dir_optim(directions, self.crystal_geometry.wall_normals[closest_hit])
            #Randomly pick a probability ?
            abs_chance = (np.random.uniform(0, 1, len(minimum_distance)) < self.p_abs)

            #! the hard coded clostest_hit == 0 is necessary to keep code efficient, 
            #! otherwise only 1 TES would absorb phonons as it is the first in line 
            hittable = len(self.crystal_geometry.detector_labels) #? only TES with labels can be hit (in order)

            #! Hard code the condition for hittable to 0 (top wall) but can be changed by modifying hittable
            absorption_condition = ( (absorbed_in_kid) & (abs_chance) & (closest_hit==0) & (not_dead) )
        
            conditions = ( absorption_condition ) | (phonon_total_path > mean_free_paths)

            positions[ conditions ] = np.zeros(3) #killed phonons 
            directions[ conditions  ] = np.zeros(3) #killed phonons

            #only save the absorbed phonons
            abs_tracker = np.array( (total_times[absorption_condition], 
                                      minimum_hit[absorption_condition] ) ).T

            #probably a faster way of appending
            for i_track in abs_tracker:
                 absorbed_phonons.append(i_track)

            intersected_pos_array[i+1] = intersected_pos

        #Return history of intersections and their time profile in the detectors
        return intersected_pos_array, absorbed_phonons

    def build_hist(self, absorbed_phonons):

        detectors = np.array(absorbed_phonons).T[1]
        times = np.array(absorbed_phonons).T[0]
        hist_phonons = np.zeros( (len(self.crystal_geometry.detector_labels), n_t_bins) )

        for i_det, _ in enumerate(self.crystal_geometry.detector_labels):
            hist_phonons[i_det], edge_1 = np.histogram(times[detectors==i_det], bins=n_t_bins, range=(0, n_t_bins))
        
        width = (edge_1[1]-edge_1[0])/2
        edge_1 = edge_1 + width
        edge_1 = edge_1[:-1]

        return hist_phonons, edge_1

    def build_response(self, hist_phonons):
        t_extra = np.arange(len(convolution_data)+len(hist_phonons[0])+pad_dim-1)*dt
        res_arr = []

        for k, l in enumerate(self.crystal_geometry.detector_labels):
            extra_plot = np.pad(hist_phonons[k], (0, pad_dim), 'constant')
            res = signal.convolve(extra_plot, convolution_data)
            res_arr.append(res)

        return res_arr, t_extra
