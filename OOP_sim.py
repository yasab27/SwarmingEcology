import numpy as np
import time
from scipy.interpolate import RectBivariateSpline
import scipy.sparse as sp
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation

def pol2cart(pol):
    return np.array((pol[0] * np.cos(pol[1]), pol[0] * np.sin(pol[1])))

class branch:

    def __init__(self, colony, W0, tip_loc, density, theta0, C0 = None):
        # --------------------------- Initialize branch variables -----------------------------

        self.width = W0 # branch width
        self.theta = theta0 # angle of the branch
        self.tip_loc = np.array(tip_loc, dtype = np.float64) # Location of the tip of the branch
        self.colony = colony # the colony which the branch is part of 
        self.density = density # Local branch density
        self.crit_R = 1.5/self.density
        self.C = np.zeros(colony.sim.dims) # cell density grid
        self.P = np.zeros(colony.sim.dims) # pattern grid
        # if first:
        #     self.P[self.RR <= r0] = 1
        #     self.tip_loc[0] += 0.5*r0*np.cos(self.theta)
        #     self.tip_loc[1] += 0.5*r0*np.sin(self.theta)
        if type(C0) is not type(None):
            self.C = C0
            self.P[self.C > 0] = 1
        if type(self.colony.Winterp) is not type(None):
            self.Winterp_bool = True
        else:
            self.Winterp_bool = False
        if type(self.colony.Dinterp) is not type(None):
            self.Dinterp_bool = True
        else:
            self.Dinterp_bool = False

    def inc_biomass(self, N, dt):
        # ------------------------------- Handle nutrient uptake into the biomass ----------------------------

        # Compute fN(x,y) across the entire 2D field. 
        fN = N / (N + self.colony.KN) * self.colony.Cm / (self.C + self.colony.Cm) * self.C

        # Compute the nutrient loss due to consumption as a function of space
        dN = -self.colony.bN*fN
        
        # Compute the nutrient loss in space due to consumption
        change_in_N = dN*dt
        
        # Now treat the gain in cell density, this is just an ODE solved via first order Euler explicit. 
        dC = self.colony.aC * fN
        self.C = self.C + dC*dt

        # return only the change so multiple branches can be updated in the same timestep
        return change_in_N

    def update_dl(self, first_timestep, dCdt_branch):
        # --------------------- Change dl according to increase in biomass --------------------
        
        # For the first branch, we just set this value for extension hard-codedly
        if first_timestep:
            dl = 0.5
        else:
            # Now using the differential in biomass, compute how long we should extend the tips of the branches in this system. 
            # Since the system is symmetric with respect to the nutrient concentration, we only need to compute this once. 
            dl = self.colony.gamma*(np.sum(dCdt_branch)*self.colony.sim.d[0]*self.colony.sim.d[1])/self.width

        return dl

    def extend(self, dl, first, N):
        # ---------------------------- Extend the branch by dl ----------------------------------------------

        terminate = False

        location = np.unravel_index(np.argmin((self.colony.sim.XX - self.tip_loc[0])**2 + (self.colony.sim.YY - self.tip_loc[1])**2), self.colony.sim.dims)
        if self.Winterp_bool:
            self.width = np.interp(N[location], self.colony.Winterp[0], self.colony.Winterp[1])

        if not first:
            # Okay, so to predict tip extension we need to predict the direction of growth for each tip. To this end we 
            # proceed in the following procedure. For each tip we consider a local circular region around it with radius
            # equal to dl. We sample ~200 points on this circle at equal intervals and for each point, interpolate
            # the local nutrient concentration. The branch then will extend in the direction of the most nutrient on this circle.
            # Thus we are assuming that branches basically are advected in the direction of \Nab N which is maximal. 
            
            # First we need to generate, for the tip, a sample of points on a circle which are dL from the center. We use
            # the theta0 vector to specify the 200 points on this circle. 
            theta0 = np.pi*np.linspace(-1,1,200)

            # This gives 'candidates' as the points dl from the current tip location
            delta = pol2cart((dl, theta0))
            candidates = np.tile(self.tip_loc, (delta.shape[1], 1)).T + delta
            
            # Now we interpolate the value of the nutrient concentration at each of these points. We do this by first fitting a bivariate
            # spline to the current nutrient concentration and then interpolating at our points of interests. 
            interp = RectBivariateSpline(self.colony.sim.XX[0, :], self.colony.sim.YY[:, 0], N.T)
            
            # Evaluate at desired points
            N_int = interp.ev(candidates[0], candidates[1])
            
            # Now for each agent, and thus each row in the interpolated matrix, we pick the direction which maximizes the nutrient
            # concentration. We thus need to compute the argmax for each row, i.e. the index at which nutrient is maximized. This can
            # be then used to look up the optimal directional angle in the next step. 
            
            # Find the maximal direction on the circle
            armax = np.argmax(N_int, axis = 0)
            
            # Update the growth direction
            self.theta = theta0[armax] # + np.random.randn(0,1)*noise_amp OPTIONALLY ADD NOISE TO GROWTH DIRECTION
            self.tip_loc += dl * np.array([np.cos(self.theta), np.sin(self.theta)])
                
            # Now hard enforce the rule that the branches will terminate growing near the edges, here defined as 85% of the distance from
            # the edge of the simulation
            terminate = np.linalg.norm(self.tip_loc) > 0.85*(self.colony.sim.XX[0,-1] - self.colony.sim.XX[0,0])/2

        else:
            # If it is the first step, grow the branches by the hard-coded initial dl
            self.tip_loc += dl * np.array([np.cos(self.theta), np.sin(self.theta)])

        location = np.unravel_index(np.argmin((self.colony.sim.XX - self.tip_loc[0])**2 + (self.colony.sim.YY - self.tip_loc[1])**2), self.colony.sim.dims)
        if self.Winterp_bool:
            self.width = np.interp(N[location], self.colony.Winterp[0], self.colony.Winterp[1])

        if self.Dinterp_bool:
            self.density = np.interp(N[location], self.colony.Dinterp[0], self.colony.Dinterp[1])
            self.crit_R = 1.5/self.density

        return terminate
    
    def fill_diffuse(self):

        # Lastly fill the width of the branches of the simulations. This is just done by setting all the points within a width/2
        # radius of the tips to 1, indicating they are filled with biofilms.
        d = np.sqrt( (self.colony.sim.XX-self.tip_loc[0])**2 + (self.colony.sim.YY-self.tip_loc[1])**2 )
        self.P[d <= self.width/2] = 1

        # Now simulate the very rapid diffusion of the cell biomass across the branch
        self.C[self.P == 1] = np.sum(self.C)/np.sum(self.P)

class colony:

    def __init__(self, sim, inoc, c0, r0, ntips0, width, density, gamma, bN, aC, KN, Cm, Winterp, Dinterp):
        # -------------------------- Initialize colony variables ------------------------------

        # assign initial condition variables to the colony
        self.inoc = inoc # starting position
        self.c0 = c0 # starting cell number
        self.r0 = r0 # radius of colony initially
        self.width = width # branch width
        self.density = density
        self.crit_R = 1.5 / self.density # distance from other branches at which bifurcation occurs
        self.gamma = gamma # colony expansion efficiency constant
        self.bN = bN # uptake rate of nutrients
        self.aC = aC # energy translation efficiency
        self.KN = KN # half saturation for nutrient uptake kinetics
        self.Cm = Cm # half saturation for cell density in monod model
        self.Winterp = Winterp
        self.Dinterp = Dinterp
        self.sim = sim # the simulation in which the colony is situated

        # initialize the position and cell mass grid 
        self.P = np.zeros(self.sim.dims)
        self.C = np.zeros(self.sim.dims)
        self.RR = np.sqrt((self.sim.XX - self.inoc[0])**2 + (self.sim.YY - self.inoc[1])**2)
        self.P[self.RR < self.r0] = 1 # start with a circular mass of cells
        self.C[self.P == 1] = self.c0/( np.sum(self.P) * self.sim.dims[0] * self.sim.dims[1]) # set cell density in this circular mass
        
        if ntips0 == None:
            ntips0 = np.round(2 * np.pi * self.r0 * self.density) # Calculate the total initial number of tips in the system
            ntips0 = max(ntips0, 2) # Threshold such that we always have 2 tips
            ntips0 = int(ntips0) # Cast to an integer for later usage
        else:
            ntips0 = int(ntips0)

        theta = np.linspace( np.pi/2 , np.pi/2 + 2*np.pi  , ntips0 + 1) # Initial positions will be radially symmetric with respect to the initial innoculation. By
        # symmetry this is the best strategy to ensure that nutrients are well distributed for a uniform background nutrient concentration 
        self.theta = theta[:-1]

        # store the total biomass of the colony
        self.biomass = self.c0*self.sim.d[0]*self.sim.d[1]/(self.sim.dims[0]*self.sim.dims[1])

        # initialize the correct number of branch objects
        initial_tips = np.array([r0*np.cos(self.theta), r0*np.sin(self.theta)]).T
        self.branches = []
        for i in range(ntips0):
            self.branches.append(branch(self, self.width, initial_tips[i], self.density, self.theta[i], self.C/ntips0))

        # store the number of tips in a variable that can be referenced elsewhere
        self.ntips = ntips0

    def inc_biomass(self):
        # ------------------------------- Handle nutrient uptake into the biomass ----------------------------

        # Compute fN(x,y) across the entire 2D field. 
        fN = self.sim.N / (self.sim.N + self.KN) * self.Cm / (self.C + self.Cm) * self.C

        # Compute the nutrient loss due to consumption as a function of space
        dN = -self.bN*fN
        
        # Now treat the gain in cell density, this is just an ODE solved via first order Euler explicit. 
        dC = self.aC * fN

        # return only the change so multiple branches can be updated in the same timestep
        return dN*self.sim.dt, dC*self.sim.dt
    
    def update_branch_biomass(self, dCdt):
        # ------------------------------------- Distribute uptaken biomass across all the branches ------------------------------
        
        dCdt_array = np.zeros((len(self.branches), *self.sim.dims))

        self.C_branch_frac = np.zeros((self.ntips, *self.sim.dims))
        for i in range(len(self.branches)):
            self.C_branch_frac[i] = self.branches[i].C / self.C
            self.C_branch_frac[i][np.isnan(self.C_branch_frac[i])] = 0
            dCdt_array[i] = self.C_branch_frac[i] * dCdt
            self.branches[i].C += dCdt_array[i]

        self.C = np.sum(list(m.C for m in self.branches), axis = 0)

        return dCdt_array

    
    def check_bifurcate(self, dl_list):
        # -------------------------------------- Check whether or not each branch should be split -------------------------------

        # create a variable for the updated branch list
        new_branch_list = self.branches.copy()

        new_branches = []

        # Now we iterate through each individual branch in our system and bifurcate our new branch IF the density criterion is met
        for k in range(len(new_branch_list)): 
            # Create a variable to store the locations of the tips
            branchtips = np.array(list(m.tip_loc for m in new_branch_list))
            
            # Compute the distance of the current tip to all other tips in the system
            dist_sq = np.linalg.norm(branchtips - np.tile(new_branch_list[k].tip_loc, (self.ntips, 1)), axis = 1)
            
            # Sort the distances
            dist_sq = np.sort(dist_sq)

            # Now if the second largest element in the list, i.e. the closest tip, exceeds the distance threshold, then we bifurcate. 
            if dist_sq[1] > new_branch_list[k].crit_R:
                
                # Append a new tip to the list located at 90 degree angles from the bifurcation point
                new_theta = new_branch_list[k].theta + 0.5*np.pi
                new_tip_loc = new_branch_list[k].tip_loc + dl_list[k] * np.array([np.cos(new_theta), np.sin(new_theta)])
                new_branches.append(branch(self,
                                            new_branch_list[k].width,
                                            new_tip_loc,
                                            new_branch_list[k].density,
                                            new_theta,
                                            new_branch_list[k].C/2))
                
                # For the second branch, we just commondere the "current" branch and just have it grow in the 45 degree of the
                # the opposite direction. IE numerically we don't consider this a process of a single branch dying and then
                # two new ones growing from it like a hydra, rather one branch starts to curve and the other branch splits 
                # off from it at the deflection angle of 180 degrees.
                
                update_theta = new_branch_list[k].theta - 0.5*np.pi
                new_branch_list[k].tip_loc += dl_list[k] * np.array([np.cos(update_theta), np.sin(update_theta)])
                new_branch_list[k].C = new_branch_list[k].C/2 

        new_branch_list += new_branches
        self.ntips += len(new_branches)
        
        return new_branch_list

    def update_C(self):
        # ------------------------------------ Update C from the branch cell concentration ------------------------------

        self.C = np.sum(list(b.C for b in self.branches), axis = 0)

class simulation:

    def __init__(self, N0, dims, dt, DN, L, totalT):
        # ------------------------------ Initialize general simulation properties and variables --------------------------

        self.N0 = N0 # initial nutrient concentration
        self.N = np.zeros(dims) + self.N0 # initial nutrient matrix
        self.dims = np.array(dims) # resolution of the environment
        self.L = L # size of the environment
        self.d = self.L/(self.dims-1) # small space step
        self.dt = dt # small time step
        self.DN = DN # Diffusion rate constant of the nutrient through space
        self.totalT = totalT # length of the simulation
        self.nt = self.totalT/self.dt # number of timesteps in the simulation

        # the following lines set up the xy positions of each point in the grid
        x = np.linspace(-self.L/2, self.L/2, self.dims[0])
        y = np.linspace(-self.L/2, self.L/2, self.dims[1])
        self.XX, self.YY = np.meshgrid(x, y)

        # initialize lists to keep track of the colonies and patterns of biomass and nutrients over all timesteps
        self.colonies = []
        self.pattern_store = []
        self.nutrient_store = []
        self.biomass_store = []

        # store the processing times taken
        self.diffusion_times = []
        self.branching_times = []
        self.bifurcation_times = []



        def diff(dx, dy, nx, ny, dt, D):
            # --------------------------------- Maths governing nutrient diffusion -------------------------------
            
            # First compute the assocatiated mu terms in both directions
            mu_x = D*dt/(dx**2)
            mu_y = D*dt/(dy**2)

            # Now compute the central difference operators
            Ix = np.eye(nx)
            Iy = np.eye(ny)

            P = Ix * -2
            P2 = np.eye(nx, k = 1)
            P3 = np.eye(nx, k = -1)

            Q = Iy * -2
            Q2 = np.eye(ny, k = 1)
            Q3 = np.eye(ny, k = -1)

            Mx = P + P2 + P3
            My = Q + Q2 + Q3

            # Impose no flux boundary conditions for the system
            Mx[0,1] = 2
            Mx[nx - 1, nx - 2] = 2
            My[1, 0] = 2
            My[ny - 2, ny - 1] = 2
            
            # Lastly, define the four major operators used to solve our two coupled 1D problems
            V1 = sp.csr_array(Ix - mu_x / 2 * Mx)
            V2 = sp.csr_array(Ix + mu_x / 2 * Mx)
            U2 = sp.csr_array(Iy - mu_y / 2 * My)
            U1 = sp.csr_array(Iy + mu_y / 2 * My)
            
            return V1, V2, U1, U2
        
        # use the above to assign the matrices needed to calculate diffusion of nutrients
        self.V1, self.V2, self.U1, self.U2 = diff(*self.d, *self.dims, self.dt, self.DN)

    def diffuse_nutrients(self):
        # -------------------------------------- Diffuse the nutrients -----------------------------------------

        # Simulate the diffusion of nutrient in space via approximate CN scheme. Recall @ defines matrix-matrix multiplication. 
        Nstar = sp.linalg.spsolve(self.V1, self.N @ self.U1)
        self.N = sp.linalg.spsolve(self.U2.T, (self.V2 @ Nstar).T).T

    def add_colony(self, inoc = (0, 0), c0 = 2000.0, r0 = 5.0, ntips0 = None, width = 2.0, density = 0.2, gamma = 7.5, bN = 160, aC = 1.2, KN = 0.8, Cm = 0.05, Winterp = None, Dinterp = None):
        # ---------------------------------------- Add a colony to the simulation ----------------------------------------

        # set up a colony object and add spaces to the storage lists for the biomass and the pattern
        
        init_dens_range = (self.XX - inoc[0])**2 + (self.YY - inoc[1])**2 < (r0)**2
        density = np.interp(np.average(self.N[init_dens_range]), Dinterp[0], Dinterp[1])
        self.colonies.append(colony(self, inoc, c0, r0, ntips0, width, density, gamma, bN, aC, KN, Cm, Winterp, Dinterp))
        self.biomass_store.append([])
        self.pattern_store.append([])

    def timestep(self, first = False, extend = False):
        # -------------------------------------- Step forward in time by the timestep ------------------------------------

        # set up a variable to track whether the simulation should end or not
        end_sim = False

        start = time.time()

        # Compute the biomass change and the nutrient change due to all the colonies in the simulation,
        # and store this all in an update matrix
        N_update = np.zeros(self.dims)
        if first:
            self.dCdt_arr = list(np.zeros((len(self.colonies[i].branches), *self.dims)) for i in range(len(self.colonies)))
        for i in range(len(self.colonies)):
            N_update_i, dCdt_i = self.colonies[i].inc_biomass()
            N_update += N_update_i
            self.dCdt_arr[i] += self.colonies[i].update_branch_biomass(dCdt_i)
        
        # Add this complete update matrix to the nutrient grid
        self.N = self.N + N_update
        self.N = np.max((self.N, np.zeros(self.dims)), axis = 0)

        # diffuse nutrients across the grid according to the diffusion model
        self.diffuse_nutrients()

        end = time.time()

        # track the time taken for the colonies to uptake nutrients and then the nutrient to diffuse
        self.diffusion_times.append(end - start)
        
        # To reduce the computational power required only calculate whether a branch should be extended or not
        # at specific times through the simulation
        if extend:

            # for each colony, follow these steps:
            # 1. Derive the length that the branches should be extended
            # 2. Check whether each branch needs to split or not
            # 3. Add length to each branch
            # 4. Diffuse nutrient across the grid 
            # 5. Store the biomass and the pattern of the colony in their storage lists
            for i in range(len(self.colonies)):
            
                dl_list = []
                for k in range(len(self.colonies[i].branches)):
                    dl_list.append(self.colonies[i].branches[k].update_dl(first_timestep = first, dCdt_branch = self.dCdt_arr[i][k]))

                start = time.time()
                new_branches = self.colonies[i].check_bifurcate(dl_list)
                end = time.time()
                self.bifurcation_times.append(end - start)

                start = time.time()
                for k in range(len(self.colonies[i].branches)):
                    terminate = self.colonies[i].branches[k].extend(dl_list[k], first, self.N)
                    if terminate:
                        end_sim = True
                end = time.time()
                self.branching_times.append(end - start)

                self.colonies[i].branches = new_branches

                for k in range(len(self.colonies[i].branches)):
                    self.colonies[i].branches[k].fill_diffuse()
                self.colonies[i].update_C()

                self.colonies[i].C = np.sum(list(m.C for m in self.colonies[i].branches), axis = 0)
                self.colonies[i].P = np.array(np.sum(list(m.P for m in self.colonies[i].branches), axis = 0), dtype = bool)
                # Store patterns throughout simulation to generate one final gif
                self.biomass_store[i].append( self.colonies[i].C.copy() )
                self.pattern_store[i].append( self.colonies[i].P.copy() )

            # store the nutrient matrix at each timestep
            self.nutrient_store.append( self.N.copy() )

        if extend:
            self.dCdt_arr = list(np.zeros((len(self.colonies[i].branches), *self.dims)) for i in range(len(self.colonies)))

        return end_sim

    def run_sim(self):
        # ------------------------------------ Run the simulation's time loop ---------------------------------------

        # set up the loop that runs the simulation
        for i in tqdm(range(int(self.nt))):
            end_sim = self.timestep(not bool(i), not bool(i % (0.2/self.dt)))

            # If the colony grows too close to the edges of the grid, stop the simulation
            if end_sim:
                break

    def animate_and_show(self):
        # ------------------ Plot and animate graphs showing how the pattern of the swarm changes over time --------

        # Sum over biomasses and patterns of all the colonies in the simulation to show the pattern of all the biomass
        self.biomass_store = np.sum(self.biomass_store, axis = 0)
        self.pattern_store = np.array(np.sum(self.pattern_store, axis = 0, dtype = bool), dtype = int)
        self.total_masses = np.sum(self.biomass_store, axis = (1,2))

        # set up and plot graphs
        fig, [ax1, ax2, ax3, ax4] = plt.subplots(1,4, figsize = (24,6))
        ax1.set_title("Pattern")
        ax2.set_title("Nutrient Concentration")
        ax3.set_title("Nutrient Cross-section")
        ax3.set_xlim(0, self.dims[0])
        ax3.set_ylim(0,np.max(self.nutrient_store[0]) + 2)
        ax3.set_aspect(np.diff(ax3.get_xlim())[0] / np.diff(ax3.get_ylim())[0])
        ax4.set_xlim( 0, len(self.pattern_store) )
        ax4.set_ylim( 0, self.total_masses.max() )
        ax4.set_title("Total Biomass")
        ax4.set_aspect(np.diff(ax4.get_xlim())[0]/np.diff(ax4.get_ylim())[0])

        # animate the stored time data
        time_series_data = list([] for i in range(0, len(self.pattern_store), 3))
        for i in range(0,len(self.pattern_store),3):
            
            time_series_data[int(i/3)] += [ax1.imshow(self.pattern_store[i], cmap = "viridis"), 
                                    ax2.imshow(self.nutrient_store[i],vmin = 0),
                                    ax3.plot( self.nutrient_store[i][500],)[0],
                                    ax4.plot( range(i) , self.total_masses[0:i] )[0]]

        ani = animation.ArtistAnimation(fig, time_series_data, repeat = False)

        plt.show()

if __name__ == "__main__":

    np.seterr(invalid = 'ignore')
    
    f = sio.loadmat('./NNdata/Parameters_gradient_Figure5.mat')
    f2 = sio.loadmat('./NNdata/Parameters_multiseeding.mat')
    Dinterp = (f['mapping_N'][0], f['mapping_optimD'][0])
    Winterp = (f['mapping_N'][0], f['mapping_optimW'][0])

    master_sim = simulation(N0 = np.broadcast_to(np.linspace(8, 16, 1001), (1001, 1001)), dims = (1001, 1001), dt = 0.02, DN = f['DN'], L = 90, totalT = 48)
    # master_sim.add_colony(inoc = (15, 0), Winterp = Winterp, Dinterp = Dinterp, Cm = f['Cm'][0,0])
    # master_sim.add_colony(inoc = (-15, 0), Winterp = Winterp, Dinterp = Dinterp)
    master_sim.add_colony(ntips0 = 8, Winterp = Winterp, Dinterp = Dinterp, Cm = f['Cm'][0,0], bN = f['bN'][0,0], gamma = f['gama'][0,0], aC = f['aC'][0,0], KN = f['KN'][0,0])
    master_sim.run_sim()
    master_sim.animate_and_show()