import numpy as np
import time
from scipy.interpolate import RectBivariateSpline
<<<<<<< HEAD
from scipy.sparse.linalg import spsolve
=======
import scipy.sparse as sp
>>>>>>> adrian_sparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation

class colony:

    def __init__(self, grid, inoc, c0, r0, width, density, gamma, bN, aC, KN, Cm):
        # -------------------------- Initialize colony variables ------------------------------

        # assign initial condition variables to the colony
        self.inoc = inoc # starting position
        self.c0 = c0 # starting cell number
        self.r0 = r0 # radius of colony initially
        self.width = width # branch width
        self.density = density # branch density to be maintained
        self.crit_R = 1.5 / self.density # distance from other branches at which bifurcation occurs
        self.gamma = gamma # colony expansion efficiency constant
        self.bN = bN # uptake rate of nutrients
        self.aC = aC # energy translation efficiency
        self.KN = KN # half saturation for nutrient uptake kinetics
        self.Cm = Cm # half saturation for cell density in monod model
        self.XX, self.YY = grid # the underlying grid on which the colony sits 
        # create more dimension variables
        self.nx, self.ny = self.XX.shape[0], self.YY.shape[0]
        self.dx, self.dy = (self.XX[0,-1] - self.XX[0,0])/(self.nx - 1), (self.YY[-1,0] - self.YY[0,0])/(self.ny - 1)

        # initialize the position and cell mass grid 
        self.P = np.zeros((self.nx, self.ny))
        self.C = np.zeros((self.nx, self.ny))
        self.RR = np.sqrt((self.XX - self.inoc[0])**2 + (self.YY - self.inoc[1])**2)
        self.P[self.RR < self.r0] = 1 # start with a circular mass of cells
        self.C[self.P == 1] = self.c0/( np.sum(self.P) * self.nx * self.ny) # set cell density in this circular mass

        ntips0 = np.round(2 * np.pi * self.r0 * self.density) # Calculate the total initial number of tips in the system
        ntips0 = max(ntips0, 2) # Threshold such that we always have 2 tips
        ntips0 = int(ntips0) # Cast to an integer for later usage

        # these store the positions of the tips of each branch
        self.rX = np.zeros(ntips0)
        self.rY = np.zeros(ntips0)

        # store the total biomass of the colony
        self.biomass = np.sum(self.C*self.dx*self.dy)

        theta = np.linspace( np.pi/2 , np.pi/2 + 2*np.pi  , ntips0 + 1) # Initial positions will be radially symmetric with respect to the initial innoculation. By
        # symmetry this is the best strategy to ensure that nutrients are well distributed for a uniform background nutrient concentration 
        self.theta = theta[:-1]
        
        # place the initial positions of these branch tips
        self.rX = 0.5*self.r0*np.cos(self.theta) + self.inoc[0]
        self.rY = 0.5*self.r0*np.sin(self.theta) + self.inoc[1]

        # store the number of tips in a variable that can be referenced elsewhere
        self.ntips = ntips0

        self.base = np.ones((self.nx, self.ny))

    def inc_biomass(self, N, dt):
        # ------------------------------- Handle nutrient uptake into the biomass ----------------------------

        # Compute fN(x,y) across the entire 2D field. 
        fN = N / (N + self.KN) * self.Cm / (self.C + self.Cm) * self.C

        # Compute the nutrient loss due to consumption as a function of space
        dN = -self.bN*fN
        
        # Compute the nutrient loss in space due to consumption
        change_in_N = dN*dt
        
        # Now treat the gain in cell density, this is just an ODE solved via first order Euler explicit. 
        dC = self.aC * fN
        self.C = self.C + dC*dt

        # return only the change so multiple colonies can be updated in the same timestep
        return change_in_N

    def update_dl(self, first_timestep = False):
        # -------------------------------------- Calculate how much to extend the branches ----------------------------------

        # Store the current biomass in the system
        biomass_pre = self.biomass
        
        # Compute the new biomass in the system at this time point by integrating the cell density over the entire domain. 
        self.biomass = np.sum(self.C)*self.dx*self.dy
        
        # Now using the differential in biomass, compute how long we should extend the tips of the branches in this system. 
        # Since the system is symmetric with respect to the nutrient concentration, we only need to compute this once. 
        dl = self.gamma*(self.biomass - biomass_pre)/(self.width*self.ntips)
        
        # For the first branch, we just set this value for extension hard-codedly
        if first_timestep:
            dl = 0.5

        return dl
    
    def check_bifurcate(self, dl):
        # -------------------------------------- Check whether or not the branch should be split -------------------------------

        # Create some varialbes to store the new locations of the tips 
        rX_new = self.rX
        rY_new = self.rY
        theta_new = self.theta

        # Now we iterate through each individual branch in our system and bifurcate our new branch IF the density criterion is met
        for k in range(self.ntips):
            
            # Compute the distance of the current tip to all other tips in the system
            dist_sq = (rX_new - self.rX[k])**2 + (rY_new - self.rY[k])**2
            
            # Sort the distances
            dist_sq = np.sort(dist_sq)
            
            # Now if the second largest element in the list, i.e. the closest tip, exceeds the distance threshold, then we bifurcate. 
            if dist_sq[1] > self.crit_R**2:
                
                # Append a new tip to the list located at 45 degree angles from the bifurcation point
                rX_new = np.append(rX_new, self.rX[k] + dl*np.sin(self.theta[k] + 0.5 * np.pi))
                rY_new = np.append(rY_new, self.rY[k] + dl*np.cos(self.theta[k] + 0.5 * np.pi))
                
                # For the second branch, we just commondere the "current" branch and just have it grow in the 45 degree of the
                # the opposite direction. IE numerically we don't consider this a process of a single branch dying and then
                # two new ones growing from it like a hydra, rather one branch starts to curve and the other branch splits 
                # off from it at the deflection angle of 90 degrees.

                rX_new[k] = rX_new[k] + dl*np.sin(self.theta[k] - 0.5*np.pi)
                rY_new[k] = rY_new[k] + dl*np.cos(self.theta[k] - 0.5*np.pi)
                
                # Lastly, append a new angle to the theta vector which corresponds to the new branch that we added. We will
                # for now use just use a dummy value, we'll go ahead and set the bifurcation direction in the later steps 
                # of the numerics. 
                theta_new = np.append(theta_new, self.theta[k])
        
        # update the colony variables with new values
        self.rX = rX_new
        self.rY = rY_new
        self.theta = theta_new
        self.ntips = self.rX.shape[0]

    def branch_extend(self, dl, N, first_timestep = False):
        
        # initialize a tracker to keep track of whether branches are too close to the edges or not
        terminated_idx = np.zeros(self.ntips, dtype = bool)
        
        # Store a previous version of the agent location before update; we'll use this to handle terminaing the growth 
        # of a branch at maximal distance. 
        rX_pre = self.rX
        rY_pre = self.rY
        
        # If this is the first step, just grow the branches from the current position by the hard-coded initial dl
        if first_timestep:
            self.rX = self.rX + dl * np.sin(self.theta)
            self.rY = self.rY + dl * np.cos(self.theta)

        else:
            
            # Okay, so to predict tip extension we need to predict the direction of growth for each tip. To this end we 
            # proceed in the following procedure. For each tip we consider a local circular region around it with radius
            # equal to dl. We sample ~200 points on this circle at equal intervals and for each point, interpolate
            # the local nutrient concentration. The branch then will extend in the direction of the most nutrient on this circle.
            # Thus we are assuming that branches basically are advected in the direction of \Nab N which is maximal. 
            
            # First we need to generate, for each tip, a sample of points on a cirlce which are dL from the center. We use
            # the delta vector to specify the 200 points on this circle. 
            delta = np.pi*np.linspace(-1,1,200)
            
            # Now for each tip we create a matrix which is ntips x delta.shape[0], where each row corresponds to 
            # angle samples around the point.
            theta0 = np.tile(delta, (self.ntips, 1))
            
            # Now consider the points on a circle dl away radiating out from our delta angles
            x_candidates = self.rX + dl*np.sin(theta0).T
            y_candidates = self.rY + dl*np.cos(theta0).T
            
            # Now we interpolate the value of the nturient concentration at each of these points. We do this by first fitting a bivariate
            # spline to the current nutrient concentration and then interpolating at our points of interests. 
            interp = RectBivariateSpline(self.XX[0, :], self.YY[:, 0], N.T)
            
            # Evaluate at desired points
            N_int = interp.ev(x_candidates, y_candidates)
            
            # Now for each agent, and thus each row in the interpolated matrix, we pick the direction which maximizes the nutrient
            # concentration. We thus need to compute the argmax for each row, i.e. the index at which nutrient is maximized. This can
            # be then used to look up the optimal directional angle in the next step. 
            
            # Find the maximal direction on the circle
            maxes = np.argmax(N_int, axis = 0)
            
            # Update the length of each tip as well as update the growth direction
            for k in range(self.ntips):
                self.rX[k] = x_candidates.T[k, maxes[k]]
                self.rY[k] = y_candidates.T[k, maxes[k]]
        
                self.theta[k] = theta0[k, maxes[k]] # + np.random.randn(0,1)*noise_amp OPTIONALLY ADD NOISE TO GROWTH DIRECTION
                
            # Now hard enforce the rule that the branches will terminate growing near the edges, here defined as 85% of the distance from
            # the edge of the simulation
            terminated_idx = np.sqrt( self.rX**2 + self.rY**2 ) > 0.85*(self.XX[0,-1] - self.XX[0,0])/2
                
            for k in range(self.ntips):
                if terminated_idx[k]:
                    self.rX[k] = rX_pre[k]
                    self.rY[k] = rY_pre[k]

        return terminated_idx
    
    def fill_diffuse(self):

        # Lastly fill the width of the branches of the simulations. This is just done by setting all the points within a width/2
        # radius of the tips to 1, indicating their are filled with biofilms.
        for k in range(self.ntips):
            d = np.sqrt( (self.XX-self.rX[k])**2 + (self.YY-self.rY[k])**2 )
            self.P[d <= self.width/2] = 1

        # Now simulate the very rapid diffusion of the cell biomass across the pattern
        self.C[self.P == 1] = self.biomass/( np.sum(self.P) * self.dx * self.dy)

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

<<<<<<< HEAD
        # Simulate the diffusion of nutrient in space via approximate CN scheme. Recall @ defines matrix-matrix multiplication.
        Nstar = np.linalg.inv( self.V1 ) @ ( self.N @ self.U1 ) # Solve equation one to get an intermediate solution
        self.N = ( self.V2 @ Nstar ) @ np.linalg.inv( self.U2 ) # Solve equation two to get the final update
=======
        # Simulate the diffusion of nutrient in space via approximate CN scheme. Recall @ defines matrix-matrix multiplication. 
        Nstar = sp.linalg.spsolve(self.V1, self.N @ self.U1)
        self.N = sp.linalg.spsolve(self.U2.T, (self.V2 @ Nstar).T).T
>>>>>>> adrian_sparse

    def add_colony(self, inoc = (0, 0), c0 = 2000.0, r0 = 5.0, width = 2.0, density = 0.2, gamma = 7.5, bN = 160, aC = 1.2, KN = 0.8, Cm = 0.05):
        # ---------------------------------------- Add a colony to the simulation ----------------------------------------

        # set up a colony object and add spaces to the storage lists for the biomass and the pattern
        self.colonies.append(colony((self.XX, self.YY), inoc, c0, r0, width, density, gamma, bN, aC, KN, Cm))
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
        for i in range(len(self.colonies)):
            N_update += self.colonies[i].inc_biomass(self.N, self.dt)
        
        # Add this complete update matrix to the nutrient grid
        self.N = self.N + N_update

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
                dl = self.colonies[i].update_dl(first_timestep = first)

                start = time.time()
                self.colonies[i].check_bifurcate(dl)
                end = time.time()
                self.bifurcation_times.append(end - start)

                start = time.time()
                terminated = self.colonies[i].branch_extend(dl, self.N, first_timestep = first)
                if terminated.any():
                    end_sim = True
                end = time.time()
                self.branching_times.append(end - start)

                self.colonies[i].fill_diffuse()

                # Store patterns throughout simulation to generate one final gif
                self.biomass_store[i].append( self.colonies[i].C.copy() )
                self.pattern_store[i].append( self.colonies[i].P.copy() )

            # store the nutrient matrix at each timestep
            self.nutrient_store.append( self.N.copy() )

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
        ax3.set_ylim(0,10.0)
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

    master_sim = simulation(N0 = 8, dims = (1000, 1000), dt = 0.02, DN = 9, L = 90, totalT = 48)
<<<<<<< HEAD
    master_sim.add_colony(inoc = (15, 0))
    master_sim.add_colony(inoc = (-15, 0))
=======
    # master_sim.add_colony(inoc = (15, 0))
    # master_sim.add_colony(inoc = (-15, 0))
    master_sim.add_colony()
>>>>>>> adrian_sparse
    master_sim.run_sim()
    master_sim.animate_and_show()


    

    