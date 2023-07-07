import numpy as np
import time
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation

class colony:

    def __init__(self, grid, inoc, c0, r0, width, density, gamma, bN, aC, KN, Cm):

        # assign initial condition variables to the colony
        self.inoc = inoc
        self.c0 = c0
        self.r0 = r0
        self.width = width
        self.density = density
        self.crit_R = 1.5 / self.density
        self.gamma = gamma
        self.bN = bN
        self.aC = aC
        self.KN = KN
        self.Cm = Cm
        self.XX, self.YY = grid
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

        self.biomass = np.sum(self.C*self.dx*self.dy)

        theta = np.linspace( np.pi/2 , np.pi/2 + 2*np.pi  , ntips0 + 1) # Initial positions will be radially symmetric with respect to the initial innoculation. By
        # symmetry this is the best strategy to ensure that nutrients are well distributed for a uniform background nutrient concentration 
        self.theta = theta[:-1]
        
        self.rX = 0.5*self.r0*np.cos(self.theta) + self.inoc[0]
        self.rY = 0.5*self.r0*np.sin(self.theta) + self.inoc[1]

        self.ntips = ntips0

    def inc_biomass(self, N, dt):
    
        # Compute fN(x,y) across the entire 2D field. 
        fN = N / (N + self.KN) * self.Cm / (self.C + self.Cm) * self.C
        
        # Compute the nutrient loss due to consumption as a function of space
        dN = -self.bN*fN
        
        # Compute the nutrient loss in space due to consumption 
        # N = N + dN*dt
        change_in_N = dN*dt
        
        # Now treat the gain in cell density, this is just an ODE solved via first order Euler explicit. 
        dC = self.aC * fN
        self.C = self.C + dC*dt

        return change_in_N

    def update_dl(self, first_timestep = False):

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
                
                # print("BIFURCATING!")
                
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
        
        self.rX = rX_new
        self.rY = rY_new
        self.theta = theta_new
        self.ntips = self.rX.shape[0]

    def branch_extend(self, dl, N, first_timestep = False):

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
        self.N0 = N0
        self.N = np.zeros(dims) + self.N0
        self.dims = np.array(dims)
        self.L = L
        self.d = self.L/(self.dims-1)
        self.dt = dt
        self.DN = DN
        self.totalT = totalT
        self.nt = self.totalT/self.dt
        x = np.linspace(-self.L/2, self.L/2, self.dims[0])
        y = np.linspace(-self.L/2, self.L/2, self.dims[1])
        self.XX, self.YY = np.meshgrid(x, y)
        self.colonies = []
        self.diffusion_times = []
        self.branching_times = []
        self.bifurcation_times = []

        self.pattern_store = []
        self.nutrient_store = []
        self.biomass_store = []

        def diff(dx, dy, nx, ny, dt, D):
            
            # First compute the assocatiated mu terms in both directions
            mu_x = D*dt/(dx**2)
            mu_y = D*dt/(dy**2)

            # Now compute the central difference operators
            Ix = np.eye(nx)
            Iy = np.eye(ny)

            P = np.diagflat(np.ones(nx)) * -2
            P2 = np.eye(nx, k = 1)
            P3 = np.eye(nx, k = -1)

            Q = np.diagflat(np.ones(ny)) * -2
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
            V1 = Ix - mu_x / 2 * Mx
            V2 = Ix + mu_x / 2 * Mx
            U2 = Iy - mu_y / 2 * My
            U1 = Iy + mu_y / 2 * My 
            
            return V1, V2, U1, U2

        self.V1, self.V2, self.U1, self.U2 = diff(*self.d, *self.dims, self.dt, self.DN)

    def diffuse_nutrients(self):

        # Simulate the diffusion of nutrient in space via approximate CN scheme. Recall @ defines matrix-matrix multiplication. 
        Nstar = np.linalg.inv( self.V1 ) @ ( self.N @ self.U1 ) # Solve equation one to get an intermediate solution
        self.N = ( self.V2 @ Nstar ) @ np.linalg.inv( self.U2 ) # Solve equation two to get the final update

    def add_colony(self, inoc = (0, 0), c0 = 2000.0, r0 = 5.0, width = 2.0, density = 0.2, gamma = 7.5, bN = 160, aC = 1.2, KN = 0.8, Cm = 0.05):

        self.colonies.append(colony((self.XX, self.YY), inoc, c0, r0, width, density, gamma, bN, aC, KN, Cm))
        self.biomass_store.append([])
        self.pattern_store.append([])

    def timestep(self, first = False, extend = False):
        
        end_sim = False

        start = time.time()
        N_update = np.zeros(self.dims)
        for i in range(len(self.colonies)):
            N_update += self.colonies[i].inc_biomass(self.N, self.dt)
        self.N = self.N + N_update
        self.diffuse_nutrients()
        end = time.time()
        self.diffusion_times.append(end - start)

        if extend:
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
            self.nutrient_store.append( self.N.copy() )

        return end_sim

    def run_sim(self):

        for i in tqdm(range(int(self.nt))):
            end_sim = self.timestep(not bool(i), not bool(i % (0.2/self.dt)))
            if end_sim:
                break

    def animate_and_show(self):

        self.biomass_store = np.sum(self.biomass_store, axis = 0)
        self.pattern_store = np.array(np.sum(self.pattern_store, axis = 0, dtype = bool), dtype = int)
        self.total_masses = np.sum(self.biomass_store, axis = (1,2))

        fig, [ax1, ax2, ax3, ax4] = plt.subplots(1,4, figsize = (24,6))
        ax1.set_title("Pattern")
        ax2.set_title("Nutrient Concentration")
        ax3.set_title("Nutrient Crosssection")
        ax3.set_ylim(0,10.0)
        ax4.set_xlim( 0, len(self.pattern_store) )
        ax4.set_ylim( 0, self.total_masses.max() )
        ax4.set_title("Total Biomass")

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
    master_sim.add_colony(inoc = (15, -10))
    master_sim.add_colony(inoc = (-15, -10))
    master_sim.add_colony(inoc = (0, 20))
    master_sim.run_sim()
    master_sim.animate_and_show()


    

    