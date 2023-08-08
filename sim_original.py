import numpy as np
import scipy.interpolate as sint
import scipy.sparse as sp
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import animation
import time

def main(config, chem): # input seeding configuration

    def diff(dx, dy, nx, ny, dt, D):
        # --------------------------------- Maths governing nutrient diffusion -------------------------------
        
        # First compute the assocatiated mu terms in both directions
        mu_x = D*dt/(dx**2)
        mu_y = D*dt/(dy**2)

        # Now compute the central difference operators
        Ix = np.eye(nx)
        Iy = np.eye(ny)

        P1 = Ix * -2
        P2 = np.eye(nx, k = 1)
        P3 = np.eye(nx, k = -1)

        Q = Iy * -2
        Q2 = np.eye(ny, k = 1)
        Q3 = np.eye(ny, k = -1)

        Mx = P1 + P2 + P3
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

    DN = 5.74856944
    DT = 5.74856944*10
    KN = 0.6634879
    KT = 0.6634879
    Cm = 0.07888089
    bN = 195.48320435
    aC = 1.1050198
    gama = 4
    bT = 900
    aT = 1
    T_c = 1
    cT = 100
    chemEffect = -1
    if not chem:
        chemEffect = 0
    N0s = np.array((8.5, 14.5, 16.5))
    mapping_N = np.array([4, 8, 12, 16, 20])
    mapping_optimW = np.array((1.5, 2, 2.5, 10, 11.11111111))
    mapping_optimD = np.array((0.08, 0.13, 0.17, 0.08, 0.07))

    ld = sio.loadmat('./NNdata/DUKE.mat')
    NutrientLevel = 1 # select nutrient level: 0-low, 1-medium, 2-high

    # Obtain optimal W & D from the mapping
    N0      = N0s[NutrientLevel]
    Width   = np.interp(N0, mapping_N, mapping_optimW)
    Density = np.interp(N0, mapping_N, mapping_optimD)
    
    # ------------------------ Seeding configurations -------------------------
    match config:
        case 1: x0 = np.array((0,)); y0 = np.array((0,)) # one dot
        case 2: x0 = 17/2 * np.array([-1, 1]); y0 = np.array([0, 0]) # two dots side by side
        case 3: x0 = 38/2 * np.array([-1, 1]); y0 = np.array([0, 0]) # two dots side by side
        case 4: x0 = 19 * np.array([-1, 0, 1]); y0 = np.array([0, 0, 0]) # three dots side by side
        case 5: x0 = 10 * np.array([0, np.sqrt(3)/2, -np.sqrt(3)/2]); y0 = 10 * np.array([1, -0.5, -0.5]) # triangular
        case 6: x0 = 20 * np.array([0, np.sqrt(3)/2, -np.sqrt(3)/2]); y0 = 20 * np.array([1, -0.5, -0.5]) # triangular
        case 7: x0 = 15 * np.array([-1, 1, 1, -1]); y0 = 15 * np.array([1, 1, -1, -1]) # square
        case 8: 
            x0 = 19 * np.array([0, 0.5, 1, 0.5, -0.5, -1, -0.5]) # core-ring
            y0 = 19 * np.array([0, np.sqrt(3)/2, 0, -np.sqrt(3)/2, -np.sqrt(3)/2, 0, np.sqrt(3)/2])
        case 9: 
            x0 = 19 * np.array([0, np.sqrt(2)/2, 1, np.sqrt(2)/2, 0, -np.sqrt(2)/2, -1, -np.sqrt(2)/2]) # ring
            y0 = 19 * np.array([1, np.sqrt(2)/2, 0, -np.sqrt(2)/2, -1, -np.sqrt(2)/2, 0, np.sqrt(2)/2])
        case 10:
            x0 = 19 * np.array([0, 0.3827, np.sqrt(2)/2, 0.9239, 1, 0.9239, np.sqrt(2)/2, 0.3827, 0, -0.3827, -np.sqrt(2)/2, -0.9239, -1, -0.9239, -np.sqrt(2)/2, -0.3827]) # ring
            y0 = 19 * np.array([1, 0.9239, np.sqrt(2)/2, 0.3827, 0, -0.3827, -np.sqrt(2)/2, -0.9239, -1, -0.9239, -np.sqrt(2)/2, -0.3827, 0, 0.3827, np.sqrt(2)/2, 0.9239])
        case 11:x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0]); y0 = 6 * np.array([0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5]) # line
        case 12: Pattern = ld['D']
        case 13: Pattern = ld['U']
        case 14: Pattern = ld['K']
        case 15: Pattern = ld['E']
        case _: x0 = None; Pattern = None


    if config >= 12 and config <= 15:
        Pattern = Pattern[::-1]
        row, col = np.nonzero(Pattern > 0)
        row = row - (Pattern.shape[0] + 1) / 2
        col = col - (Pattern.shape[1] + 1) / 2
        domainsize = 42
        L = 90
        x0 = col * L / domainsize
        y0 = row * L / domainsize

    # -------------------------------------------------------------------------

    # Parameters
    L      = 90
    totalt = 24

    dt = 0.02
    nt = totalt / dt
    global dims
    dims = np.array((1001, 1001))
    d = np.array((L/(dims[0] - 1), L/(dims[1] - 1)))
    x  = np.linspace(-L/2, L/2, dims[0])
    y  = np.linspace(-L/2, L/2, dims[1])
    [xx, yy] = np.meshgrid(x, y)

    noiseamp = 0 * np.pi
    nseeding = x0.shape[0]

    # Initialization
    C = np.zeros(dims)      # Cell density
    N = np.zeros(dims) + N0
    Tox = np.zeros((nseeding, *dims))
    if chem:
        gC = np.zeros((*dims, nseeding))
    else:
        gC = 0
    r0 = 5    # initial radius 
    C0 = 1.6

    # calculate the actual length of boundary of each inoculum
    nseg = 50; seglength = 2 * np.pi * r0 / nseg
    theta = np.linspace(0, 2 * np.pi, nseg + 1); theta = theta[:-1]
    colonyarray = [] # boundary of each colony
    for iseed in range(nseeding):
        colony = np.array((r0 * np.sin(theta) + x0[iseed], r0 * np.cos(theta) + y0[iseed]))
        colonyarray.append(colony)
    boundarylengths = np.zeros((nseeding,))
    for iseed in range(nseeding):
        boundarylengths[iseed] = seglength * colonyarray[iseed].shape[1]

    # ------------------------------------------------------------------------

    ntips0 = np.ceil(boundarylengths * Density * 2) # initial branch number
    ntips0 = np.array(ntips0, dtype = int)
    ColonyDomain = np.zeros((*dims, nseeding), dtype = int) # the domain covered by each colony
    BranchColonyID = np.concatenate((list(i * np.ones((ntips0[i],), dtype = int) for i in range(nseeding))))

    Width = np.tile(Width, np.sum(ntips0))
    Density = np.tile(Density, np.sum(ntips0))
    rr = np.zeros((*dims, nseeding))
    theta = []; Tipx = []; Tipy = []
    for iseed in range(nseeding):

        rr[:,:,iseed] = np.sqrt((xx - x0[iseed]) ** 2 + (yy - y0[iseed]) ** 2)
        ColonyDomain[:, :, iseed] = (rr[:, :, iseed] <= r0)

        Tipxi = np.ones((ntips0[iseed],)) * x0[iseed];  Tipx = np.concatenate((Tipx, Tipxi)) # x coordinates of every tip
        Tipyi = np.ones((ntips0[iseed],)) * y0[iseed];  Tipy = np.concatenate((Tipy, Tipyi)) # y coordinates of every tip
        thetai = np.linspace(np.pi/2, 2 * np.pi+np.pi/2, ntips0[iseed] + 1) 
        thetai = thetai[:-1] + iseed /50 * np.pi # growth directions of every branch
        theta = np.concatenate((theta, thetai))

    rr = np.min(rr, axis = 2)
    C[np.sum(ColonyDomain, axis = 2) == 1] = C0 / (np.sum(ColonyDomain) * d[0] * d[1]); C_pre = C
    Tox_pre = Tox

    ntips0 = np.sum(ntips0)

    BranchDomain = np.zeros((*dims, ntips0), dtype = int) # the domain covered by each branch
    for k in range(ntips0): BranchDomain[:, :, k] = ColonyDomain[:, :, BranchColonyID[k]]

    delta = np.linspace(-1, 1, 201) * np.pi
    MatV1N, MatV2N, MatU1N, MatU2N = diff(*d, *dims, dt, DN)
    MatV1T, MatV2T, MatU1T, MatU2T = diff(*d, *dims, dt, DT)

    biomass_store = []
    pattern_store = []
    nutrient_store = []
    toxin_store = []

    for i in tqdm(range(int(nt))):
        
        # -------------------------------------
        # Nutrient distribution and cell growth
        
        fN = N / (N + KN) * Cm / (C + Cm) * C
        dN = - bN * fN
        N  = N + dN * dt 
        NV = sp.linalg.spsolve(MatV1N, N @ MatU1N); N = sp.linalg.spsolve(MatU2N.T, (MatV2N @ NV).T).T
        dC = aC * fN * (1 - gC)

        if chem:

            dTox = np.zeros((nseeding, *dims))

            for j in range(nseeding):
                fTother = Tox[~(np.arange(nseeding) == j)] / (Tox[~(np.arange(nseeding) == j)] + KT) * Cm / (np.tile(C, (nseeding - 1, 1, 1)) + Cm) * np.tile(C, (nseeding - 1, 1, 1))
                dTox[j] = dTox[j] + bT * fN * ColonyDomain[:, :, j] * fTother.sum(0)
                dTox[~(np.arange(nseeding) == j)] += - cT * fTother
                dC += ColonyDomain[:, :, j] * (np.sum(fTother, axis = 0) * aT * chemEffect + aC * fN * (1 - fTother.sum(0)))
            
            for j in range(nseeding):
                Tox[j] = Tox[j] + dTox[j] * dt  
                ToxiV = sp.linalg.spsolve(MatV1T, Tox[j] @ MatU1T); Tox[j] = sp.linalg.spsolve(MatU2T.T, (MatV2T @ ToxiV).T).T
        
        C  = C + dC * dt

        # -------------------------------------
        # Branch extension and bifurcation
        ntips = Tipx.shape[0]
        
        if not bool(i % (0.2/dt)):
        
            dBiomass = (C - C_pre) * d[0] * d[1] 
            # compute the amount of biomass accumulation in each branch
            BranchDomainSum = np.sum(BranchDomain, axis = 2)
            ntips = Tipx.shape[0]
            dE = np.zeros((ntips,))
            for k in range(ntips):
                with np.errstate(divide = 'ignore'):
                    branchfract = 1 / (BranchDomainSum * BranchDomain[:, :, k])
                branchfract[np.isinf(branchfract)] = 0
                dE[k] = np.sum(dBiomass * branchfract)        
                # extension rate of each branch
                dl = gama * dE / Width
            if i == 0: dl = r0 - Width/2

            # Bifurcation
            R = 1.5 / Density  # a branch will bifurcate if there is no other branch tips within the radius of R
            TipxNew = Tipx; TipyNew = Tipy; thetaNew = theta; dlNew = dl; WidthNew = Width; DensityNew = Density
            BranchDomainNew = BranchDomain; BranchColonyIDNew = BranchColonyID
            for k in range(ntips):
                dist2othertips = np.sqrt((TipxNew - Tipx[k]) ** 2 + (TipyNew - Tipy[k]) ** 2)
                dist2othertips = np.sort(dist2othertips)
                if dist2othertips[1] > R[k]:
                    TipxNew = np.append(TipxNew, Tipx[k] + dl[k] * np.sin(theta[k] + 0.5 * np.pi)) # splitting the old tip to two new tips
                    TipyNew = np.append(TipyNew, Tipy[k] + dl[k] * np.cos(theta[k] + 0.5 * np.pi)) 
                    TipxNew[k] = TipxNew[k] + dl[k] * np.sin(theta[k] - 0.5 * np.pi)
                    TipyNew[k] = TipyNew[k] + dl[k] * np.cos(theta[k] - 0.5 * np.pi)
                    dlNew = np.append(dlNew, dl[k] / 2)
                    dlNew[k] = dl[k] / 2
                    thetaNew = np.append(thetaNew, theta[k])
                    BranchDomainNew = np.append(BranchDomainNew, BranchDomain[:, :, k].reshape((*dims, 1)), axis = 2)
                    BranchColonyIDNew = np.append(BranchColonyIDNew, BranchColonyID[k])
                    WidthNew = np.append(WidthNew, Width[k])
                    DensityNew = np.append(DensityNew, Density[k])
            Tipx = TipxNew; Tipy = TipyNew; theta = thetaNew; dl = dlNew; Width = WidthNew; Density = DensityNew
            BranchDomain = BranchDomainNew; BranchColonyID = BranchColonyIDNew

            ntips = Tipx.shape[0]
            # Determine branch extension directions
            Tipx_pre = Tipx; Tipy_pre = Tipy
            if i == 0:
                Tipx = Tipx + dl * np.sin(theta)
                Tipy = Tipy + dl * np.cos(theta)
            else:
                thetaO = np.ones((ntips, delta.shape[0])) * delta
                TipxO = np.tile(Tipx, (delta.shape[0], 1)).T + np.tile(dl, (delta.shape[0], 1)).T * np.sin(thetaO)
                TipyO = np.tile(Tipy, (delta.shape[0], 1)).T + np.tile(dl, (delta.shape[0], 1)).T * np.cos(thetaO)
                DirMatO = np.zeros((ntips, delta.shape[0]))
                if chem:
                    for j in range(nseeding):
                        DirMat = N + T_c * chemEffect * np.sum(Tox[~(np.arange(nseeding) == j)], axis = 0)
                        interp = sint.RectBivariateSpline(x, y, DirMat.T)
                        DirMatO[BranchColonyID == j] = interp.ev(TipxO[BranchColonyID == j], TipyO[BranchColonyID == j])
                else:
                    interp = sint.RectBivariateSpline(x, y, N.T)
                    DirMatO = interp.ev(TipxO, TipyO)
                ind = np.argmax(DirMatO, axis = 1) # find the direction with maximum nutrient
                for k in range(ntips):
                    Tipx[k] = TipxO[k, ind[k]]
                    Tipy[k] = TipyO[k, ind[k]]
                    theta[k] = thetaO[k, ind[k]]
                    Width[k] = np.interp(DirMatO[k, ind[k]], mapping_N, mapping_optimW)
                    Density[k] = np.interp(DirMatO[k, ind[k]], mapping_N, mapping_optimD)

            # Growth stops when approaching edges
            ind = np.sqrt(Tipx ** 2 + Tipy ** 2) > 0.8 * L/2
            Tipx[ind] = Tipx_pre[ind]
            Tipy[ind] = Tipy_pre[ind]

            # Fill the width of the branches
            for k in range(ntips):
                dist = np.sqrt((Tipx[k] - xx) ** 2 + (Tipy[k] - yy) ** 2)
                BranchDomain[:, :, k] = np.bitwise_or(BranchDomain[:, :, k], dist <= Width[k]/2)
                ColonyDomain[:, :, BranchColonyID[k]] = np.bitwise_or(ColonyDomain[:, :, BranchColonyID[k]], dist <= Width[k]/2)
            ColonyDomainSum = np.sum(ColonyDomain, axis = 2)
            for j in range(nseeding):
                with np.errstate(divide = 'ignore'):
                    colonyfract = 1 / (ColonyDomainSum * ColonyDomain[:, :, j])
                colonyfract[np.isinf(colonyfract)] = 0
                C[ColonyDomain[:, :, j] == 1] = np.sum(C * colonyfract)/np.sum(ColonyDomain[:, :, j]) # Make cell density uniform

            C_pre = C

            biomass_store.append( C.copy() )
            pattern_store.append( np.sum(ColonyDomain*(np.arange(nseeding) + 1), axis = 2) )
            nutrient_store.append( N.copy() )
            if chem:
                toxin_store.append( Tox.copy() )

    return biomass_store, pattern_store, nutrient_store, toxin_store

if __name__ == '__main__':

    colors = ['#a1dab4', '#886ac4', '#60a240', '#2a406b', '#341f14', '#000000'][::-1]
    my_cmap = ListedColormap(colors, name="my_cmap")

    only_show_pattern = True

    if only_show_pattern:
        save = True

        chem = False

        # fig, axs = plt.subplots(2, 7, figsize = (21, 6))
        # for i in range(14):
        #     _, pattern_store, _, _ = main(i+2, chem)
        #     axs[int(i/7), i % 7].imshow(pattern_store[-1], cmap = 'plasma')
        #     axs[int(i/7), i % 7].set_title('Pattern {}'.format(i+1))

        # if save:
        #     plt.savefig('Fig_6.png')

        config = 1

        _, pattern_store, _, _ = main(config, chem)

        fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
        fig.set_size_inches(9, 3)
        ax1.set_title('t = 0')
        ax2.set_title('t = 12')
        ax3.set_title('t = 24')
        ax1.imshow(pattern_store[0], cmap = my_cmap)
        ax2.imshow(pattern_store[59], cmap = my_cmap)
        ax3.imshow(pattern_store[119], cmap = my_cmap)
        # time_series_data = list([] for i in range(0, len(pattern_store), 3))

        # for i in range(0, len(pattern_store), 3):
        #     time_series_data[int(i/3)] += [ax1.imshow(pattern_store[i], cmap = 'tab10')]

        # ani = animation.ArtistAnimation(fig, time_series_data, repeat = False)

        if save:
            plt.savefig('single_colony_growth_stages.png')

        plt.show()
        
    else:
        save = False
        chem = True      
        config = 3
        biomass_store, pattern_store, nutrient_store, toxin_store = main(config, chem)

        total_masses = np.sum(biomass_store, axis = (1,2))
        # set up and plot graphs
        fig, [ax1, ax2, ax3, ax4, ax5, ax6] = plt.subplots(1,6, figsize = (24,6))
        ax1.set_title("Pattern")
        ax2.set_title("Nutrient Concentration")
        ax3.set_title("Nutrient Cross-section")
        ax3.set_xlim(0, pattern_store[0].shape[0])
        ax3.set_ylim(0,np.max(nutrient_store[0]) + 2)
        ax3.set_aspect(np.diff(ax3.get_xlim())[0] / np.diff(ax3.get_ylim())[0])
        ax4.set_xlim( 0, len(pattern_store) )
        ax4.set_ylim( 0, total_masses.max() )
        ax4.set_title("Total Biomass")
        ax4.set_aspect(np.diff(ax4.get_xlim())[0]/np.diff(ax4.get_ylim())[0])
        ax5.set_title('Chemical 1')
        ax6.set_title('Chemical 2')

        # animate the stored time data
        time_series_data = list([] for i in range(0, len(pattern_store), 3))
        for i in range(0,len(pattern_store),3):
            
            time_series_data[int(i/3)] += [ax1.imshow(pattern_store[i], cmap = "tab10", vmin=0, vmax=3),
                                    ax2.imshow(nutrient_store[i],vmin = 0),
                                    ax3.plot( nutrient_store[i][500],)[0],
                                    ax4.plot( range(i) , total_masses[0:i] )[0],
                                    ax5.imshow(toxin_store[i][0], cmap = 'magma'),
                                    ax6.imshow(toxin_store[i][1], cmap = 'magma')]

        ani = animation.ArtistAnimation(fig, time_series_data, repeat = False)

        if save:
            plt.savefig('config_{}'.format(config))

        plt.show()