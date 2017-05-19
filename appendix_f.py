import numpy as np
import matplotlib.pyplot as plt
import FVis

#rho_x=[]
class Solver():
	def __init__(self,nx,nz,T,eta=0):
		"""constants"""
		self.nx = nx			# number of boxes x-direction
		self.nz= nz				# number of boces z-direction
		self.T = T				# Initial temperature
		self.my = 0.6			# average nuclear mass, unitless
		self.mu = 1.6605e-27		# kg Atomic mass unit
		self.kB = 1.38064852e-23 	# J/K Boltzmann constant
		self.x = np.arange(self.nx)	# x array
		self.z = np.arange(self.nz)	# z array
		self.eta = eta			# constant for viscosity
		self.G = 6.67408e-11 		# m^3 kg^-1 s^-2, gravitational constant
		self.R_sun=6.957e8 		# m, sun radius
		self.M_sun=1.989e30 	# kg, sun mass
		self.g = self.G*self.M_sun/self.R_sun**2	# gravitational force on sun

	def init_1d(self):
		""" Initial values for 1d sod shoch tube test """
		self.rho_x=[] # list to store rho in for plotting
		self.legend_t=[] # legend for plotting rho
		self.dt_sum=0
		self.i=0	# counter

		self.dx= 1.	# length of one box in x
		self.dz =1.	# length of one box in z

		self.ux = np.zeros(int(self.nx))	# velocity
		self.rho=np.zeros(int(self.nx))+1	# density
		self.rho[50:]=10					# shock front
		self.P = self.rho/(self.my*self.mu)*self.kB*self.T # pressure
		
	def init_2d(self):
		""" initial values for 2d pressure wave simulation """
		self.dx = 1.
		self.dz =1.

		self.ux = np.zeros((self.nz,self.nx))	# velocity matrix for x
		self.uz = np.zeros((self.nz,self.nx))	# velocity matrix for z
		self.rho = np.zeros((self.nz,self.nx))+1 	# density
		center1 = np.asarray((75,75))			# shock center 1
		center2 = np.asarray((125,125))			# shock center 2

		for i in range(int(self.nx)):			# get ciruclar shock blobs
			for k in range((self.nz)):
				if np.linalg.norm(np.asarray((i,k)) - center1) <=5:
					self.rho[i,k]=10.
				if np.linalg.norm(np.asarray((i,k)) - center2) <=5:
					self.rho[i,k]=10.
		self.P = self.rho/(self.my*self.mu)*self.kB*self.T	# Pa, pressure
	
	def init_convect(self):
		""" Initial values for simulating convection in the solar photosphere"""

		self.dx = 15e6/(self.nx-1)		# x-direction is 15 Mm long
		self.dz = 5e6/(self.nz-1)		# z-direction is 5 Mm long
		self.ux = np.zeros((self.nz,self.nx))	# initial velocity in x-direction
		self.uz = np.zeros((self.nz,self.nx))	# initial velocity in z-direction

		self.rho = np.zeros((self.nz,self.nx))	# kg m^-3, density
		self.P = np.zeros((self.nz,self.nx)) 	# Pa
		self.T = np.zeros((self.nz,self.nx)) 	#K
		self.e = np.zeros((self.nz,self.nx))

		self.nabla = 3./5.

		self.P[-1,:] = 1.8e8	# Pa, pressure on top of box, photosphere pressure
		self.T[-1,:] = 5778.	# K, temperature on top of box, photosphere temperature
		self.rho[-1,:] = self.P[-1,:]/(self.T[-1,:]*self.kB)*self.mu*self.my # kg/m^3, density 
		self.e[-1,:] = self.rho[-1,:]*self.kB/(self.mu*self.my)*self.T[-1,:]
	
		#get initial conditions for entire matrices
		for n in range(self.nx-1,0,-1): # have to flip matrices, working from down-up now
			if n==self.nx-1:
				dP_dz = self.g*self.rho[n,:]
				dT_dz = self.nabla*self.T[n,:]/self.P[n,:]*dP_dz	
				self.P[n-1,:] = self.P[n,:] + dP_dz*self.dz
				self.T[n-1,:] = self.T[n,:] + dT_dz*self.dz
				self.rho[n-1,:] = self.P[n-1,:]*self.mu*self.my/(self.T[n-1,:]*self.kB)
				self.e[n-1,:] = self.rho[n-1,:]*self.kB/(self.mu*self.my)*self.T[n-1,:] # energy per volume
			else:
				dP_dz = self.g*self.rho[n,:]
				dT_dz = self.nabla*self.T[n,:]/self.P[n,:]*dP_dz	

				self.P[n-1,:] = self.P[n+1,:] + 2*dP_dz*self.dz
				self.T[n-1,:] = self.T[n+1,:] + 2*dT_dz*self.dz
				self.rho[n-1,:] = self.P[n-1,:]*self.mu*self.my/(self.T[n-1,:]*self.kB)
				self.e[n-1,:] = self.rho[n-1,:]*self.kB/(self.mu*self.my)*self.T[n-1,:] # energy per volume
			#print self.P[n-1,:]
		# put in center blob

		center2=np.asarray((self.nz/2.,self.nx/2.))
		center1 =np.asarray((self.nz-10.,self.nx-10.))
		for i in range(self.nx):
			for k in range(self.nz):
				if np.linalg.norm(center1-np.asarray((i,k)))<=5:
					self.e[i,k]*=5.2
				if np.linalg.norm(center2-np.asarray((i,k)))<=5:
					self.e[i,k]*=5.2
		self.T_top = self.T[-1,:]		# for boundry conditions
		self.T_bottom = self.T[0,:]	# for boundry conditions

	def plot_rho(self):
		""" plot rho at differnt times dt"""
		y = np.array(self.rho_x)
		for n in range(len(y[:,0])):
			plt.plot(self.x,y[n,:])
		plt.legend(self.legend_t,loc ='lower right')
		plt.title('Time evolution of 1d shock wave in space')
		plt.xlabel(r'$x$[m]')
		plt.ylabel(r'$\rho\mathrm{[kg/m^3]}$')
		plt.show()

	def upwind_x(self,phi,u,dx,i):
		""" takes in variable phi, calculates spatial derivative in x direction
			both 1d and 2d by use of the upwind method, i=1 2d, i=0 convection"""
		dphi_dx = np.zeros((self.nz,self.nx))
		if self.nz ==1: 
			dphi_dx = np.zeros(self.nx)
			i=0	
		a = (np.roll(phi,-1, axis = i)-phi)/dx
		b = (phi-np.roll(phi,1, axis = i))/dx
		dphi_dx[np.where(u<0)] = a[np.where(u<0)]
		dphi_dx[np.where(u>=0)] = b[np.where(u>=0)]
		return dphi_dx

	def upwind_z(self,phi,u,dz,i):
		""" takes in variable phi, calculates spatial derivative in z direction
			both in 1d and 2d by use of the upwind method. i=0, 2d, i=1 convection"""
		#dphi_dz=self.BC_z_periodic(self,phi,u,upwind)
		#return dphi_dz		
	
		dphi_dz = np.zeros((self.nz,self.nx))
		a = (np.roll(phi,-1,axis=i)-phi)/dz
		b = (phi-np.roll(phi,1,axis=i))/dz
		dphi_dz[np.where(u<0)] = a[np.where(u<0)]
		dphi_dz[np.where(u>=0)] = b[np.where(u>=0)]
		return dphi_dz	

	def midpoint_x(self,phi,u,dx,i):
		""" uses central difference method of calculating spatial derivatives
			in x direction, both 1d and 2d. i=1 2d, i=0 convection"""
		if self.nz==1: i=0
		phi_plus = np.roll(phi,-1,axis=i)
		phi_minus = np.roll(phi,1,axis=i)
		dphi_dx = (phi_plus-phi_minus)/(2*dx)
		return dphi_dx

	def midpoint_z(self,phi,u,dz,i):
		""" uses central difference method of calculating spatial derivatives
			in z direction, both 1d and 2d, i=0 2d, i=1 convection"""
		#dphi_dz=self.BC_z_periodic(self,phi,u,upwind=False)
		#return dphi_dz		
		phi_plus = np.roll(phi,-1,axis=i)-phi
		phi_minus = np.roll(phi,1,axis=i)-phi
		dphi_dz = (phi_plus-phi_minus)/(2*dz)
		return dphi_dz

	def viscosity(self,duxdz,duzdx):
		""" calculates viscosity, diagonal terms are zero, off-diagonal
			terms are equal"""
		tau_xz = self.rho*self.eta*(duxdz+duzdx)
		return tau_xz

	def rel_phi(self,phi,dphi_dt):
		""" calculates the relative phi to be used in calculating dt"""
		return np.max(np.abs(dphi_dt/phi))
	
	def step_1d(self):
		""" step function for 1d sod shock tube test"""
		p = 0.1		# for calculating dt
		cs = np.sqrt(5./3*self.P/self.rho)	# sound speed
		dt1 = np.abs(0.3*self.dx/np.max(cs)) # backup timestep

		drho_dx =self.upwind_x(self.rho,self.ux,self.dx,i=0)	#drho/dx
		dP_dx = self.midpoint_x(self.P,self.ux,self.dx,i=0)		#dP/dx
		du_dx =self.midpoint_x(self.ux,self.ux,self.dx,i=0)		#dux/dx

		drho_dt = -self.rho*du_dx - self.ux*drho_dx			#drho/dt
		dP_dt = drho_dt *self.kB*self.T/(self.my*self.mu)	#dP/dt
		du_dt=-(self.ux*du_dx +1./self.rho*dP_dx)			#dux/dt

		# to calculate dt
		rel_rho = self.rel_phi(self.rho,drho_dt)
		rel_u = self.rel_phi(self.ux,du_dx)
		rel_x = self.rel_phi(self.dx,self.ux)

		delta = np.max([rel_rho,rel_x])
		if delta==0.0:		
			dt =dt1
		else:dt = p/delta	# new dt
	
		if self.i%50 ==0:		# save rho every 50th time step
			self.rho_x.append(self.rho.tolist())
			self.legend_t.append('t=%.2f s'% (self.dt_sum))

		self.i+=1
		self.dt_sum+=dt

		# update parameters using 1st order Euler
		self.rho[:] = self.rho + dt*drho_dt
		self.ux[:] = self.ux + dt*du_dt
		self.P[:] = self.P + dt*dP_dt

		return dt


	def step_2d(self):
		""" step function for 2d shock wave"""
		p=0.1		# for calculating dt
		cs = np.sqrt(5./3*self.P/self.rho)	# speed of sound
		dt1 = np.abs(0.3*self.dx/np.max(cs))	# back up time step

		# rho spatial derivatives
		drho_dx = self.upwind_x(self.rho,self.ux,self.dx,i=1)
		drho_dz = self.upwind_z(self.rho,self.uz,self.dz,i=0)

		# P spatial derivatives
		dP_dx = self.midpoint_x(self.P,self.ux,self.dx,i=1)	
		dP_dz = self.midpoint_z(self.P,self.uz,self.dz,i=0)
			
		# velocity in x and z direction spatial derivatives
		dux_dx = self.midpoint_x(self.ux,self.ux,self.dx,i=1)
		duz_dz = self.midpoint_z(self.uz,self.uz,self.dz,i=0)
		# cross terms
		duz_dx = self.midpoint_x(self.uz,self.ux,self.dx,i=1)
		dux_dz = self.midpoint_z(self.ux,self.uz,self.dz,i=0)

		
		# viscocity stress tensor
		tau_xz = self.viscosity(dux_dz,duz_dx)

		# tau spatial derivatives
		dtau_zx_dx = self.midpoint_x(tau_xz,self.ux,self.dx,i=1)
		dtau_xz_dz = self.midpoint_z(tau_xz,self.uz,self.dz,i=0)

		# calculate temporal derivatives		
		drho_dt = -self.rho*(dux_dx + duz_dz) - self.ux*drho_dx - self.uz*drho_dz
		dP_dt = drho_dt *self.kB*self.T/(self.my*self.mu)

		dux_dt = -self.ux*dux_dx -self.uz*dux_dz + 1./self.rho*(-dP_dx  + dtau_xz_dz) 
		duz_dt = -self.uz*duz_dz -self.ux*duz_dx + 1./self.rho*(-dP_dz  + dtau_zx_dx)

		# to calculate dt
		rel_rho = self.rel_phi(self.rho,drho_dt)
		rel_x = self.rel_phi(self.dx,self.ux)
		rel_z = self.rel_phi(self.dz,self.uz)
	
		delta = np.max([rel_rho,rel_x,rel_z])
		if delta==0.0:
			dt = dt1	
		else: dt = p/delta #new dt
		
		# update parameters
		self.rho[:] = self.rho + dt*drho_dt
		self.P[:] 	= self.P + dt*dP_dt
		self.ux[:] 	= self.ux + dt*dux_dt
		self.uz[:] 	= self.uz + dt*duz_dt
		
		return dt

	def step_convect(self):
		""" step function used when simulating convection """

		p=0.1	# to be used in calculating dt
		cs = np.sqrt(abs(5./3*self.P/self.rho))
		#print self.rho
		#print self.P
		#cs=np.sqrt(5./3*self.T*self.kB/(self.mu*self.my))

		dt1 = np.array([np.abs(0.3*self.dx/np.max(cs)),np.abs(0.3*self.dz/np.max(cs))])	# back up time step

		# rho spatial derivatives
		drho_dx = self.upwind_x(self.rho,self.ux,self.dx,i=0)
		drho_dz = self.upwind_z(self.rho,self.uz,self.dz,i=1)

		# P spatial derivatives
		dP_dx = self.midpoint_x(self.P,self.ux,self.dx,i=0)	
		dP_dz = self.midpoint_z(self.P,self.uz,self.dz,i=1)
		# BCs
		dP_dz[-1,:] = self.g*self.rho[-1,:]
		dP_dz[0,:] = self.g*self.rho[0,:]
			# if used in own time derivative, use upwind, else use midpoint 
		# velocity derivatives
		dux_dx = self.midpoint_x(self.ux,self.ux,self.dx,i=0)
		duz_dz = self.midpoint_z(self.uz,self.uz,self.dz,i=1)
		# velocity cross terms
		duz_dx = self.midpoint_x(self.uz,self.ux,self.dx,i=0)
		dux_dz = self.midpoint_z(self.ux,self.uz,self.dz,i=1)

		# BCs
		
		dux_dz[-1,:]= 0.
		dux_dz[0,:]= 0.

		# energy spatial derivatives
		de_dx = self.upwind_x(self.e,self.ux,self.dx,i=0)
		de_dz = self.upwind_z(self.e,self.uz,self.dz,i=1)
	
		# viscocity
		tau_xz = self.viscosity(dux_dz,duz_dx)
		# tau spatial derivatives
		dtau_zx_dx = self.midpoint_x(tau_xz,self.ux,self.dx,i=0)
		dtau_xz_dz = self.midpoint_z(tau_xz,self.uz,self.dz,i=1)
		
		# calculate temporal derivatives
		drho_dt = -self.rho*(dux_dx + duz_dz) - self.ux*drho_dx - self.uz*drho_dz
		dP_dt = drho_dt *self.kB*self.T/(self.my*self.mu)

		dux_dt = -self.ux*dux_dx -self.uz*dux_dz + 1./self.rho*(-dP_dx  + dtau_xz_dz) 
		duz_dt = -self.uz*duz_dz -self.ux*duz_dx + 1./self.rho*(-dP_dz  + dtau_zx_dx) + self.g

		de_dt = -(dux_dx + duz_dz)*(self.e + self.P) -self.ux*de_dx - self.uz*de_dz

		# to calculate dt
		rel_rho = self.rel_phi(self.rho,drho_dt)
		rel_x = self.rel_phi(self.dx,self.ux)
		rel_z = self.rel_phi(self.dz,self.uz)
		rel_e = self.rel_phi(self.e,de_dt)
	
		delta = np.max([rel_rho,rel_x,rel_z,rel_e])
		if delta==0.0:
			dt = p#np.min(dt1) # new dt	
		else: dt = p/delta

		# update parameters
		self.rho[:] = self.rho + dt*drho_dt
		self.ux[:] 	= self.ux + dt*dux_dt
		self.uz[:] 	= self.uz + dt*duz_dt
		self.e[:]	= self.e + dt*de_dt
		self.T[:]	= self.e*self.mu*self.my/(self.rho*self.kB) #self.T + dt*self.dT
		self.P[:] 	= self.rho*self.kB*self.T/(self.mu*self.my)#self.P + dt*dP_dt

		# BCs
		self.uz[-1,:] = 0
		self.uz[0,:] = 0
		self.T[-1,:] = 0.9*self.T_top
		self.T[0,:] = 1.1*self.T_bottom
		self.rho[-1,:] = dP_dz[-1,:]/self.g
		self.rho[0,:] = dP_dz[0,:]/self.g
			# ux=0 at endpoints
		#self.e[-1,:] = self.rho[-1,:]*self.T[-1,:]*self.kB/(self.mu*self.my)
		#self.e[0,:] = self.rho[0,:]*self.T[0,:]*self.kB/(self.mu*self.my)
		return dt

	def plot_formatting(self,fam='serif',fam_font='Computer Modern Roman',font_size=15,tick_size=15):
		""" you get to define what font and size of xlabels and axis ticks you"""
		"""like, if you want bold text or not.								  """
	
		plt.rc('text',usetex=True)
		axis_font={'family': fam,'serif':[fam_font],'size':font_size}
		plt.rc('font',**axis_font)
		plt.rc('font',weight ='bold')
		#plt.rcParams['text.latex.preamble']=[r'\boldmath']
		plt.xticks(fontsize=tick_size)
		plt.yticks(fontsize=tick_size)


if __name__=='__main__':
	vis = FVis.FluidVisualiser()
	sim_1D = True
	sim_2D = False
	sim_convect = False

	if sim_1D ==True: # simulate 1d sod shock tube
		solver = Solver(nx=100,nz=1,T=1e-8) 	# 1e-8
		solver.init_1d()
		vis.save_data(1500,solver.step_1d,rho=solver.rho, u=solver.ux,P=solver.P,sim_fps=0.5)
		vis.animate_1D('rho',save=True,video_name='project3_1d_shock_wave')
		vis.animate_1D('rho')
		solver.plot_formatting()
		vis.plot_avg('rho') # plots relative average of rho for every time step against x
		vis.plot_q('rho')
		solver.plot_rho() # plots rho for different t


	if sim_2D==True: 	# simulate 2d sod shock tube with or without viscosity
		solver = Solver(nx=200,nz=200,T=1e-7,eta=0) # 
		#solver = Solver(nx=200,nz=200,T=1e-7,eta=0.001) # with viscosity
		solver.init_2d()
		vis.save_data(1000,solver.step_2d,rho=solver.rho, u=solver.ux,w=solver.uz,P=solver.P,sim_fps=0.5)
		#vis.animate_2D('P',snapshots=[210,960])	# snapshots before and after colision
		#vis.animate_2D('rho',snapshots=[600])	# snapshots for viscosity and nod viscosity
		vis.animate_2D('rho')#,save=True,video_name='project3_2d_shock_wave')

	if sim_convect == True: # simulate convection
		#solver = Solver(nx=10,nz=10,T=5778,eta=0) # 1e-7 
		solver = Solver(nx=100,nz=100,T=5778,eta=0) # 1e-7 
		# 80
		solver.init_convect()
		vis.save_data(150,solver.step_convect,rho=solver.rho, u=solver.ux,w=solver.uz,P=solver.P,e=solver.e,sim_fps=0.5)
		vis.animate_2D('rho',extent=[0, 15, 0, 5, 'Mm'],showQuiver=False)
	
	vi