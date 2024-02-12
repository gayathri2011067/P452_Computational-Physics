
import numpy as np
import scipy as scipy
import math as math
from scipy.optimize import root

#pretty table,input to list,read input


############################################################################################################################################################
#                                       ASSIGNMENT 1                                  
############################################################################################################################################################
            
#__________Contents____________________________________________________________________________________
#1 For solving non-linear equations
        #bracketting method
        #bisection
        #secant method
        #fixed point method
#2 Numerical integration methods
        #midpoint
        #trapezoidal
        #simpsons
        #gaussian quadrature
#3 Solving ODEs
        #forward euler
        #backward eular
        #predictor-corrector
        #semi implicit euler
        #verlet and velocity verlet

############################################################################################################################################################
class non_linear_equation_1D:
    def __init__(self, f, df, a0, b0, tol):
        # Initialize the class with the function f, its derivative df, initial guesses a0 and b0, and tolerance tol.
        self.f = f
        self.a0 = a0
        self.b0 = b0
        self.tol = tol
        self.df = df
    
    def bracketing(a, b, f):
        # Expand the interval until the function changes sign.
        while f(a) * f(b) >= 0:
            # If the signs of the function at both ends are the same,
            # expand the interval to try to find a change in sign.
            if abs(f(a)) > abs(f(b)):
                b = b + 1.5 * (b - a)  # Expand the upper bound
            else:
                a = a - 1.5 * (b - a)  # Expand the lower bound
        return (a, b)  # Return the updated interval
    
    def bisection(self):
        # Method for finding the root using the bisection technique.
        a = self.a0
        b = self.b0
        fn = self.f
        epsilon = self.tol
        # Find an interval where the function changes sign.
        a, b = non_linear_equation_1D.bracketing(a, b, fn)
        count = 0
        # Bisection loop
        while abs(b - a) > epsilon:
            # Divide the interval in half and check which half contains the root.
            c = (a + b) / 2  # Calculate midpoint
            if fn(a) * fn(c) > 0:
                a = c  # Root lies in the right half
            else:
                b = c  # Root lies in the left half
            count += 1  # Count iterations
        return c, count  # Return the root and number of iterations
    
    def secant(self, guess1, guess2):
        # Method for finding the root using the secant method.
        fn = self.f
        t = self.tol
        x0 = guess1
        x1 = guess2
        # Secant iteration loop
        x2 = x1 - ((fn(x1) * (x1 - x0)) / (fn(x1) - fn(x0)))  # Calculate next approximation
        step = 1
        while abs(x2 - x1) > t:
            if step > 100:
                raise ValueError("The roots are not converging")  # Raise error if not converging
                break
            else:
                x0 = x1
                x1 = x2
                x2 = x1 - fn(x1) * (x1 - x0) / (fn(x1) - fn(x0))  # Update approximation
                step += 1
        return x2, step  # Return the root and number of iterations
    
    def fixed_point(self, x0):
        # Method for finding the root using the fixed-point iteration.
        g = self.f
        tol = self.tol
        # Fixed point iteration loop
        x1 = g(x0)  # Calculate next approximation
        step = 1
        while abs(x1 - x0) > tol:
            if step > 100:
                print("The roots are not converging")  # Print warning if not converging
                break
            else:
                x0 = x1
                x1 = g(x0)  # Update approximation
                step += 1
        return x1, step  # Return the root and number of iterations
###############################################################################################################################################################
 
class Integration_1:
    def __init__(self, f, a, b, N):
        # Initializing the class with the function, integration bounds, and number of subintervals.
        self.a = a  # Lower bound of integration
        self.b = b  # Upper bound of integration
        self.N = N  # Number of subintervals
        self.f = f  # Function to integrate

    def midpoint(self):
        # Midpoint rule for numerical integration
        h = (self.b - self.a) / self.N  # Calculate step size
        I = 0  # Initialize integral value
        x = self.a  # Initialize starting point
        while x <= self.b:
            I += h * self.f(x + h / 2)  # Add contribution from each subinterval
            x += h  # Move to the next subinterval
        return I  # Return the approximate integral value

    def trapezoidal(self):
        # Trapezoidal rule for numerical integration
        h = (self.b - self.a) / self.N  # Calculate step size
        I = 0  # Initialize integral value
        x = self.a  # Initialize starting point
        while x <= self.b:
            I += h / 2 * (self.f(x) + self.f(x + h))  # Add contribution from each trapezoid
            x += h  # Move to the next subinterval
        return I  # Return the approximate integral value

    def simpsons(self):
        # Simpson's rule for numerical integration
        h = (self.b - self.a) / self.N  # Calculate step size
        I = 0  # Initialize integral value
        x = self.a  # Initialize starting point
        while x <= self.b:
            I += h / 3 * (self.f(x) + 4 * self.f(x + h / 2) + self.f(x + h)) * 0.5  # Add contribution from each interval
            x += h  # Move to the next subinterval
        return I  # Return the approximate integral value


class Gaussian_Quadrature: 
    def __init__(self, f, a, b, degree):
        # Initializing the class with the function, integration bounds, and degree of the Gaussian Quadrature.
        self.a = a  # Lower bound of integration
        self.b = b  # Upper bound of integration
        self.N = degree  # Degree of the Gaussian Quadrature
        self.f = f  # Function to integrate   

    def Pn(self, x, n):
        # Calculate Legendre polynomial of degree n at point x
        if n == 0:
            return 1
        elif n == 1:
            return x
        else:
            return ((2 * n - 1) * x * self.Pn(x, n - 1) - (n - 1) * self.Pn(x, n - 2)) / n

    def Pn_drvtv(self, x, n):
        # Calculate the derivative of Legendre polynomial of degree n at point x
        if n == 0:
            return 0
        elif n == 1:
            return 1
        else:
            return (n * (x * self.Pn(x, n) - self.Pn(x, n - 1))) / (1 - x ** 2)

  

    def find_legendre_roots(self):
        # Find the roots of the Legendre polynomial using numerical root-finding methods
        n = self.N
        num_roots = self.N
        legendre_roots = []  # Renamed to avoid conflict
        for i in range(1, num_roots + 1):
            guess = np.cos((2 * i - 1) * np.pi / (2 * num_roots))  # Initial guess for the root
            result = root(self.Pn, guess, args=(n,), jac=self.Pn_drvtv, method='hybr')  # Solve for the root
            if result.success:
                legendre_roots.append(result.x[0])  # Store the root
        return legendre_roots



    def find_weights(self):
        # Calculate the weights for Gaussian Quadrature using Legendre roots
        n = self.N
        roots = self.find_legendre_roots()
        weights = []
        for i in range(n):
            w = 2 / ((1 - roots[i] ** 2) * (self.Pn_drvtv(roots[i], n)) ** 2)  # Calculate weight
            weights.append(w)  # Store the weight
        return weights


    def integrate(self):
        # Perform Gaussian Quadrature integration
        a = self.a
        b = self.b
        n = self.N
        f = self.f
        sum = 0
        weights = self.find_weights()
        roots = self.find_legendre_roots()
        for i in range(n):
            y = ((b - a) * 0.5 * roots[i]) + ((b + a) * 0.5)  # Calculate the mapping of roots to integration bounds
            weightf = weights[i] * f(y)  # Weight the function evaluation
            sum += weightf  # Accumulate the weighted sum
        val = (b - a) * 0.5 * sum  # Compute the final integral value
        return val  # Return the approximate integral value

#############################################################################################################################################################################################################################
    
class ODE_Solve_XY:
    def __init__(self, dy, xi,yi,xf,N):
        self.xi = xi
        self.yi = yi
        self.xf = xf
        self.N = N
        self.dy = dy
        
    def forward_euler(self):
        x_ini=self.yi
        t_ini=self.xi
        t_final=self.xf
        n=self.N
        dx=self.dy
        dt=(t_final-t_ini)/n
        xlist=[]
        ylist=[]
        t=t_ini
        while t<=t_final:
            xlist.append(t)
            ylist.append(x_ini)
            x_ini+=dt*dx(t,x_ini)
            t+=dt
        return xlist,ylist        
    
    def backward_euler(self):
        x0=self.xi
        y0=self.yi
        xf=self.xf
        num_points=self.N
        fn=self.dy

        h = (xf - x0) / num_points
        x_values = np.linspace(x0, xf, num_points + 1)
        y_values = np.zeros(num_points + 1)
        y_values[0] = y0

        for i in range(1, num_points + 1):
            y_values[i] = y_values[i - 1] + h * fn(x_values[i], y_values[i - 1])

        return x_values, y_values

    def predictor_corrector(self):
        dybydx=self.dy
        x0=self.xi
        y0=self.yi
        x_f=self.xf
        n=self.N

        h=(x_f-x0)/n
        xlist=[]
        ylist=[]
        x=x0
        y=y0
        xlist.append(x)
        ylist.append(y)
        while x<x_f:
            k1=dybydx(x,y)*h
            k2=dybydx(x+h,y+k1)*h
            y=y+0.5*(k1+k2)
            x=x+h
            xlist.append(x)
            ylist.append(y)
        return xlist,ylist
    
    def RK2_solve(self):

        dybydx=self.dy
        x0=self.xi
        y0=self.yi
        xf=self.xf
        N=self.N

        h=(xf-x0)/N
        xlist=[]
        ylist=[]
        x=x0
        y=y0
        xlist.append(x)
        ylist.append(y)
        while x<xf:
            k1=h*dybydx(x,y)
            k2=dybydx(x+(h/2),y+(k1/2))*h
            y=y+k2
            x=x+h
            xlist.append(x)
            ylist.append(y)
        return xlist,ylist
    
    def RK4_solve(self):

        dybydx=self.dy
        x0=self.xi
        y0=self.yi
        x_f=self.xf
        N=self.N

        h=(x_f-x0)/N
        xlist=[]
        ylist=[]
        x=x0
        y=y0
        xlist.append(x)
        ylist.append(y)
        while x<x_f:
            k1=h*dybydx(x,y)
            k2=h*dybydx(x+(h/2),y+(k1/2))
            k3=h*dybydx(x+(h/2),y+(k2/2))
            k4=h*dybydx(x+h,y+k3)
            y=y+(k1+2*k2+2*k3+k4)/6
            x=x+h
            xlist.append(x)
            ylist.append(y)
        return xlist,ylist     

def semi_implicit_euler_solve(f,g,x0,v0,t0,t_max,step_size):
   
    h=step_size
    vlist=[]
    xlist=[]
    tlist=[]
    x=x0
    v=v0
    t=t0
    while t<=t_max:
        xlist.append(x)
        vlist.append(v)
        tlist.append(t)
        v=v + (h*g(x,t))
        x=x + (h*f(v,t))
        t=t+h
    return xlist,vlist,tlist    

class verlet_algorithm:

    def __init__(self,a, x0, v0, t0, t_max, step_size):
        self.a = a
        self.x0 = x0
        self.v0 = v0
        self.t0 = t0
        self.t_max = t_max
        self.step_size = step_size

    def verlet_solve(self):
        #Defining the variables
        h=self.step_size
        xlist=[]
        vlist=[]
        tlist=[]
        tm=self.t_max
        x=self.x0
        t=self.t0
        v=self.v0
        acc_fn=self.a

        #The first 
        xlist.append(x)
        vlist.append(v)
        tlist.append(t)
        x1=(x)+(h*v)+(0.5*h*h*self.a(x))
        v1=(x1-x)/h
        t=t+h
        xlist.append(x1)
        vlist.append(v1)
        tlist.append(t)

        #The rest of the steps
        while t<=tm:
            x2=(2*x1)-(x)+(h*h*acc_fn(x1))
            v=(x2-x)/(2*h)
            x=x1
            x1=x2
            t=t+h
            xlist.append(x)
            vlist.append(v)
            tlist.append(t)

        return xlist,vlist,tlist    

    def velocity_verlet(self):
        
        h=self.step_size
        x=self.x0
        v=self.v0
        t=self.t0

        xlist=[x]
        vlist=[v]
        tlist=[t]
        pass


def crank_nicolson(dt,nodes, length, time_steps, boundary_conditions, initial_conditions):
    # Initialize solution matrix 'U' with boundary conditions
    
    dx = length / (nodes - 1)
    alpha = 0.5 * (dx / dt)
    U = np.zeros((nodes, time_steps + 1))
    X = np.linspace(0, length, nodes)
    U[:, 0] = initial_conditions(X)
    U[0, :], U[-1, :] = boundary_conditions[0], boundary_conditions[1]

    # Calculate matrices A and B
    r = alpha * dt / (2 * dx**2)
    A = np.diag(1 + 2 * r * np.ones(nodes))
    A += np.diag(-r * np.ones(nodes - 1), k=1)
    A += np.diag(-r * np.ones(nodes - 1), k=-1)

    B = np.diag(1 - 2 * r * np.ones(nodes))
    B += np.diag(r * np.ones(nodes - 1), k=1)
    B += np.diag(r * np.ones(nodes - 1), k=-1)

    # Iteratively solve the heat equation
    for t in range(1, time_steps + 1):
        U[:, t] = np.linalg.solve(A, np.dot(B, U[:, t - 1]))

    return U,alpha

def solve_poisson(N,u, dx, dy, f, max_iter=10000, tol=1e-6):
    #solving using finite element
    N_x=N
    N_y=N
    for k in range(max_iter):
        u_old = u.copy()
        for i in range(1, N_x - 1):
            for j in range(1, N_y - 1):
                # Compute the finite element approximation
                u[i, j] = (u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1] - dx**2 * f(i * dx, j * dy)) / 4
        if np.linalg.norm(u - u_old) < tol:
            break
    return u
