#solving linear equations-lu decomposition-Matrix A is factorized or decomposed into a product of lower triangular L(top 0) and upper triangular U matrices,
#then forward backward substitution is done-First solve for y from L · y = b using forward substitution and then use it to solve for x by backward substitution from U · x = y
def lu_dec(A):
    

    n = len(A) 
    if n != len(A[0]): 
        print('Not square!')
        return None

    for j in range(n):
        for i in range(1,n):
            if i <= j:
                sum = 0
                for k in range(0,i):
                    sum += (A[i][k] * A[k][j])
                A[i][j] = A[i][j] - sum
            else:
                sum = 0
                for k in range(0,j):
                    sum += (A[i][k] * A[k][j])
                A[i][j] = (A[i][j] - sum) / A[j][j]
    return A

def for_back_LU(A,B):
    Y = []
    n = len(B)
    for i in range(n):
        s = 0
        for j in range(i):
            s += (A[i][j] * Y[j])
        Y.append(B[i][0]-s)
    
    X = Y[:]
    for i in range(n-1,-1,-1):
        s = 0
        for j in range(i + 1, n):
            s+= (A[i][j] * X[j])
        X[i] = (Y[i] - s) / A[i][i]

    for i in range(n):
        X[i] = [X[i]]

    return X
def LU_solve(A,B):
    return for_back_LU(lu_dec(A),B)


def simpsons_rule(f, a, b, N):
    # Simpson's rule for numerical integration
    h = (b - a) / N  # step size
    I = 0
    x = a  
    while x <= b:
        I += h / 3 * (f(x) + 4 * f(x + h / 2) + f(x + h)) * 0.5  
        x += h  
    return I  

def pde_explicit(f,nx,nt,lx,lt,N):
    hx=lx/nx
    ht=lt/nt
    a = ht/(pow(hx,2))
    V0,V1 = [0],[0]
    for i in range(nx+1):
        V1.append(0)
    for i in range(nx+1):
        V0.append(f(i*hx))
    for j in range(N):
        for i in range(nx+1):
            if i==0:
                V1[i]=(1-2*a)*V0[i] + (a*V0[i+1])
            elif i==nx:
                V1[i]=(a*V0[i-1]) + (1-2*a)*V0[i]
            else:
                V1[i]=(a*V0[i-1]) + (1-2*a)*V0[i] + (a*V0[i+1])
        for i in range(nx+1):
            V0[i] = V1[i]        
    if N==0:
        return V0
    else:
        return V1



# Define the RK4 solver
def RK4_solve(dy, xi, yi, xf, N, E):
   
    h = (xf - xi) / N
    xlist = []
    ylist = []
    x = xi
    y = yi
    xlist.append(x)
    ylist.append(y)
    while x < xf:
        k1 = h * dy(y, x, E)
        k2 = h * dy(y + 0.5 * k1, x + 0.5 * h, E)
        k3 = h * dy(y + 0.5 * k2, x + 0.5 * h, E)
        k4 = h * dy(y + k3, x + h, E)
        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x += h
        xlist.append(x)
        ylist.append(y)
    return xlist, ylist

# Define the shooting method
def shooting_method(d2ydt2, x0, xf, N, E_guess, tol=1e-6):
# Initial value for the function 
    y0 = 0
    # Perform shooting method
    def residual(E):
        X, Y = RK4_solve(d2ydt2, x0, y0, xf, N, E)
        return Y[-1]  
    
    a, b = 0, E_guess  
    while abs(b - a) > tol:
        c = (a + b) / 2
        if residual(c) == 0:
            return c
        elif residual(a) * residual(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2


def newton_raphson(x0,f,fd):
    count=0
    xn=x0
    xn1=xn-(f(xn)/fd(xn))
    while True:
        xn=xn1
        xn1=xn-(f(xn)/fd(xn)) 
        count+=1
        if abs(xn1-xn)<10**-4:
            break          
    return xn1,count+1

def bracket(a0,b0,f):
    n=0
    while f(a0)*f(b0)>=0:
        if abs(f(a0))>abs(f(b0)):
            b0=b0+1.5*(b0-a0)
        else:
            a0=a0-1.5*(b0-a0)       
    return(a0,b0)
def regula_falsi(a0,b0,f,T):
    epsilon=T
    a0,b0=bracket(a0,b0,f)
    for i in range(0,1):
        c0=b0-(((b0-a0)*f(b0))/(f(b0)-f(a0)))
        cold=c0  
        if f(a0)*f(c0)<0:
            b0=c0       
        else:    
            a0=c0 

     
    count=1

    while True:
        cold=c0
        c0=b0-(((b0-a0)*f(b0))/(f(b0)-f(a0)))
        if f(a0)*f(c0)<0:
            b0=c0       
        else:    
           a0=c0 
        if abs(cold-c0)<epsilon:
            break 
        count+=1
    return c0,count
