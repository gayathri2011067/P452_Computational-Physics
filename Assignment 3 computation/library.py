import numpy as np
import scipy as scipy
import math as math
from scipy.optimize import root
# from Library_old import *
#####################################################################################
#                            Random Number Generation                             
#####################################################################################
class rng():
    def __init__(self,seed, a = 1103515245, c = 12345 ,m = 32768):
        # initiation of data input
        self.term = seed
        self.a = a
        self.c = c
        self.m = m
    def gen(self):
        # generates a random number
        self.term = (((self.a * self.term) + self.c) % self.m)
        return self.term
    def genlist(self,length):
        # returns a list of 'n' random numbers in the range (0,1) where 'n' is 'length'.
        RNs = []
        for i in range(length):
            self.term = (((self.a * self.term) + self.c) % self.m)
            RNs.append(self.term / self.m)
        return RNs  

#####################################################################################
#                               Matrix Operations                             
#####################################################################################
def print_matrix(matrix):
    '''
    This function prints the matrix A defined in the init function
    if flag=1, it prints the matrix B
    '''
    for row in matrix:
        for element in row:
            print("\t",element, end="\t")
        print()




def read_matrices(filename: str,delimiter: str = '\t'):
    '''
    Reading matrices from a file

    # Parameters
    - filename: The name of the file from which the matrices are to be read
    # Returns
    - The list of matrices read from the file seperated from "#"
    '''
    matrices = []
    current_matrix = []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()  
            if not line or line.startswith("#"):
                if current_matrix: 
                    matrices.append(current_matrix)
                    current_matrix = []  
                continue
            
            try:
                row = [float(num) for num in line.split(delimiter)]
                current_matrix.append(row)
            except ValueError:
                # print("Skipping non-numeric line:", line)
                pass
        if current_matrix:
            matrices.append(current_matrix)
    return matrices



def matrix_copy(A1):
    '''
    This function returns a copy of the matrix A
    '''
    A2 = [[0 for i in range(len(A1[0]))] for j in range(len(A1))]
    for i in range(len(A1)):
        A2[i]=A1[i][:]
    return A1 


def add_matrix(A,B):
    '''
    This function adds the matrix A and B
    '''
    result = matrix_copy(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = result[i][j] + B[i][j]
    return result




def multiply_matrix(A,B):
    '''
    This function multiplies the matrix A and B
    '''
    result = [[0 for i in range(len(B[0]))] for j in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result   


def get_transpose(A):
    '''
    This function returns the transpose of the matrix A
    '''
    for i in range(len(A)):
        for j in range(i+1,len(A[i])):
            A[i][j],A[j][i]=A[j][i],A[i][j]
    return A

def get_det(A):
    '''
    # Determinant of a Matrix
    '''
    A=[A[i][:] for i in range(len(A))]
    n = len(A)
    if n != len(A[0]):
        print('Not a square matrix')
        return None
    for curr in range(n):
        if A[curr][curr] == 0:
            max_row = curr
            for row in range(curr + 1,n):

                if abs(A[row][curr]) > abs(A[max_row][curr]):
                    max_row = row
            if max_row == curr:
                print('The matrix is singular!')
                return None
            A[curr],A[max_row] = A[max_row], A[curr]
        for i in range(curr + 1,n):
            if A[i][curr] != 0:
                lead_term = A[i][curr]/A[curr][curr]
                for j in range(curr,len(A[i])):
                    A[i][j] = A[i][j] - (A[curr][j] * lead_term)
    prdct = 1
    for i in range(n):
        prdct *= A[i][i]
    return prdct






def Get_Gauss_jordan_inv(A):
    '''
    # Inverse of a Matrix using Gauss Jordan Method
    ## Parameters
    - A: The matrix whose inverse is to be found
    ## Returns
    - The inverse of the matrix A
    '''

    if len(A) != len(A[0]):
        raise ValueError('Matrix is not square')

    n = len(A)
    I = []
    for row in range(n):
        I.append(list())
        for col in range(n):
            if col == row:
                I[row].append(1)
            else:
                I[row].append(0)
    for curr in range(n): 
        if A[curr][curr] == 0:
            max_row = curr
            for row in range(curr + 1,n):
                if abs(A[row][curr]) > abs(A[max_row][curr]):
                    max_row = row
            if max_row == curr:
                return None
            A[curr],A[max_row] = A[max_row], A[curr]
            I[curr],I[max_row] = I[max_row], I[curr]
        if A[curr][curr] != 1:
            pivot = A[curr][curr]
            for i in range(n):
                A[curr][i] = A[curr][i] / pivot
                I[curr][i] = I[curr][i] / pivot
        for i in range(0,n):
            if i == curr:
                continue
            if A[i][curr] != 0:
                lead = A[i][curr]
                for j in range(0,n):
                    A[i][j] = A[i][j] - (A[curr][j] * lead)
                    I[i][j] = I[i][j] - (I[curr][j] * lead)
    return I



def check_symmetry(A):
    '''
    # Checks if a matrix is symmetric
    ## Parameters
    - A: The matrix to be checked
    ## Returns
    - True if the matrix is symmetric, False otherwise
    '''
    for i in range(len(A)):
        for j in range(len(A[i])):
            if i != j:
                if A[i][j] != A[j][i]:
                    return False
    return True

#####################################################################################
#                          Solution of Linear Equations                             
#####################################################################################
def gauss_jordan_solve(A,B):
    '''
    # Gauss Jordan Method
    ## Parameters
    - A: The matrix A in the equation A.X = B
    - B: The matrix B in the equation A.X = B
    ## Returns
    - Solution: X,The solution of the equation A.X = B
    '''
    augmat = A #constructing augmented matrix
    for row in range(len(augmat)):
        augmat[row].append(B[row][0])
    for curr in range(len(augmat)): #curr takes the index of each column we are processing
        if augmat[curr][curr] == 0: #row swap if zero
            max_row = curr
            for row in range(curr + 1,len(augmat)):
                if abs(augmat[row][curr]) > abs(augmat[max_row][curr]):
                    max_row = row
            if max_row == curr: #if max elemnt is still zero, max_row is not changed; no unique solution
                return None
            augmat[curr],augmat[max_row] = augmat[max_row], augmat[curr]
        #making the pivot element 1
        if augmat[curr][curr] != 1:
            pivot_term = augmat[curr][curr]
            for i in range(len(augmat[curr])):
                augmat[curr][i] = augmat[curr][i] / pivot_term
        #making others zero
        for i in range(0,len(augmat)):
            if i == curr: #skipping the pivot point
                continue
            if augmat[i][curr] != 0:
                lead_term = augmat[i][curr]
                for j in range(curr,len(augmat[i])): #elements before the curr column are zero in curr row, so no need to calculate
                    augmat[i][j] = augmat[i][j] - (augmat[curr][j] * lead_term)
    solution = []
    for i in range(len(augmat)):
        solution.append([augmat[i][-1]]) #Taking last elements into a list to form column matrix
    return solution




'''
# LU Decomposition Method for solving the equation A.X = B
'''
def LU_decompose(A):
    '''
    # LU Decomposition of a Matrix
    '''
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
    '''
    # Forward and Backward Substitution for LU Decomposition
    '''
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


def LU_Solve_eqn(A,B):
    '''
    # LU Decomposition Method for solving the equation A.X = B
    '''
    L=LU_decompose(A)
    L1=for_back_LU(L,B)
    return L1






'''
# Cholesky Decomposition Method for solving the equation A.X = B
'''
def Cholesky_decompose(A):

    if check_symmetry(A) == False:
        raise ValueError('The matrix is not symmetric')
      
    from math import sqrt
    if len(A) != len(A[0]):
        return None
    n = len(A)

    for row in range(n):
        for col in range(row,n): 
            if row == col:
                sum = 0
                for i in range(row):
                    sum += (A[row][i] ** 2)
                A[row][row] = sqrt(A[row][row] - sum)
            else:
                sum = 0
                for i in range(row):
                    sum += (A[row][i] * A[i][col])
                A[row][col] = (A[row][col] - sum) / A[row][row]
                A[col][row] = A[row][col]   
    for row in range(n):
        for col in range(row + 1,n):
            A[row][col] = 0   
    return A

def sup_chol_dec(A):
    L=Cholesky_decompose(A)
    LT=get_transpose(L)
    LP=[[0 for i in range(len(L))] for i in range(len(L))]
    for i in range(len(L)):
        for j in range(len(L)):
            LP[i][j]=L[i][j]+LT[i][j]
    for i in range(len(L)):
        LP[i][i]=LP[i][i]/2
    return LP    

def Cholesky_solve(A,B):
    '''
    # Solving using Cholesky Decomposition
        Solves the Equation A.X = B
    ## Parameters
    - A: The matrix A in the equation A.X = B
    - B: The matrix B in the equation A.X = B
    ## Returns
    - X: The solution of the equation A.X = B
    '''
    Y = []
    n = len(B)
    for i in range(n):
        sum = 0
        for j in range(i):
            sum += (A[i][j] * Y[j])
        Y.append((B[i][0]-sum)/A[i][i])

    X = Y
    for i in range(n-1,-1,-1):
        sum = 0
        for j in range(i + 1, n):
            sum += (A[i][j] * X[j])
        X[i] = (Y[i] - sum) / A[i][i]

    for i in range(n):
        X[i] = [X[i]]

    return Y





'''
Solution using Gauss Seidel Method
'''

def Check_diagonal_dominance(m) :
    n=len(m)
    for i in range(0, n) :        
        sum = 0
        for j in range(0, n) :
            sum = sum + abs(m[i][j])    
        sum = sum - abs(m[i][i])
        if (abs(m[i][i]) < sum) :
            return False 
    return True

def Make_diag_dominant(A,B):
    '''
    # Making the matrix A Diagonally Dominant
    Returns the Diaonally Dominant matrix A and the corresponding B
    '''
    n = len(A)
    for i in range(n):
        sum=0
        for j in range(n):
            if j != i:
                sum += abs(A[i][j])
        if abs(A[i][i])>sum:
            continue
        else:
            c = i + 1
            flag = 0
            while c<n:
                sum = 0
                for j in range(n):
                    if j != i:
                        sum += abs(A[c][j])
                if abs(A[c][i])>sum:
                    flag = 1
                    break
                else:
                    c+=1
            if flag==0:
                return None,None
            else:
                A[i],A[c]=A[c],A[i]
                B[i],B[c]=B[c],B[i]
    return A,B




def gauss_seidel(A,B,tolerance):

    '''
    # Gauss Seidel Method
    Solves the Linear Equation A.X = B using Gauss Seidel Method
    It does not check for the Diagonal Dominance of the matrix A
    The limit of the Steps is 200, if the solution is not found in 200 steps, it returns an error message
    '''
    n = len(A)
    X = list([0] for i in range(n))
    for steps in range(200):
        flag = 1
        for i in range(n):
            sum = 0
            for j in range(i):
               sum += (A[i][j] * X[j][0])
            for j in range(i+1,n):
                sum += (A[i][j] * X[j][0])
            temp = (B[i][0] - sum) / (A[i][i])
            if (abs(temp) - abs(X[i][0])) > tolerance:
               flag = 0
            X[i][0] = temp
        if flag == 1:
            return X,steps + 1
    print('Eqn not solved after 200 steps')
    return None,200

def Gauss_seidel_solve(A,B,T):
    '''
    # Gauss Seidel Method
    This function solves the Linear Equation A.X = B using Gauss Seidel Method
    Also Checks for Diagonal Dominance of the matrix A. If the matrix is not diagonally dominant, it returns an error message
    '''
    if Check_diagonal_dominance(A)==True:
        return gauss_seidel(A,B,T)
    else:
        raise ValueError('The matrix is not diagonally dominant')   
    
def Get_gauss_seidel_inv(A, tolerance):
    '''
    # Inverse Using Gauss Seidel Method
    '''
    if len(A) != len(A[0]):
        raise ValueError('The matrix is not square')
    
    A_inv_cols = []
    
    for i in range(len(A)):
        I_col = np.zeros((len(A),1)).tolist()
        I_col[i][0] = 1
        A_inv_col, _ = Gauss_seidel_solve(A,I_col, tolerance)
        A_inv_cols.append(A_inv_col)
    A_inv = np.array(A_inv_cols[0])
    for i in range(1,len(A)):
        A_inv = np.append(A_inv,A_inv_cols[i],axis = 1) 
    return A_inv




def Gauss_jacobi_method(A: list, B: list,guess: list,tol: float):
    '''
    # Gauss Jacobi Method
    Solves the Linear Equation A.X = B using Gauss Jacobi Method
    '''
    XK=guess
    XK1=[[0] for i in range(len(A))]
    count=0
    sumaijxj=0
    flag=0
    while flag!=1:
        for i in range(len(XK1)):
            sumaijxj=0
            for j in range(len(A[i])):
                if i!=j:
                    sumaijxj+=A[i][j]*XK[j][0]    
            XK1[i][0]=(1/A[i][i])*(B[i][0]-sumaijxj)
        for i in range(len(A)):
            if abs(XK[i][0]-XK1[i][0])<tol:
                flag=1    
        count+=1
        for i in range(len(XK1)):
            XK[i][0]=XK1[i][0]
    return XK,count

def Gauss_Jacobi_solve(A,B,guess,T):
    '''
    # Gauss Jacobi Method 
    This function check for the Diagonal Dominance of the matrix A. 
    If the matrix is not diagonally dominant, it makes the matrix diagonally dominant and then solves the equation
    '''
    if Check_diagonal_dominance(A)==True:
        return Gauss_jacobi_method(A,B,guess,T)
    else:
        print(" The given Matrix was not Diagonallly Dominant") 
        print(" Made the Matrix Diagonally Dominant and then solved the equation")
        A,B=Make_diag_dominant(A,B)
        return Gauss_jacobi_method(A,B,guess,T)



def conjugate_gradient(A: list,B: list,guess: list,T: float):
    '''
    # Cojugate Gradient Method
    Solves the Linear Equation A.X = B using Conjugate Gradient Method
    ## Parameters
    - A: The matrix A in the equation A.X = B. *A must be a symmetric and positive definite matrix*
    - B: The matrix B in the equation A.X = B
    - guess: The initial guess for the solution
    - T: Tolerance
    ## Returns
    - X: The solution of the equation A.X = B
    - i: Number of iterations required to reach the tolerance
    '''
    x0=guess
    r0 = np.add(B, -1 * np.matmul(A, x0))
    d0 = np.copy(r0)
    i=1
    while True:
        alpha1 = np.matmul(np.transpose(r0), r0) / np.matmul(np.transpose(d0), np.matmul(A, d0))
        x1 = np.add(x0, alpha1[0][0]*d0)
        r1 = np.add(r0, -1 * alpha1[0][0] * np.matmul(A, d0))
        if np.linalg.norm(r1) < T and i<=len(A):
            return x1.tolist(),i
        
        elif i>len(A):
            print("Maybe the matrix A dosent satisfy the conditions for the Conjugate Gradient Method")
            return None
        else: 
            beta1 = np.matmul(np.transpose(r1), r1) / np.matmul(np.transpose(r0), r0)
            d1 = np.add(r1, beta1[0][0] * d0)
            x0 = x1
            del x1
            r0 = r1
            del r1
            d0 = d1
            del d1
            i+=1






























#####################################################################################
#                        Solution of Non- Linear Equations                             
#####################################################################################
def bracket(a0: float,b0: float,f: callable):
    '''
    # Parameters
    - a0: Lower bound of the interval
    - b0: Upper bound of the interval
    # Returns
    - a0: Lower bound of the bracketed interval
    - b0: Upper bound of the bracketed interval
    '''
    n=0
    while f(a0)*f(b0)>=0:
        if abs(f(a0))>abs(f(b0)):
            b0=b0+1.5*(b0-a0)
        else:
            a0=a0-1.5*(b0-a0)       
    return(a0,b0)


def bisection(f: float,a0: float,b0: float,T: float,ifbracket: bool=False):
    '''
    # Bisection Method
    ## Parameters
    - f: Function to find the root     
    - a0: Lower bound of the interval to find the root
    - b0: Upper bound of the interval to find the root
    - T: Tolerance
    - ifbracket: If True, the function will bracket teh interval before using the bisection method
    ## Returns
    - c0: The root of the function
    - count: Number of iterations required tolerance
    '''
    if ifbracket==True:
        a0,b0=bracket(a0,b0,f)
    count=0
    while (abs(b0-a0))>T:
        c0=(a0+b0)/2
        if f(a0)*f(c0)>0:
            a0=c0
        else:
            b0 = c0 
        count+=1      
    return c0,count 



## Newton-Raphson for multivariable is not implemented
def newton_raphson_single(f: float,fd: float,x0: float,T: float):
    '''
    # Newton-Raphson Method for Single Variable
    ## Parameters
    - f: Function to find the root    
    - fd: Derivative of the function f    
    - x0: Initial guess
    - T: Tolerance

    ## Returns
    - xn1: The root of the function
    - count: Number of iterations required to reach the tolerance
    '''
    count=0
    xn=x0
    xn1=xn-(f(xn)/fd(xn))
    while True:
        xn=xn1
        xn1=xn-(f(xn)/fd(xn)) 
        count+=1
        if abs(xn1-xn)<T:
            break          
    return xn1,count+1  


def secant_method(f: float,x0: float,x1: float,tol: float):
    '''
    # Secant Method
    ## Parameters
    - x0: 1st guess
    - x1: 2nd guess
    - f: Function to find the root
    - tol: Tolerance
    ## Returns
    - x2: The root of the function
    - step: Number of iterations required to reach the tolerance
    '''
    x2=x1-((f(x1)*(x1-x0))/(f(x1)-f(x0)))
    step=1
    while abs(x2-x1)>tol:
        if step>100:
            raise ValueError("The roots are not converging")
        else:
            x0=x1
            x1=x2
            x2=x1-f(x1)*(x1-x0)/(f(x1)-f(x0))
            step+=1
    return x2,step

def regula_falsi(f,a0,b0,T,ifbracket=False):
    '''
    # Regula Falsi Method
    ## Parameters
    - f: Function to find the root
    - a0: Lower bound of the interval to find the root
    - b0: Upper bound of the interval to find the root
    - T: Tolerance
    ## Returns
    - c0: The root of the function
    - count: Number of iterations required to reach the tolerance
    '''
    if ifbracket==True:
        a0,b0=bracket(a0,b0,f)
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

def fixed_point_single(g: float,x0: float,tol: float):
    '''
    # Fixed Point Method for Single Variable
    ## Parameters
    - g: Function to find the root where the roots to be find is f(x)=0 from where g(x)=x is deducted.
    - x0: Initial guess
    - tol: Tolerance
    ## Returns (x1,step)
    - x1: The root of the function
    - step: Number of iterations required to reach the tolerance
    '''
    x1=g(x0)
    step=1
    while abs(x1-x0)>tol:
        if step>100:
            print("The roots are not converging")
            break
        else:
            x0=x1
            x1=g(x0)
            step+=1
    return x1,step

def fixed_point_multi(glist,x0list,tol):
    '''
    # Fixed Point Method for Multi Variable set of equations
    ## Parameters
    - glist: List of functions to find the root where the roots to be find is f(x)=0 from where g(x)=x is deducted.
    - x0list: List of initial guesses
    - tol: Tolerance
    ## Returns (x0list,step)
    - x0list: The list of roots of the function
    '''
    if len(glist)!=len(x0list):
        raise IndexError("The number of functions and initial guesses are not equal")
    else:
        for i in range(len(glist)):
            x0list[i] = (glist[i](x0list))
        step=1
        flag=1
        while flag==1:
            if step>100:
                print("The roots are not converging")
                return x0list,step
            else:
                temp = x0list[:]

                for i in range(len(glist)):
                    x0list[i] = (glist[i](x0list))
                step+=1

            for j in range(len(x0list)):
                if abs(temp[j] - x0list[j]) / x0list[j] < tol:
                    flag = 0
                else:
                    flag = 1
                    break
        return x0list,step

#####################################################################################
#                              Numerical Integration                             
#####################################################################################
def midpoint(f: float,a: float,b: float,n: int):
    '''
    # Midpoint Method
    ## Parameters
    - f: Function to be integrated
    - a: Lower limit of the integral
    - b: Upper limit of the integral
    - n: Number of intervals
    ## Returns
    - I: The value of the integral
    '''
    h=(b-a)/n
    I=0
    x=a
    while x<b-h/2:
        I+=h*f(x+h/2)
        x+=h
    return I


def trapezoidal(f: float,a: float,b: float,n: int):
    '''
    # Trapezoidal Method
    ## Parameters
    - f: Function to be integrated
    - a: Lower limit of the integral
    - b: Upper limit of the integral
    - n: Number of intervals
    ## Returns (I)
    - I: The value of the integral
    '''
    h=(b-a)/n
    I=0
    x=a
    while x<b:
        I+=(h/2)*(f(x)+f(x+h))
        x+=h
    return I


def simpsons(f: callable,a: float, b: float, n: int):
    '''
    # Simpson's Method
    ## Parameters
    - f: Function to be integrated
    - a: Lower limit of the integral
    - b: Upper limit of the integral
    - n: Number of intervals (must be even for getting accurate results)
    '''    
    if n%2!=0:
        print("Value obtained maynot be accurate as n is not even. Please enter an even value of n.")
    h=(b-a)/n
    k=0.0
    x=a + h
    for i in range(1,int(n/2) + 1):
        k += 4*f(x)
        x += 2*h

    x = a + 2*h
    for i in range(1,int(n/2)):
        k += 2*f(x)
        x += 2*h
    return (h/3)*(f(a)+f(b)+k)


class Gaussian_Quadrature: 
    '''
    # Gaussian Quadrature (Class)
    ## Parameters
    - f: Function to be integrated
    - a: Lower limit of the integral
    - b: Upper limit of the integral
    - degree: Degree of the LEgendre Polynomial
    ## Returns
    - val: The value of the integral
    '''
    def __init__(self,f,a,b,degree):
        self.a = a
        self.b = b
        self.N = degree
        self.f = f   

    def Pn(self,x,n):
        '''
        # Legendre Polynomial
        ## Parameters
        - x: Variable
        - n: Degree of the polynomial
        ## Returns
        - The value of the Legendre Polynomial
        '''
        if n == 0:
            return 1
        elif n == 1:
            return x
        else:
            return ((2*n-1)*x*self.Pn(x,n-1)-(n-1)*self.Pn(x,n-2))/n

    def Pn_drvtv(self,x,n):
        '''
        # Derivative of Legendre Polynomial
        ## Parameters
        - x: Variable
        - n: Degree of the polynomial
        ## Returns
        - The value of the derivative of the Legendre Polynomial
        '''
        if n == 0:
            return 0
        elif n == 1:
            return 1
        else:
            return (n*(x*self.Pn(x,n)-self.Pn(x,n-1)))/(1-x**2)
    
    def find_legendre_roots(self):
        '''
        # Finding the roots of the Legendre Polynomial
        ## Returns
        - The roots of the Legendre Polynomial of the given degree
        '''
        n=self.N
        num_roots=self.N
        roots = []
        for i in range(1, num_roots + 1):
            guess = np.cos((2*i - 1) * np.pi / (2 * num_roots))
            result = root(self.Pn, guess, args=(n,), jac=self.Pn_drvtv, method='hybr')
            if result.success:
                roots.append(result.x[0])
        return roots
    

    def find_weights(self):
        """
        # Finding the weights for the Gaussian Quadrature   
        ## Returns
        - The weights for the Gaussian Quadrature as a list
        """
        n=self.N
        roots=self.find_legendre_roots()
        weights=[]
        for i in range(n):
            w=2/((1-roots[i]**2)*(self.Pn_drvtv(roots[i],n))**2)
            weights.append(w)
        return weights


    def integrate(self):
        '''
        Algorithm for the Gaussian Quadrature
        '''
        a=self.a
        b=self.b
        n=self.N
        f=self.f
        sum=0
        weights=self.find_weights()
        roots=self.find_legendre_roots()
        for i in range(n):
            y=((b-a)*0.5*roots[i])+((b+a)*.5)
            weightf=weights[i]*f(y)
            sum+=weightf
        val=(b-a)*0.5*sum
        return val     



def monte_carlo(f: float,a: float,b: float,N: int,seed: int):
    '''
    # Monte Carlo Integration
    ## Parameters
    - f: Function to be integrated
    - a: Lower limit of the integral
    - b: Upper limit of the integral
    - N: Number of random numbers to be generated
    - seed: Seed for the random number generator
    ## Returns
    - F: The value of the integral
    '''
    p=rng(seed)
    F=0
    for i in range(N):
        k=p.gen()
        k=((b-a)*(k/32768))+a
        F+=((b-a)*f(k))/N   
    return F   

def monte_carlo_error(f: float,a: float,b: float,N: int,seed: int):
    '''
    # Monte Carlo Integration
    ## Parameters
    - f: Function to be integrated
    - a: Lower limit of the integral
    - b: Upper limit of the integral
    - N: Number of random numbers to be generated
    - seed: Seed for the random number generator
    '''
    rn=rng(seed)
    F=0
    F1=0
    for i in range(N):
        p=rn.gen()
        p=((b-a)*(p/32768))+a
        F+=f(p)
        F1+=pow(f(p),2)  
    return (F1/N)-pow((F/N),2) 

#####################################################################################
#                              Solving dy/dx = f(x,y)                             
#####################################################################################
def forward_euler(dx: float,x_ini: float,t_ini: float,t_final: float,N: int):
    '''
    # Forward Euler Method
    for solving the differential equation dy/dx = f(x,y)
    also called explicit Euler method
    ## Parameters
    - dx: The function f(x,y): dy/dx = f(x,y)
    - x_ini: Initial value of y such that y(t_ini) = x_ini
    - t_ini: Initial value of x
    - t_final: Final value of x
    - N: Number of steps to divide the interval [t_ini,t_final]
    ## Returns
    - xlist: List of x values
    - ylist: List of y values satisfying the function dy/dx = f(x,y)
    '''
    dt=(t_final-t_ini)/N
    xlist=[]
    ylist=[]
    t=t_ini
    while t<=t_final:
        xlist.append(t)
        ylist.append(x_ini)
        x_ini+=dt*dx(t,x_ini)
        t+=dt
    return xlist,ylist

def backward_euler(f: float,y0: float,x0: float,xf: float,num_points: int):
    '''
    # Backward Euler Method
    for solving the differential equation dy/dx = f(x,y)
    ## Parameters
    - f: The function f(x,y): dy/dx = f(x,y)
    - y0: Initial value of y such that y(x0) = y0
    - x0: Initial value of x
    - xf: Final value of x
    - num_points: Number of steps to divide the interval [x0,xf]
    ## Returns
    - x_values: List of x values
    - y_values: List of y values satisfying the function dy/dx = f(x,y)
    '''
    h = (xf - x0) / num_points
    x_values = np.linspace(x0, xf, num_points + 1)
    y_values = np.zeros(num_points + 1)
    y_values[0] = y0

    for i in range(1, num_points + 1):
        # Use backward Euler method formula: y[i] = y[i-1] + h * f(x[i], y[i])
        y_values[i] = y_values[i - 1] + h * f(x_values[i], y_values[i - 1])

    return x_values, y_values



def predictor_corrector(dybydx: float,y0: float,x0: float,x_f: float,N: int):
    '''
    # Predictor Corrector Method
    for solving the differential equation dy/dx = f(x,y)
    ## Parameters
    - dybydx: The function f(x,y): dy/dx = f(x,y)   
    - y0: Initial value of y such that y(x0) = y0
    - x0: Initial value of x
    - x_f: Final value of x
    - N: Number of steps to divide the interval [x0,xf]
    ## Returns
    - xlist: List of x values
    - ylist: List of y values satisfying the function dy/dx = f(x,y)
    '''
    h=(x_f-x0)/N
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

def RK2_solve(dybydx: float,y0: float,x0: float,xf: float,N: int):
    '''
    # Runge-Kutta 2nd Order Method
    for solving the differential equation dy/dx = f(x,y)
    ## Parameters
    - dybydx: The function f(x,y): dy/dx = f(x,y)
    - y0: Initial value of y such that y(x0) = y0
    - x0: Initial value of x
    - xf: Final value of x
    - N: Number of steps to divide the interval [x0,xf]
    ## Returns
    - xlist: List of x values
    - ylist: List of y values satisfying the function dy/dx = f(x,y)
    '''
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

def RK4_solve(dybydx: float,y0: float,x0: float,x_f: float,N: int):
    '''
    # Runge-Kutta 4th Order Method
    for solving the differential equation dy/dx = f(x,y)
    ## Parameters
    - dybydx: The function f(x,y): dy/dx = f(x,y)
    - y0: Initial value of y such that y(x0) = y0
    - x0: Initial value of x
    - x_f: Final value of x
    - N: Number of steps to divide the interval [x0,xf]
    ## Returns
    - xlist: List of x values
    - ylist: List of y values satisfying the function dy/dx = f(x,y)
    '''
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

def RK4_solve_coupled(fnlist,x0,y0s,limit,h):
    '''
    # Runge-Kutta 4th Order Method for Coupled Equations
    ## Parameters
    - fnlist: List of functions to be solved that is dy_i/dx = f_i(x,t) where x is a list of variables.
    - x0: the initial value of t or x (acc to qn)
    - y0s: the value of each of the solution at x0
    - limit: The limit to which the plot is to be made
    - h: step size
    ## Returns
    - datT: List of x values or t values
    - datY: List of List of y values for each variable y_i
    -
    '''
    limit -= h/2 
    n = len(y0s) 
    k1 = [0 for i in range(n)]
    k2 = [0 for i in range(n)]
    k3 = [0 for i in range(n)]
    k4 = [0 for i in range(n)]
    tys= [0 for i in range(n)] 
    datY = []
    for i in range(n):
        datY.append([y0s[i]])
    datT = [x0]
    while x0 < limit:
        for i in range(n):
            k1[i] = h*fnlist[i](y0s,x0)    
        for i in range(n):
            tys[i] = y0s[i] + (k1[i] / 2)
        for i in range(n):
            k2[i] = h*fnlist[i](tys, (x0 + (h/2)))
        for i in range(n):
            tys[i] = y0s[i] + (k2[i] / 2)
        for i in range(n):
            k3[i] = h*fnlist[i](tys, (x0 + (h/2)))   
        for i in range(n):
            tys[i] = y0s[i] + k3[i]
        for i in range(n):
            k4[i] = h*fnlist[i](tys, (x0 + h))
        for i in range(n):
            y0s[i] += ((k1[i] + (2 * k2[i]) + (2 * k3[i]) + (k4[i])) / 6)
        x0 += h
        for i in range(n):
            datY[i].append(y0s[i])
        datT.append(x0)
    return datT, datY


def shooting_solve(fns,x0,y0,x1,y1,tol,h,guess1=0):  
    '''
    # Shooting Method
    ## Parameters
    - fns: List of functions to be solved that is dy_i/dx = f_i(x,t) where x is a list of variables.
    - x0: i nitial value of t or x (acc to qn)
    - y0: v alue of the solution at x0
    - x1: final value of t or x (acc to qn)
    - y1: value of the solution at x1
    - guess1: The initial guess for the second variable
    - tol: The tolerance for the solution
    - h: step size
    ## Returns
    - X: List of x values or t values
    - Y: List of List of y values for each variable y_i
    '''  
    if guess1 == 0:
        guess1 = (y1-y0)/(x1-x0)

    X,Y = RK4_solve_coupled(fns,x0,[y0,guess1],x1,h)
    ye1 = Y[0][-1]
    if abs(ye1 - y1) < tol:
        return X,Y
    if ye1 < y1:
        guess1side = -1
    else :
        guess1side = 1
    guess2 = guess1 + 2   
    X,Y =RK4_solve_coupled(fns,x0,[y0,guess2],x1,h)
    ye2 = Y[0][-1]
    if ye2 < y1:
        guess2side = -1
    else :
        guess2side = 1
    while guess1side * guess2side != -1:

        if abs(y1-ye2) > abs(y1-ye1):
            guess2 = guess1 - abs(guess2-guess1)
        else:
            guess2 += abs(guess2-guess1)
        X,Y = RK4_solve_coupled(fns,x0,[y0,guess2],x1,h)
        ye2 = Y[0][-1]
        if ye2 < y1:
            guess2side = -1
        else :
            guess2side = 1
    i = 0
    while True:
        newguess = guess1 + (((guess2 - guess1)/(ye1 - ye2))*(y1 - ye2))
        i += 1
        X,Y = RK4_solve_coupled(fns,x0,[y0,newguess],x1,h)
        yvalnew = Y[0][-1]
        if abs(yvalnew - y1) <tol:
            break
        if yvalnew < y1:
            guess1 = newguess
            ye1 = yvalnew
        else:
            guess2 = newguess
            ye2 = yvalnew
    return X,Y


def finite_element_solve(f: callable,x_i: float, y_i:float,x_f: float,y_f: float,y_f_prime: float, N: int):
    '''
    # Finite Element Method
    for solving the differential equation d2y/dx2 = f(x,y) with boundary conditions y(x_i) = y_i and y'(x_f) = y_f
    ## Parameters
    - f: The function f(x,y) in the differential equation
    - x_i: Initial value of x
    - y_i: Initial value of y
    - x_f: Final value of x
    - y_f: Final value of y
    - y_f_prime: value of y' at x=x_i
    - N: Number of steps to divide the interval [x_i,x_f]
    ## Returns
    - x: List of x values
    - y: List of y values
    '''
    x = np.linspace(x_i, x_f, N+1)
    h = (x_f - x_i) / (N+1)
    A = 2*np.eye(N-1) + np.diagflat([-1 for i in range(N-2)],-1) + np.diagflat([-1 for i in range(N-2)],1)
    A = A.tolist()
    B = []
    for i in range(len(A)):
        if i == 0:
            B.append([(-h**2)*f(x[i],y_i)  + y_i])
        elif i == len(A)-1:
            B.append(f(x[i],y_f) - y_f)
    return A,None
    #not complete








def semi_implicit_euler_solve(f,g,x0,v0,t0,t_max,step_size):
    '''
    # Semi-Implicit Euler Method
    ## Parameters:
    - f: f(v,t):
        dx/dt = f(v,t)
    - g: g(x,t)
        dv/dt = g(x,t)
    - x0: initial position: x(t0) = x0
    - v0: initial velocity: v(t0) = v0
    - t0: initial time
    - t_max: final time:
        The to which the Solution is to be calculated
    - step: The size of the interval
    ## Returns:
    - xlist: List of x values
    - vlist: List of v values
    - tlist: List of t values
    The function returns a 3-tuple of lists containing the x, v and t values respectively
    '''
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


def verlet_solve(a: float,x0: float,v0: float,t0: float,t_max: float,h: float):
    '''
    # Verlet Method
    ## Parameters
    - a: The function a(x) which gives the acceleration of a(x(t)) = F(x(t))/m
    - x0: initial position: x(t0) = x0
    - v0: initial velocity: v(t0) = v0
    - t0: initial time
    - t_max: final time:
        The to which the Solution is to be calculated
    - h: The size of the interval that (t0,t_max) is divided into
    ## Returns   
    - xlist: List of x values
    - vlist: List of v values
    - tlist: List of t values 
    '''
    xlist=[]
    vlist=[]
    tlist=[]
    x=x0
    t=t0
    v=v0
    xlist.append(x)
    vlist.append(v)
    tlist.append(t)
    x1=(x)+(h*v)+(0.5*h*h*a(x))
    v1=(x1-x)/h
    t=t+h
    xlist.append(x1)
    vlist.append(v1)
    tlist.append(t)
    while t<=t_max:
        x2=(2*x1)-(x)+(h*h*a(x1))
        v=(x2-x)/(2*h)
        x=x1
        x1=x2
        t=t+h
        xlist.append(x)
        vlist.append(v)
        tlist.append(t)
    return xlist,vlist,tlist    

def velocity_verlet_solve(a: callable,x0: float,v0: float,t0: float,t_f: float,h: float):
    '''
    # Velocity Verlet Method
    ## Parameters
    - a: The function a(x) which gives the acceleration of a(x(t)) = F(x(t))/m
    - x0: initial position: x(t0) = x0
    - v0: initial velocity: v(t0) = v0
    - t0: initial time
    - t_f: final time:
        The to which the Solution is to be calculated
    - h: The size of the interval that (t0,t_max) is divided into
    ## Returns
    - xlist: List of x values
    - vlist: List of v values
    - tlist: List of t values    
    '''
    xlist=[x0]
    vlist=[v0]
    tlist=[t0]
    i=1
    while t0<t_f:
        v_half=v0+(0.5*h*a(x0))
        x1= x0 +v_half*h
        v1= v_half+(0.5*h*a(x1))
        xlist.append(x1)
        vlist.append(v1)
        t0+=h
        tlist.append(t0)
        x0=x1
        v0=v1
        i+=1
    return xlist,vlist,tlist 

def leap_frog_solve(F: callable,pi0: float,x_i: float,x_f: float,t_i: float, t_f: float,N: int):
    '''
    # Leap Frog Method
    Used to solve the hamiltons equation of motion
                    d2x/dt2 = A(x,t)
    '''
    t = np.linspace(t_i, t_f, 2*N)
    dt = (t_f - t_i) / N
    x = np.zeros(2*N)
    pi = np.zeros(2*N)
    x[0] = x_i
    pi[0] = pi0
    pi[1] = pi[0] - dt*F(x[0],t[0])
    x[2] = x[0] + dt*pi[1]

    return pi

# LEAP FROG NOT DONE



#####################################################################################
#                              Solving Heat Equation                             
#####################################################################################
'''
Here we will be solving the heat equation of the form:
    du(x,t)/dt = alpha * d^2u(x,t)/dx^2
    where T is the temperature and alpha is the thermal diffusivity
'''
def pde_explicit_solve(g: callable,a: callable, b: callable, x0: float, x_m: float, t0: float, t_m: float, N_x: int, N_t: int,req_time_step: int,iflist=True):
    '''
    # Explicit Finite Difference Method
    for solving the heat equation
    ## Parameters
    - g: Initial condition function u(x,t=0) = g(x)
    - a: Boundary condition function u(x=0,t) = a(t)
    - b: Boundary condition function u(x=x_m,t) = b(t)
    - x0: Initial value of x
    - x_m: Final value of x
    - t0: Initial value of t
    - t_m: Final value of t
    - N_x: Number of steps to divide the interval [x0,x_m]
    - N_t: Number of steps to divide the interval [t0,t_m]
    - req_time_step: The time step to which the solution is to be calculated
    - iflist: If True, the function will return the list of u values, if False, the function will return u as a column matrix or a vector
    ## Returns
    - x: List of x values
    - t: List of t values
    - u: List of List of u values or vector depending on the value of iflist
    '''
    hx = (x_m - x0) / N_x
    ht = (t_m - t0) / N_t
    x=[x0 + i*hx for i in range(1,N_x)]
    alpha = ht / (hx**2)

    if alpha > 0.5:
        raise ValueError("The value of alpha should be less than 0.5")
    u = [[g(i)] for i in x]
    A=[[0 for i in range(N_x-1)] for j in range(N_x-1)]

    for i in range(len(A)):
        for j in range(len(A[i])):
            if i==j:
                A[i][j]=1+2*alpha
            elif abs(i-j)==1:
                A[i][j]=-alpha

    A1 = Get_Gauss_jordan_inv(A)
    del A
    An = np.linalg.matrix_power(A1,req_time_step)   
    del A1
    v_req = np.matmul(An,u).tolist()
    del An
    v_req.insert(0,[a(t0)])
    v_req.append([b(t0)])
    x.insert(0,x0)
    x.append(x_m)
    ulist=[]
    if iflist==False:
        return x,v_req,[t0 + i*ht for i in range(N_t+1)]
    else:
        for i in range(len(v_req)):
            ulist.append(v_req[i][0])        
        return x,ulist,[t0 + i*ht for i in range(req_time_step+1)]
    

def crank_nicolson(g: callable,a: callable, b: callable, x0: float, x_m: float, t0: float, t_m: float, N_x: int, N_t: int,req_time_step: int,iflist=True,k=1):
    '''
    # Crank Nicolson Method
    for solving the heat equation of the form u_xx = k*u_t
    ## Parameters
    - g: Initial condition function u(x,t=0) = g(x)
    - a: Boundary condition function u(x=0,t) = a(t)
    - b: Boundary condition function u(x=x_m,t) = b(t)
    - x0: Initial value of x
    - x_m: Final value of x
    - t0: Initial value of t
    - t_m: Final value of t
    - N_x: Number of steps to divide the interval [x0,x_m]
    - N_t: Number of steps to divide the interval [t0,t_m]
    - req_time_step: The time step to which the solution is to be calculated
    - iflist: If True, the function will return the list of u values, if False, the function will return u as a column matrix or a vector
    - k: The thermal diffusivity
    ## Returns
    - x: List of x values
    - t: List of t values
    - u: List of List of u values or vector depending on the value of iflist
    '''
    hx = (x_m - x0) / N_x
    ht = (t_m - t0) / N_t
    x=[x0 + i*hx for i in range(1,N_x)]
    alpha = (ht / (hx**2))/k  
    u = [[g(i)] for i in x]
    B = [[0 for i in range(N_x-1)] for j in range(N_x-1)]    
    I = [[0 for i in range(N_x-1)] for j in range(N_x-1)]   
    for i in range(len(B)):
        for j in range(len(B[i])):
            if i==j:
                B[i][j]=2*alpha
                I[i][j]=2
            elif abs(i-j)==1:
                B[i][j]=-1*alpha

    matrix1=[[I[i][j]-B[i][j] for j in range(N_x-1)] for i in range(N_x-1)]
    matrix2=[[I[i][j]+B[i][j] for j in range(N_x-1)] for i in range(N_x-1)] 
    matrix21=Get_Gauss_jordan_inv(matrix2)
    del matrix2
    matrix3=np.matmul(matrix21,matrix1)     
    del matrix1,matrix21
    matrix4=np.linalg.matrix_power(matrix3,req_time_step)
    del matrix3
    v_req = np.matmul(matrix4,u).tolist()
    del matrix4
    v_req.insert(0,[a(t0)])
    v_req.append([b(t0)])
    x.insert(0,x0)
    x.append(x_m)
    ulist=[]
    if iflist==False:
        return x,v_req,[t0 + i*ht for i in range(N_t+1)]
    else:
        for i in range(len(v_req)):
            ulist.append(v_req[i][0])        
        return x,ulist,[t0 + i*ht for i in range(req_time_step+1)]
    

def du_fort_frankel_solve(g: callable,a: callable,b: callable,x_i: float,x_f: float,t_i: float,t_f: float,Nx: int, Nt: int):
    '''
    # Du Fort Frankel Method
    for solving the heat equation of the form u_xx = k*u_t
    ## Parameters
    - g: Initial condition function u(x,t=0) = g(x)
    - a: Boundary condition function u(x=0,t) = a(t)
    - b: Boundary condition function u(x=x_m,t) = b(t)
    - x_i: Initial value of x
    - x_f: Final value of x
    - t_i: Initial value of t
    - t_f: Final value of t
    - Nx: Number of steps to divide the interval [x_i,x_f]
    - Nt: Number of steps to divide the interval [t_i,t_f]
    ## Returns
    - ulist: List of List of u values
    - x: List of x values
    '''
    hx = (x_f - x_i) / Nx
    ht = (t_f - t_i) / Nt
    alpha = ht / (hx**2)
    x=[(x_i + i*hx) for i in range(0,Nx+1)]
    ulist = np.zeros((Nx+1,Nt+1))
    for i in range(Nx+1):
        ulist[i][0] = g(x[i])
    for i in range(Nt+1):
        ulist[0][i] = a(t_i + i*ht)
        ulist[Nx][i] = b(t_i + i*ht)

    a1 = (1 - 2*alpha)/(1 + 2*alpha)
    a2 = 2*alpha/(1 + 2*alpha)
    return ulist


#####################################################################################
#                              Solving Poisson/Laplace Equation                             
#####################################################################################
def poisson_laplace(rho, x_i, x_f, y_i, y_f, u_iy, u_fy, u_xi, u_xf, N):
    '''
    Parameters:
    - rho: Function rho(x,y) del^2 u = -rho(x,y)
    - x_i: Initial x value
    - x_f: Final x value
    - y_i: Initial y value
    - y_f: Final y value
    - u_iy: Function u_iy(x)
    - u_fy: Function u_fy(x)
    '''
    '''
    defining the grid N+2 x N+2
    '''
    x = np.linspace(x_i, x_f, N+2)
    y = np.linspace(y_i, y_f, N+2)
    hx = (x_f - x_i)/(N + 1)
    hy = (y_f - y_i)/(N + 1)
    if hx != hy:
        raise ValueError("The grid is not square")
    h = hx 
    A = np.zeros((N**2,N**2))
    '''
    Defining the matrix A
    '''
    for i in range(N**2):
        A[i, i] = 4
        if i == 0:
            A[i, i+1]=-1
            A[i, i+N]=-1
        elif i < N:
            A[i, i-1]=-1
            A[i, i+1]=-1
            A[i, i+N]=-1
        elif i < (N**2-N):
            A[i, i-1]=-1
            A[i, i+1]=-1
            A[i, i-N]=-1
            A[i, i+N]=-1
        elif i < (N**2-1):
            A[i, i-1]=-1
            A[i, i+1]=-1
            A[i, i-N]=-1
        else:
            A[i, i-1] = -1
            A[i, i-N] = -1
    '''
    Defining the matrix B
    '''
    B = []
    for i in range(1,N+1):
        for j in range(1,N+1):
            sum = rho(x[i], y[j]) * h**2
            if i == 0:
                sum += u_xi(y[j])
            if i == N:
                sum += u_xf(y[j])
            if j == 0:
                sum += u_iy(x[i])
            if j == N:
                sum += u_fy(x[i])
            B.append(sum)    
    B = np.array(B)[:,None]
    A=A.tolist()
    B=B.tolist()
    u =  gauss_jordan_solve(A,B)
    u = np.array(u).reshape((N,N))
    u = np.append(u_iy(y[1:-1,None]), u, axis = 1)
    u = np.append(u, u_fy(y[1:-1, None]), axis = 1)
    u = np.append([u_xi(x)], u, axis = 0)
    u = np.append(u, [u_xf(x)], axis = 0)
    '''
    x : Grid points in the x direction
    y : Grid points in the y direction
    u : array Solution to the Poisson equation
    '''
    return x, y, u


#####################################################################################
#                              Solving Wave Equation
#####################################################################################

def wave_solve(x0,t0,u0,hx,ht,k=1):
    '''
    # Wave Equation Solver
    This solves the equation of the form u_xx = k*u_tt
    '''
    alpha = (ht / (hx**2))

    pass




#####################################################################################
#                              Eigenvalues/Eigenvectors                             
#####################################################################################

def power_method_find(A :list,x0: list,tol = 1e-6):
    '''
    # Power Method
    This function finds the largest eigenvalue and the corresponding eigenvector

    ## Condition
    - n x n matrix A has n linearly independent eigenvectors
    - Eigenvalues can be ordered in magnitude : |λ1| > |λ2| > · · · > |λn|. The λ1 is called the dominant eigenvalue and the corresponding eigenvector is the dominant eigenvector of A.

    ## Paremeters
    - A: The matrix for which the eigenvalues and eigenvectors are to be found
    - x0: The initial guess for the eigenvector
    - tol: The tolerance for the solution
    ## Returns
    - eigval: The largest eigenvalue
    - eigvec: The corresponding eigenvector
    '''
    A=np.array(A)
    x0=np.array(x0)
    x_copy = np.copy(x0)
    lam_0 = np.matmul(np.matmul(np.linalg.matrix_power(A,2),x0).T,np.matmul(np.linalg.matrix_power(A,1),x0))/np.matmul(np.matmul(np.linalg.matrix_power(A,1),x0).T,np.matmul(np.linalg.matrix_power(A,1),x0))
    lam_1 = np.matmul(np.matmul(np.linalg.matrix_power(A,3),x0).T,np.matmul(np.linalg.matrix_power(A,2),x0))/np.matmul(np.matmul(np.linalg.matrix_power(A,2),x0).T,np.matmul(np.linalg.matrix_power(A,2),x0))
    i=3
    while abs(lam_1-lam_0)>tol:
        lam_0 = lam_1
        lam_1 = np.matmul(np.matmul(np.linalg.matrix_power(A,i+1),x0).T,np.matmul(np.linalg.matrix_power(A,i),x0))/np.matmul(np.matmul(np.linalg.matrix_power(A,i),x0).T,np.matmul(np.linalg.matrix_power(A,i),x0))
        i+=1

    eigval = lam_1
    eigvec = np.matmul(np.linalg.matrix_power(A,i-1),x_copy)
    norm = np.linalg.norm(eigvec)
    eigvec = eigvec/norm
    return eigval,eigvec,i  



def QR_factorize(A: list):
    '''
    # QR Factorization
    This function finds the QR factorization of a matrix A

    ## Parameters
    - A: The matrix to be factorized
    ## Returns
    - Q: The orthogonal matrix
    - R: The upper triangular matrix
    '''
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R






#####################################################################################
#                                 Data Fitting                             
#####################################################################################

def linear_fit(xlist: list,ylist: list,elist: list):
    '''
    # Linear Regression
    This function finds the best fit line for a given set of data points
    Finds the fit for the equation y = a + bx
    ## Parameters
    - xlist: The x-coordinates of the data points
    - ylist: The y-coordinates of the data points
    - elist: The error in the y-coordinates of the data points. If elist=False, the function will assume that the error is 1 for all data points
    ## Returns
    - slope: The slope of the best fit line
    - intercept: The y-intercept of the best fit line
    - chi_sq: The chi-squared value of the best fit line
    '''
    # Raise an error if the lengths of xlist, ylist, and elist are not the same
    if len(xlist) != len(ylist):
        raise ValueError('The length of xlist, ylist, and elist must be the same')
    
    # If elist is False, assume that the error is 1 for all data points
    if elist == False:
        elist = [1]*len(xlist)
    # Convert the lists to numpy arrays
    xlist = np.array(xlist)
    ylist = np.array(ylist)
    elist = np.array(elist)
    n=len(xlist)
    # Calculate the sums
    S=np.sum(1/((elist)**2))
    Sx = np.sum(xlist/((elist)**2))
    Sy = np.sum(ylist/((elist)**2))
    Sxx = np.sum((xlist**2)/((elist)**2))
    Syy = np.sum((ylist**2)/((elist)**2))
    Sxy = np.sum((xlist*ylist)/((elist)**2))

    # Calculate the slope and intercept
    Delta = S*Sxx - Sx**2

    intercept=(Sxx*Sy-Sx*Sxy)/Delta
    slope=(S*Sxy-Sx*Sy)/Delta
    # Calculate the error in the slope and intercept
    # error_intercept = np.sqrt(Sxx/Delta)
    # error_slope = np.sqrt(S/Delta)
    # cov = -Sx/Delta
    # Pearsen's correlation coefficient
    r_sq = Sxy/np.sqrt(Sxx*Syy) 

    return slope,intercept,np.sqrt(r_sq)





def polynomial_fit(xlist: list,ylist: list,sigma_list: list,degree: int,tol=1e-6):
    '''
    # Polynomial Fitting
    This function finds the best fit polynomial for a given set of data points
    Finds the fit for the equation y = a0 + a1*x + a2*x^2 + ... + an*x^n
    ## Parameters
    - xlist: The x-coordinates of the data points
    - ylist: The y-coordinates of the data points
    - sigma_list: The error in the y-coordinates of the data points
    - degree: The degree of the polynomial to be fit
    ## Returns
    - a: The coefficients of the best fit polynomial
    '''
    xlist = np.array(xlist)
    ylist = np.array(ylist)
    sigma_list = np.array(sigma_list)
    A_matrix = np.zeros((degree+1,degree+1))

    for i in range(degree+1):
        for j in range(degree+1):
            A_matrix[i][j] = np.sum((xlist**(i+j))/(sigma_list**2))
    B_matrix = np.zeros(degree+1)
    for i in range(degree+1):
        B_matrix[i] = np.sum((ylist*(xlist**i))/(sigma_list**2))
    # a = Gauss_seidel_solve(A_matrix.tolist(),B_matrix.tolist(),T=tol)
    a = np.linalg.solve(A_matrix,B_matrix)    
    return a


