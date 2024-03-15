import numpy as np
#printing a matrix
def printmatrix(A):
    for i in range(0,len(A)):
        print("\n")
        for j in range(0,len(A[i])):
            print(A[i][j],",",end='')
    print("\n")    

#determinant of matrix(also checks for singular matrix)
def determinant(A):
    for i in range(len(A)): #the index of each column 
        rs = 0  #initialising the row swap as 0
        if A[i][i] == 0:
            a = i
            for j in range(i + 1,len(A)):
                if abs(A[j][i]) > abs(A[a][i]):
                    a = j
            if a == i: #if max element is still zero, a is not changed; no unique solution
                print("The matrix is singular!")
                return None
            A[i],A[a] = A[a], A[i]
            rs += 1

        #making other elements below b zero
        for k in range(i + 1,len(A)):
            if A[k][i] != 0:
                b = A[k][i] / A[i][i]
                for j in range(i,len(A[k])): 
                    A[k][j] = A[k][j] - (A[i][k] * b) 
    p = 1
    if rs % 2 == 1:
        p = -1
    for i in range(len(A)):
        p *= A[i][i]
    if p == 0:
        print("Singular matrix")
        return None
#matrix multiplication
def mat_multiplication(A,B): 
    #finding AB matrix

    #verifiying matrix multiplication is possible (cols of A = rows of B)
    if len(A[0]) != len(B):
        return None

    #creating the sum matrix with zeroes
    sum = []
    for i in range(len(A)): #no. of row in sum is row of A
        sum.append(list(0 for i in range( len(B[0]) ))) #no. of col in sum is col of B

    #filling the sum matrix
    for row in range(len(sum)):
        for col in range(len(sum[0])):
            #finding each term in sum matrix
            temp_sum = 0
            for i in range(len(B)):
                temp_sum += A[row][i] * B[i][col]
                sum[row][col]=temp_sum

    return sum
#taking dot product
def mat_dot(A,B):
    #returns the dot product of two column matrices
    #if len(A) != len(B) or len(A[0]) != 1 or len(B[0]) != 1: #works only if column matrice of same length
        #return None
    dotprdct = 0
    for row in range(len(A)):
        dotprdct += (A[row][0] * B[row][0])
    return dotprdct
#taking transpose
def get_transpose(matrix):
    rows = len(matrix)
    columns = len(matrix[0])
    matrix_T = []
    for j in range(columns):
        row = []
        for i in range(rows):
           row.append(matrix[i][j])
        matrix_T.append(row)
    return matrix_T
#taking inverse-see gauss jordan,if the b matric is identitiy
#checking symmetry
def symcheck(A):
    for i in range(0,len(A)):
        for j in range(0,len(A[i])):  
            if A[i][j]!=A[j][i]:
                return False
            else:
                continue    
    if not False:
        return True
    else:
        return False 
#checking diagonally dominant
def diagonal_dom_check(m) :
    n=len(m)
    for i in range(0, n) :        
        sum = 0
        for j in range(0, n) :
            sum = sum + abs(m[i][j])    
        sum = sum - abs(m[i][i])
        if (abs(m[i][i]) < sum) :
            return False 
    return True
#making diagonally dominant
def make_DD(A,B):
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


#to solve linear equations
#gauss jordan-makes an augmented matrix,then make one A matrix ref
def getone(sd,pp):
    for i in range(len(sd[0])):
        if sd[pp][pp] != 1:
            q00 = sd[pp][pp]

            for j in range(len(sd[0])):
                sd[pp][j] = sd[pp][j] / q00
def getzero(sd,r, c):
    for i in range(len(sd[0])):
        if sd[r][c] != 0:
            q04 = sd[r][c]

            for j in range(len(sd[0])):
                sd[r][j] = sd[r][j] - ((q04) * sd[c][j])

def gauss_jordan(sd):
    for i in range(len(sd)):
        getone(sd,i)

        for j in range(len(sd)):
            if i != j:
                getzero(sd,j, i)
    return sd            

#inverse of a matrix using gauss jordan
def get_inv(A):
    A=gauss_jordan(A)
    inv=[[0 for i in range(int(len(A[1])/2))] for i in range(int(len(A[1])/2))]
    for i in range(len(inv)):
        for j in range(len(inv)):
            inv[i][j]=A[i][int((len(A)/2))+j+1]
    return inv   


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




#Gauss jordan to solve:
def gauss_jordan_solve(A,B):
  
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

#cholesky,when factorization of Hermitian, positive definite matrix (which often is the case in physics) into a product of L and L transpose
def Chol_dec(A):
    from math import sqrt
    if symcheck(A) == False:
        print('Non symmetric!')
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
        
    return A
def forward_backward_cholesky(A,B):
    Y = []#forward-L-Y,backward-LT-X
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

def Cholesky_solve(A,B): 
    A = Chol_dec(A)
    if A is None:
        print('Cholesky Solve not possible!')
        return None
    return forward_backward_cholesky(A,B)


#solution using iterative methods,print the number of iteration also
#Jacobi
    #A = D + (L + U)
    #A · x = b ⇒D + (L + U)· x = b
    #diagonally dominant matrices only
    #input a guess and tolerance,guess can be anything,maybe zero matrix
def jacobi_method(A,B,guess,tol):
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

def Jacobi_with_DD_Check(A,B,guess,T):
    if diagonal_dom_check(A)==True:
        return jacobi_method(A,B,guess,T)
    else:
        print(" The given Matrix is not Diagonallly Dominant") 
#gauss-seidel
    #Convergence is only guaranteed if the matrix is either strictly diagonally dominant or symmetric and positive definite.
    #A=L* + U
    #L* times (k+1)th iteration of X =b - U times k-th iteration of X
    #like jacobi,question matrices and tolerance as input
    #no guess required
def gauss_seidel(A,B,tolerance):
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

def GSM_with_DD_check(A,B,T):
    if diagonal_dom_check(A)==True:
        return gauss_seidel(A,B,T)
    else:
        print(" The given Matrix is not Diagonallly Dominant")    

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


#conjugate gradient
def conjugate_gradient_1(A, b, x0, tol, max_iter=1000):
    
    A = np.array(A)
    b = np.array(b)
    x = np.array(x0)
    r = b - np.dot(A, x)
    p = r
    rsold = np.dot(r, r)
    
    for iter in range(max_iter):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r, r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
        
    return x, iter
def conjugate_gradient_solve(A: list,B: list,guess: list,T: float):
   
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


