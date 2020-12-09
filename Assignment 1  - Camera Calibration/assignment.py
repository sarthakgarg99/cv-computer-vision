import cv2 
import numpy as np



# function that performs homogenization and normarlisation and return T and U which will be 
# used for denormalisation
def normalise(xy,XYZ):
    rows,cols = xy.shape
    xy = np.vstack([xy,np.ones((1,cols))])
    XYZ = np.vstack([XYZ,np.ones((1,cols))])
    xy_centroid = np.mean(xy, axis=1)
    XYZ_centroid = np.mean(XYZ, axis =1)

    # Transformation matrix to shift centroid to origin

    T1 = np.array([[1,0,-xy_centroid[0]],[0,1,-xy_centroid[1]],[0,0,1]])
    U1 = np.array([[1,0,0,-XYZ_centroid[0]],[0,1,0,-XYZ_centroid[1]],[0,0,1,-XYZ_centroid[2]],[0,0,0,1]])
    xy_new = np.dot(T1,xy)
    XYZ_new = np.dot(U1,XYZ)

    dist_xy = np.mean(np.sqrt(np.sum(xy_new**2, axis = 0)-1))
    dist_XYZ = np.mean(np.sqrt(np.sum(XYZ_new**2, axis = 0)-1))
    
    #Find transformations so that the average L2 norm of points is sqrt(2) and sqrt(3)
    T2 = np.sqrt(2)/dist_xy
    U2 = np.sqrt(3)/dist_XYZ

    #find final transformation matrix
    T = T1*T2
    T[2][2] =1
    U = U1*U2
    U[3][3] = 1

    #normalise points by applying transformations
    xyn = np.dot(T,xy)
    XYZn = np.dot(U,XYZ)
    return xyn, XYZn, T, U


#function to perform denormalisation takes T and U as input
def denormalise(M_n,T,U):
    M = np.dot(np.linalg.inv(T),np.dot(M_n,U))
    return M


# function to perform DLT on normalised coordinate to return normalised Projection matrix
def DLT(xyn,XYZn):
    
    #returns normalised projection matrix
    num_points = xyn.shape[1]
    zero_row = np.zeros((1,4))
    
    # generate P matrix
    P = np.empty((0,12))
    for i in range(num_points):
        A11 = XYZn[:,i].T.reshape((1,4))
        A12 = zero_row
        A13 = -xyn[0,i]* XYZn[:,i].T.reshape((1,4))
        first_row = np.concatenate((A11,A12,A13), axis=1)
        A21 = zero_row
        A22 = XYZn[:,i].T.reshape((1,4))
        A23 = -xyn[1,i]* XYZn[:,i].T.reshape((1,4))
        second_row = np.concatenate((A21,A22,A23), axis=1)
        to_append = np.concatenate((first_row,second_row), axis=0)
        P = np.vstack([P,to_append])

    #perform SVD operation and extract M, the projection matrix
    U,S,VH = np.linalg.svd(P, full_matrices=True, compute_uv=True, hermitian=False)
    V = VH.T.conj()
    M_vec = V[:,-1].reshape((12,1))
    M = np.concatenate((M_vec[0:4,0].T.reshape((1,4)),M_vec[4:8,0].T.reshape((1,4)), M_vec[8:12,0].T.reshape((1,4))), axis=0)
    return M


# function to calculate intrinsic parameters 
# simple formulae used mentioned in the report
def intrinsic(M):
    a1 = (M[0, 0:3])
    a2 = (M[1, 0:3])
    a3 = (M[2, 0:3])
    
    # calculating rho, x_not, y_not, theta, alpha, beta
    rho = 1 / np.linalg.norm(a3)
    x_not = rho * rho * np.dot(a1, a3)
    y_not = rho * rho * np.dot(a2, a3)
    theta = -1 * np.dot(np.cross(a1, a3), np.cross(a2, a3))/(np.linalg.norm(np.cross(a1, a3)) * np.linalg.norm(np.cross(a2, a3)))
    theta = np.arccos(theta) * 360 / (2 * np.pi)
    alpha = rho * rho * np.linalg.norm(np.cross(a1, a3)) * np.sin(theta)
    beta = rho * rho * np.linalg.norm(np.cross(a2, a3)) * np.sin(theta)
    
    # calculating projection matrix
    K = np.zeros((3,3))
    K[0][0] = alpha
    K[0][1] = -1 * alpha * (np.cos(theta)/ np.sin(theta))
    K[0][2] = x_not
    K[1][1] = beta / np.sin(theta)
    K[1][2] = y_not
    K[2][2] = 1
    return K, rho, x_not, y_not, theta, alpha, beta

# function to calculate extrinsic parameters 
# simple formulae used mentioned in the report
def extrinsic(M):
    a1 = (M[0, 0:3])
    a2 = (M[1, 0:3])
    a3 = (M[2, 0:3])
    rho = 1 / np.linalg.norm(a3)
    
    r3 = rho * a3
    r1 = rho * rho * np.sin(theta) * np.cross(a2, a3) / beta
    r2 = np.cross(r3, r1)
    
    R = [r1, r2, r3]
    t = rho * np.dot(np.linalg.inv(K), M[0:3, 3])
    return R, t


# function used to resize the image window
# reference : https://stackoverflow.com/questions/35180764/opencv-python-image-too-big-to-display
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)




choice = input("Do you want to use existing data base? (Y/N)")

if choice == "N":
    # Taking input points from the user XYZ coordinate
    num_point = input("Enter number of points: ") 

    img_cord = np.zeros((2, int(num_point)))
    space_cord = np.zeros((3, int(num_point)))
    for i in range(0, int(num_point)):
        X = input("Enter X of this point: ")
        Y = input("Enter Y of this point: ")
        Z = input("Enter Z of this point: ")

        space_cord[0, i] = X
        space_cord[1, i] = Y
        space_cord[2, i] = Z

    cur_num = 0


    def click_event(event, x, y, flags, params): 
        global cur_num
        if(cur_num < int(num_point)):
            if event == cv2.EVENT_LBUTTONDOWN: 
                img_cord[0, cur_num] = x
                img_cord[1, cur_num] = y
                cur_num = cur_num + 1;
                cv2.circle(resize, (x,y), 5, (0,255,0), -1)
                cv2.imshow('image', resize)





    # taking input from image 
    # reference : https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
    
    img = cv2.imread('use.JPG', 1) 

    cv2.namedWindow('image')
    cv2.startWindowThread()
    resize = ResizeWithAspectRatio(img, width=1280)
    cv2.imshow('image', resize)

    cv2.setMouseCallback('image', click_event) 

    cv2.waitKey(0)

    cv2.destroyAllWindows(); 

    cv2.waitKey(10000)


else:
    # loding stored dataset
    num_point = 9
    img_cord = np.array([[ 724.,  847.,  997.,  850.,  366.,  216.,  215.,   31.,  500.], [1235.,  923.,  553.,  242., 1091.,  921.,  556.,  118.,  302.]])
    space_cord = np.array([[ 3.8,  7.6, 11.4,  7.6,  0. ,  0. ,  0. ,  0. ,  0. ],[ 0. ,  0. ,  0. ,  0. ,  7.6, 11.4, 11.4, 15.2,  3.8], [ 0. ,  7.6, 15.2, 22.8,  3.8,  7.6, 15.2, 22.8, 22.8]])
    

# performing steps mentioned in the flow chart using the above defined function

print("=========================================================================")
print()
print("The XYZ matrix is : ")
print(space_cord)
print()
print("=========================================================================")
print()
print("The xy matrix is : ")
print(img_cord)
print()
print("=========================================================================")
print()
print("Performing normalisation .... ")
xy_n, XYZ_n, T, U = normalise(img_cord, space_cord)
print("Normalisation Done.")
print()
print("=========================================================================")
print()
print("Normalised XYZ: ")
print(XYZ_n)
print()
print("=========================================================================")
print()
print("Normalised xy: ")
print(xy_n)
print()
print("=========================================================================")
print()
print("T: ")
print(T)
print()
print("=========================================================================")
print()
print("U: ")
print(U)
print()
print("=========================================================================")
print()
print("Performing DLT on normalised xy and XYZ to get normalised M(Projection matrix) ....")
M_n = DLT(xy_n, XYZ_n)
print("DLT Done")
print()
print("=========================================================================")
print()
print("Normalised Projection Matrix")
print(M_n)
print()
print("=========================================================================")
print()
print("Performing denormalisation on normalised M using T and U ..... ")
M = denormalise(M_n, T, U)
print("Denormalisation Done")
print()
print("=========================================================================")
print()
print("Plotting predicted points with blue and original points with green... (press any key on the image window to continue...)")
XYZ = np.vstack([space_cord,np.ones((1,space_cord.shape[1]))])
XYZ = np.hstack([XYZ, np.array([[1], [0], [1], [1]])])
ans = np.dot(M, XYZ)
xa = ans[0,:] / ans[2,:]
ya = ans[1,:] / ans[2,:]
img = cv2.imread('use.JPG', 1) 
cv2.namedWindow('image')
cv2.startWindowThread()
resize = ResizeWithAspectRatio(img, width=1280)
for i in range(int(num_point)):
    x = img_cord[0][i]
    y = img_cord[1][i]
    cv2.circle(resize, (int(x),int(y)), 12, (0,255,0), -1)
for i in range(int(num_point)):
    x = xa[i]
    y = ya[i]
    cv2.circle(resize, (int(x),int(y)), 8, (255,0,0), -1)
cv2.imshow('image', resize)
cv2.waitKey(0)

cv2.destroyAllWindows(); 
num_point *= 10
cv2.waitKey(1000)
print()
print("=========================================================================")
print()
print("Calculating Root Mean Square Error: ")
rms = 0
for i in range(9):
    rms += (xa[i] - img_cord[0][i]) ** 2
    rms += (ya[i] - img_cord[1][i]) ** 2
rms /= num_point
rms = rms ** (1/2)
print("Root mean Square Error is: " + str(rms))
print()
print("=========================================================================")
print()
print("Calculating Intrinsic Parameters ...")
K, rho, x_not, y_not, theta, alpha, beta = intrinsic(M)
print("\u03C1: " + str(rho))
print("(x\N{SUBSCRIPT ZERO}, y\N{SUBSCRIPT ZERO}): (" + str(x_not) + ", " + str(y_not) + ")")
print("\u03B8: " + str(theta))
print("\u03B1: " + str(alpha))
print("\u03B2: " + str(beta))
print("Calibration matrix(K): ")
print(K)
print()
print("=========================================================================")
print()
print("Calculating Extrinsic Parameters ... (R and t)")
R, t = extrinsic(M)
print("R: ")
print(np.array(R))
print("t: ")
print(t)
print()
print("=========================================================================")
print()
print("Plotting all the points ... (press any key to continue)")

XYZ = np.zeros((4,0))
for i in range(5):
    for j in range(7):
        XYZ = np.hstack([XYZ, np.array([[i * 3.8], [0], [j * 3.8], [1]])])
        XYZ = np.hstack([XYZ, np.array([[0], [i * 3.8], [j * 3.8], [1]])])
img = cv2.imread('use.JPG', 1) 
cv2.namedWindow('image')
cv2.startWindowThread()
resize = ResizeWithAspectRatio(img, width=1280)
ans = np.dot(M, XYZ)
xa = ans[0,:] / ans[2,:]
ya = ans[1,:] / ans[2,:]
for i in range(int(70)):
    x = xa[i]
    y = ya[i]
    cv2.circle(resize, (int(x),int(y)), 8, (255,0,0), -1)
cv2.imshow('image', resize)
cv2.waitKey(0)
cv2.destroyAllWindows(); 
cv2.waitKey(1000)

print()
print("=========================================================================")





