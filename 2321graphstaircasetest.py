import numpy
from numpy import linalg  # Need for Matrix operations
from numpy.linalg import inv  # Need for inverse of a matrix
from numpy.linalg import matrix_power  # Need for powers of matrix
import matplotlib as mpl  # Need for plotting
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import randomcolor
import io # Need for the buffer
from PIL import Image

A = numpy.array([[2, 3],[1, 2]]);
A_inverse = numpy.linalg.inv(A);
# Find the eigenvalues and eigvenvectors
u, v = numpy.linalg.eig(A);
vt = numpy.transpose(v);
I = numpy.identity(2);
#eigenvectors

def backward_paths(vertex, path_length):
    # All backwrd paths of a given path_length on the graph with edges starting at the given vertex.
    if vertex == 1:
        paths = [["edge1"], ["edge2"], ["edge8"]]
    elif vertex == 2:
        paths = [["edge3"], ["edge4"], ["edge5"], ["edge6"], ["edge7"]]
    else:
        print("Please enter vertex 1 or 2")
    while len(paths[0]) < path_length:
        # Apply the for loop until paths are the desired length.
        paths1 = []
        paths2 = []
        for j in range(len(paths)):
            # For each path, check the last element and append only the edges that are allowed to follow that last element.
            if paths[j][-1] in {"edge1", "edge2", "edge3", "edge4", "edge5"}:
                k = paths[j]
                paths1.extend([k + ["edge1"], k + ["edge2"], k + ["edge8"]])
            elif paths[j][-1] in {"edge6", "edge7", "edge8"}:
                l = paths[j]
                paths2.extend([l + ["edge6"], l + ["edge7"], l + ["edge3"], l + ["edge4"], l + ["edge5"]])
            else:
                break
        paths = paths1 + paths2 
    return paths

def forward_paths(vertex, path_length):
    # All forward paths of a given path_length on the graph with edges starting at the given vertex.
    if vertex == 1:
        paths = [["edge1"], ["edge2"], ["edge3"], ["edge4"], ["edge5"]]
    elif vertex == 2:
        paths = [["edge6"], ["edge7"], ["edge8"]]
    else:
        print("Please enter vertex 1 or 2")
    while len(paths[0]) < path_length:
        # Apply the for loop until paths are the desired length.
        paths1 = []
        paths2 = []
        for j in range(len(paths)):
            # For each path, check the last element and append only the edges that are allowed to follow that last element.
            # There are three possible cases for the last element of each path, corresponding to the the three vertices of the graph.
            if paths[j][-1] in {"edge1", "edge2", "edge8"}:
                k = paths[j]
                paths1.extend([k + ["edge1"], k + ["edge2"], k + ["edge3"],k + ["edge4"],k + ["edge5"]])
            elif paths[j][-1] in {"edge3", "edge4","edge5", "edge6","edge7" }:
                l = paths[j]
                paths2.extend([l + ["edge6"], l + ["edge7"], l + ["edge8"]])
            else:
                break
        paths = paths1 + paths2 
    return paths


def relabel(paths, labels):
    # Relabels each of the edges of the paths, according to a chosen label on the edges e1 to e8.
    labellings = {
        "edge1": labels[0],
        "edge2": labels[1],
        "edge3": labels[2],
        "edge4": labels[3],
        "edge5": labels[4],
        "edge6": labels[5],
        "edge7": labels[6],
        "edge8": labels[7],
    }
    label_list = [[labellings[element] for element in path] for path in paths]
    return label_list

#labeled_paths is a list of relabled paths, we want to attach a vector to each element of this list.
def matrix_sum_backward(labeled_paths):
    # Converts each backward path into a vector.
    # For each path, we apply the matrix A_inverse to each label of the path according to the index of the label within the path.
    # Then, we sums all of these up.
    summed = [
        sum(
            [
                matrix_power(A, i).dot(labeled_paths[j][i])
                for i in range(len(labeled_paths[j]))
            ]
        )
        for j in range(len(labeled_paths))
    ]
    return summed

def matrix_sum_forward(labeled_paths):
    # Converts each backward path into a vector.
    # For each path, we apply the matrix A_inverse to each label of the path according to the index of the label within the path.
    # Then, we sums all of these up.
    summed = [
        sum(
            [
                matrix_power(A_inverse, i+1).dot(labeled_paths[j][i])
                for i in range(len(labeled_paths[j]))
            ]
        )
        for j in range(len(labeled_paths))
    ]
    return summed


B = numpy.array([[.5, -.866025],[-.288675, .5]]);

def projection_contplane(vectors):
    # This function projects each of the vectors onto the contracting line.
    
    proj_list = [
       B.dot(vector)
        for vector in vectors
    ]
    return proj_list

    
C= numpy.array([[.5, .866025],[.288675, .5]]);

def projection_expplane(vectors):
    # This function projects each of the vectors onto the contracting plane.
    # By solving a system of equations that give the desired coefficeint on the expanding eigenvector, and so by using different,
    # coefficients on the contracting eigenvectors as well.
    proj_list = [
        -C.dot(vector)
        for vector in vectors
    ]
    return proj_list

def projection_contplane_onevec(vector):
    new=numpy.dot(vt[1],vector)/numpy.dot(vt[1],vt[1])*vt[1]
    return new

def projection_expplane_onevec(vector):
    new=numpy.dot(vt[0],vector)/numpy.dot(vt[0],vt[0])*vt[0]
    return new


def extract_xy(list):
    # This function breaks apart a list of 2d vectors into their x, y components and outputs the list of x's and y's.
    xlist, ylist = zip(*[list[i] for i in range(len(list))])
    return xlist, ylist

#seed = [[1,0],[2,1],[0,0],[1,1],[2,2],[1,1],[0,0],[1,0]] #regular
seed = [[3,0],[3,1],[0,0],[1,0],[2,0],[1,0],[0,0],[2,0]] #Lshaped
#seed = [[2,0],[2,1],[0,0],[1,0],[2,2],[1,1],[0,0],[1,0]]  #notregular
#seed = [[-3,-2],[-1,-1],[-3,-1],[-2,-1],[-1,0],[-3,-1],[-2,-1],[-3,-2]] #alternate point
#seed =[[0,0],[0,1],[-10,-11],[0,11],[-10,0]] #regular and separated


random_rauzy_xyplane_v1_unstable = projection_expplane(matrix_sum_forward(relabel(forward_paths(1, 5), seed)));
random_rauzy_xyplane_v2_unstable = projection_expplane(matrix_sum_forward(relabel(forward_paths(2, 5), seed)));

random_rauzy_xyplane_v1_stable = projection_contplane(matrix_sum_backward(relabel(backward_paths(1, 5), seed)));
random_rauzy_xyplane_v2_stable = projection_contplane(matrix_sum_backward(relabel(backward_paths(2, 5), seed)));

test_v1=[(unstable_element + stable_element) for stable_element in random_rauzy_xyplane_v1_stable for unstable_element in random_rauzy_xyplane_v1_unstable]
test_v2=[(unstable_element + stable_element) for stable_element in random_rauzy_xyplane_v2_stable for unstable_element in random_rauzy_xyplane_v2_unstable]


random_xlist_v1, random_ylist_v1=extract_xy(test_v1)
random_xlist_v2, random_ylist_v2=extract_xy(test_v2)

random_xlist_v1 = [x % 1 for x in random_xlist_v1]
random_ylist_v1 = [x % 1 for x in random_ylist_v1]
random_xlist_v2 = [x % 1 for x in random_xlist_v2]
random_ylist_v2 = [x % 1 for x in random_ylist_v2]

x_min = min(min(random_xlist_v1), min(random_xlist_v2))
y_min = min(min(random_ylist_v1), min(random_ylist_v2))
x_max = max(max(random_xlist_v1), max(random_xlist_v2))
y_max = .5


fig, ax = plt.subplots()
aspect_ratio = (x_max - x_min) / (y_max - y_min)
ax.set_aspect(1)
plt.xlim(-.5, 1.5)
plt.ylim(-.5, 1.5)
plt.scatter(random_xlist_v1, random_ylist_v1, c='b', marker=".", s=.00003)
plt.scatter(random_xlist_v2, random_ylist_v2, c='#03C04A', marker=".", s=.00003)
#box

plt.plot([0,0], [0,1], c='black', alpha=.5)
plt.plot([0,1], [1,1],c='black', alpha=.5)
plt.plot([1,1], [1,0],c='black', alpha=.5)
plt.plot([0,1], [0,0],c='black',alpha=.5)
plt.show()
