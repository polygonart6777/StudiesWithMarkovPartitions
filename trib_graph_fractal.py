import numpy
from numpy import linalg  # Need for Matrix operations
from numpy.linalg import inv  # Need for inverse of a matrix
from numpy.linalg import matrix_power  # Need for powers of matrix
import matplotlib as mpl  # Need for plotting
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import randomcolor

A = numpy.array([[1, 1, 1], [1, 0, 0], [0, 1, 0]])
# Find the eigenvalues and eigvenvectors
u, v = numpy.linalg.eig(A)
# Combine the real parts of two eigenvectors to define the real matrix similar to A, working in R^3.  Needed for projection onto the contracting and expanding spaces.
real_evector = numpy.transpose(v)[0].real
im_evector_real_part = numpy.transpose(v)[1].real
im_evector_im_part = numpy.transpose(v)[1].imag
real_matrix = [real_evector, im_evector_real_part, im_evector_im_part]



def backward_paths(vertex, path_length):
    # All backwrd paths of a given path_length on the graph with edges starting at the given vertex.
    if vertex == 1:
        paths = [["edge2"], ["edge1"], ["edge3"]]
    elif vertex == 2:
        paths = [["edge4"]]
    elif vertex == 3:
        paths = [["edge5"]]
    else:
        print("Please enter vertex 1, 2 or 3")
    while len(paths[0]) < path_length:
        # Apply the for loop until paths are the desired length.
        paths1 = []
        paths2 = []
        paths3 = []
        for j in range(len(paths)):
            # For each path, check the last element and append only the edges that are allowed to follow that last element.
            # There are three possible cases for the last element of each path, corresponding to the the three vertices of the graph.
            if paths[j][-1] in {"edge1", "edge4"}:
                k = paths[j]
                paths1.extend([k + ["edge1"], k + ["edge2"], k + ["edge3"]])
            elif paths[j][-1] in {"edge2", "edge5"}:
                l = paths[j]
                paths2.append(l + ["edge4"])
            elif paths[j][-1] == "edge3":
                m = paths[j]
                paths3.append(m + ["edge5"])
            else:
                break
        paths = paths1 + paths2 + paths3
    return paths


def relabel(paths, labels):
    # Relabels each of the edges of the paths, according to a chosen label on the edges e1 to e5.
    labellings = {
        "edge1": labels[0],
        "edge2": labels[1],
        "edge3": labels[2],
        "edge4": labels[3],
        "edge5": labels[4],
    }
    label_list = [[labellings[element] for element in path] for path in paths]
    return label_list


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

def projection_contplane(vectors):
    # This function projects each of the vectors onto the contracting plane.
    # By solving a system of equations that give the desired coefficeint on the expanding eigenvector, and so by using different,
    # coefficients on the contracting eigenvectors as well.
    proj_list = [
        vector
        - numpy.linalg.solve(numpy.transpose(real_matrix), vector)[0] * real_evector
        for vector in vectors
    ]
    return proj_list


def projection_xyplane(list):
    # Projects vectors from the contracting plane onto the xy-plane.
    # rewriting using coordinates [1,0,0] and [0,0,1].
    # I chose to take basis vectors that have the second component as zero, by changing up one of the scalars (\alpha \beta where v= \alpha v1 + \beta v2)
    proj_list = []
    for i in range(len(list)):
        solution = numpy.linalg.solve(numpy.transpose(real_matrix), list[i])
        x = solution[1]
        y = solution[2] * im_evector_im_part[1] / im_evector_real_part[1]
        proj_list.append([x, y])
    return proj_list


def random_rauzy_xyplane(seed,path_length):
    
    #initiates each of the above functions. Path length should really not get any higher than 20. 
    random_rauzy_xyplane_v1 = projection_xyplane(
        projection_contplane(matrix_sum_backward(relabel(backward_paths(1, path_length), seed)))
    )
    random_rauzy_xyplane_v2 = projection_xyplane(
        projection_contplane(matrix_sum_backward(relabel(backward_paths(2, path_length), seed)))
    )
    random_rauzy_xyplane_v3 = projection_xyplane(
        projection_contplane(matrix_sum_backward(relabel(backward_paths(3, path_length), seed)))
    )

    def extract_xy(list):
        # This function breaks apart a list of 2d vectors into their x, y components and outputs the list of x's and y's.
        xlist, ylist = zip(*[list[i] for i in range(len(list))])
        return xlist, ylist

    random_xlist_v1, random_ylist_v1 = extract_xy(random_rauzy_xyplane_v1)
    random_xlist_v2, random_ylist_v2 = extract_xy(random_rauzy_xyplane_v2)
    random_xlist_v3, random_ylist_v3 = extract_xy(random_rauzy_xyplane_v3)

    # Nice axes for aspect ratio
    x_min = min(min(random_xlist_v1), min(random_xlist_v2), min(random_xlist_v3))
    y_min = min(min(random_ylist_v1), min(random_ylist_v2), min(random_ylist_v3))
    x_max = max(max(random_xlist_v1), max(random_xlist_v2), max(random_xlist_v3))
    y_max = max(max(random_ylist_v1), max(random_ylist_v2), max(random_ylist_v3))

    #Random colors for the plot
    rand_color = randomcolor.RandomColor()
    color_blue = rand_color.generate(hue="blue", count=1)
    color_green = rand_color.generate(hue="green", count=1)
    color_red = rand_color.generate(hue="red", count=1)

    
    # Plotting and saving as a buffer for the Twitter Bot code to use.
    #figs = plt.figure()
    fig, ax = plt.subplots()
    aspect_ratio = (x_max - x_min) / (y_max - y_min)
    ax.set_aspect(aspect_ratio)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.scatter(random_xlist_v1, random_ylist_v1, c=color_blue, marker=".", s=0.03)
    plt.scatter(random_xlist_v2, random_ylist_v2, c=color_green, marker=".", s=0.03)
    plt.scatter(random_xlist_v3, random_ylist_v3, c=color_red, marker=".", s=0.03)
    plt.axis("off")
    plt.title(seed)
    plt.show()


##################Testing

#seed = [[random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]for i in range(5)]
seed = [[1,0,0],[0,0,0],[0,0,0],[1,0,0],[0,0,0]]; #rauzy
#seed = [[0,0,0],[1,0,0],[0,0,0],[1,1,0],[0,0,0]]; #flipped sub
#seed = [[0,0,-3],[0,0,-3],[0,0,-3],[0,0,0],[1,4,-3]];

random_rauzy_xyplane(seed,17)

