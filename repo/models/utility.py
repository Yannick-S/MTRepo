import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_point_cloud(point_cloud_in, color='C0', alpha=1, arrow=None, show=True, ax=None, path=None):

    if type(point_cloud_in) == torch.Tensor:
        point_cloud = point_cloud_in.clone()
        length = point_cloud.size()[0]
        assert point_cloud.size()[1] == 3, "point_cloud does not have 3 coordinates per point"
        point_cloud = point_cloud.numpy()
    elif type(point_cloud_in) == np.ndarray:
        point_cloud = point_cloud_in.copy()
        length = point_cloud.shape[0]
        assert point_cloud.shape[1] == 3, "point_cloud does not have 3 coordinates per point"
    else:
        assert False, "point_cloud is neither of type np.ndarray or torch.Tensor, it is " + type(point_cloud) 
    

    min_x = point_cloud[:,0].min()
    min_y = point_cloud[:,1].min()
    min_z = point_cloud[:,2].min()

    max_x = point_cloud[:,0].max()
    max_y = point_cloud[:,1].max()
    max_z = point_cloud[:,2].max()

    len_x = max_x - min_x
    len_y = max_y - min_y
    len_z = max_z - min_z

    if True: 
        point_cloud[:,0] = (point_cloud[:,0] - min_x)#/(max_x - min_x) 
        point_cloud[:,1] = (point_cloud[:,1] - min_y)#/(max_y - min_y) 
        point_cloud[:,2] = (point_cloud[:,2] - min_z)#/(max_z - min_z) 

    max_max = max((max_x - min_x, max_y - min_y, max_z - min_z))

    if type(point_cloud) == np.ndarray:
        corner_point = np.array((max_max, max_max, max_max)).reshape([1,3])
        point_cloud = np.append(point_cloud, corner_point, axis=0)

    if True:
        point_cloud[:,0] = (point_cloud[:,0] + min_x)#/(max_x - min_x) 
        point_cloud[:,1] = (point_cloud[:,1] + min_y)#/(max_y - min_y) 
        point_cloud[:,2] = (point_cloud[:,2] + min_z)#/(max_z - min_z) 

    if ax == None:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_aspect('equal')

    if color=='angle':
        color = np.zeros((point_cloud.shape[0]-1,3))
        print(color.shape)
        #color[:,0]= np.arctan2(point_cloud[:-1,0], point_cloud[:-1,1])
        #color[:,1]= np.arctan2(point_cloud[:-1,1], point_cloud[:-1,2])
        #color[:,2]= np.arctan2(point_cloud[:-1,2], point_cloud[:-1,0])
        color[:,0]= point_cloud[:-1,0]
        color[:,1]= point_cloud[:-1,1]
        color[:,2]= point_cloud[:-1,2]
        color = color - color.min()
        color = color / color.max()
    #ax.scatter(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2])
    ax.scatter(point_cloud[:-1,0], point_cloud[:-1,1], point_cloud[:-1,2],color=color, alpha=alpha)
    ax.scatter(point_cloud[-1,0], point_cloud[-1,1], point_cloud[-1,2], alpha=0)

    if type(arrow) == torch.Tensor or type(arrow) == np.ndarray:
        if type(arrow) == torch.Tensor:
            length = arrow.size()[0]
            assert arrow.size(1) == 3, "ararow does not have 3 coordinates per point"
            arrow = arrow.numpy()
        elif type(arrow) == np.ndarray:
            length = arrow.shape[0]
            assert arrow.shape[1] == 3, "arrow does not have 3 coordinates per point"
        else:
            assert False, "arrow is neither of type np.ndarray or torch.Tensor, it is " + type(arrow) 

        zero = np.zeros(arrow.shape) 
        arrow = arrow/arrow.max()/2

        for i in range(length):
            color = 'C' + str(i)
            ax.quiver(0,0,0,  arrow[i,0],  arrow[i,1], arrow[i,2], color=color)
        #ax.quiver(zero[:,0], zero[:,1], zero[:,2],
            #arrow[0,:]*0.3,
            #arrow[1,:]*0.3,
            #arrow[2,:]*0.3,
            #color=['b','g','r'])
        #ax.quiver(zero[:,0], zero[:,1], zero[:,2],
        #    arrow[0,:]*0.1,
        #    arrow[1,:]*0.1,
        #    arrow[2,:]*0.1,
        #    color=['b','g','r'])

    if path:
        plt.savefig(path, format='png', dpi=1000)
        plt.clf()
        plt.cla()
    elif show:
        plt.show()
    else:
        return ax

def knn_id(x,id_0,k=20):
    x_0 = x[id_0,:]

    distances = np.zeros((x.shape[0]))
    for i in range(x.shape[0]):
        distances[i] = np.linalg.norm(x_0 - x[i])
    argsort = distances.argsort()
    return argsort[1:k+1], argsort[k+1:]

def plot_voxel(point_cloud, d=16):
    x = point_cloud.clone()
    
    knn_ids, rest = knn_id(x.numpy(), 15, k=50)

    ax = plot_point_cloud(x[knn_ids],color='C1', show=False)
    ax = plot_point_cloud(x[rest],color='C0',  alpha=0.2, show=False,ax=ax)
    plot_point_cloud(x[15].reshape(1,3),color='C2',show=True, ax=ax)

    x_max = x[:,0].max()
    x_min = x[:,0].min()
    y_max = x[:,1].max()
    y_min = x[:,1].min()
    z_max = x[:,2].max()
    z_min = x[:,2].min()

    x[:,0] = x[:,0] - x_min
    x[:,0] = x[:,0]/(x_max - x_min)

    x[:,1] = x[:,1] - y_min
    x[:,1] = x[:,1]/(y_max - y_min)

    x[:,2] = x[:,2] - z_min
    x[:,2] = x[:,2]/(z_max - z_min)

    ls = np.arange(0,1,1/d)
    ls = ls * 0.3 + 0.5
    ones = np.ones((d,d))
    r = np.zeros((d,d,d))
    g = np.zeros((d,d,d))
    b = np.zeros((d,d,d))
    for i in range(d):
        r[i,:,:] = ones*ls[i]
        g[:,i,:] = ones*ls[i]
        b[:,:,i] = ones*ls[i]
    
    final_colors = np.zeros((d,d,d,4))
    final_colors[:,:,:,0] = r
    final_colors[:,:,:,1] = g
    final_colors[:,:,:,2] = g
    final_colors[:,:,:,3] = np.ones((d,d,d))*0.5
    

    voxel = np.zeros((d,d,d))
    for id_ in range(x.size(0)):
        point = x[id_,:]
        point = point / torch.tensor((1/d,1/d,1/d))
        point = point.int()
        while point.max() == d:
            point[point.argmax()] -= 1
        voxel[point[0],point[1],point[2]] = 1
    if False:
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    v_center = torch.tensor((1/d*(i + 1/2),1/d*(j + 1/2),1/d*(k + 1/2)))
                    for id_ in range(x.size(0)):
                        dist = v_center - x[id_,:]
                        dist = max( (dist[0].abs(), dist[1].abs() , dist[2].abs()))
                        if dist <= 1/(d*2):
                            voxel[i,j,k] = 1
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxel,facecolors=final_colors, edgecolor='k')
    plt.show()
    
def get_orthog(vectors):
    out = torch.zeros(vectors.size(0),3,3)
    for i in range(vectors.size(0)):
        vec0 = vectors[i]
        if torch.norm(vec0) == 0:
            out[i,:,:] = torch.eye(3)
            continue
        
        vec0 = vec0/torch.norm(vec0)

        vec1 = torch.tensor([vec0[1],vec0[2],vec0[0]])  # find a non-parallel vector
        vec2 = torch.cross(vec0, vec1)           # find an orthogonal vector
        vec2 = vec2/torch.norm(vec2)            # normalize it
        vec1 = torch.cross(vec0, vec2)           # find the last orthogonal vector, overwrite vec1 which was only non-parallel
        vec1 = vec1/torch.norm(vec1)            # normalize it

        out[i,0,:] = vec0
        out[i,1,:] = vec1
        out[i,2,:] = vec2

    return  out
    
def get_plane(x, edge_index, k):
    #l = int(k/10) #TODO:
    l = k 

    assert edge_index.size(1) % k == 0, "Length of Edge_index must be divisible by k"

    out = torch.zeros(x.size(0),3,3)

    centers = int(edge_index.size(1)/k)
    centers2 = int(x.size(0))

    print("Delete here! I think ", centers, " and ", centers2 , " are the same... if not think!")

    for c in range(centers):
        X = x[edge_index[1,c*k:c*k+k],:]
        X = X - x[c,:]

        # TODO: optimze the next few lines. They scale everything to be within
        # [-1,1] but still keep x_0 at cetner.
        ma = torch.max(X, dim=0)[0]
        mi = torch.min(X, dim=0)[0]
        bla = torch.zeros((2,3))
        bla[0,:] = ma
        bla[1,:] = mi.abs()

        scaler = 1/ bla.max(dim=0)[0]

        X = X * scaler
        X_ = X[:l,:]
        U, S, V = torch.svd(X_)
        out[c,:,:] = V

        #if c == 0:
        if False:
            print("In Get Plane")
            print(edge_index[1,c*k:(c+1)*k])
            plot_point_cloud(x)
            plot_point_cloud(X, arrow=torch.t(V))


    return out

def graclus_out(pos, x, cluster):
    uniques = cluster.unique()
    new_nr = uniques.size(0)

    new_pos = torch.zero(new_nr, 3)
    new_x = torch.zero(new_nr, x.size(1), x.size(2))
    for i in range(new_nr):
        cluster_id = cluster[i]
        new_pos[cluster_id] = pos[i]

    return

if __name__ == "__main__":

    test_vecotrs = torch.rand([100,3])
    vects = get_orthog(test_vecotrs)
    

    quit()

    test_cloud = torch.rand([100,3])
    plot_point_cloud(test_cloud)
    
    test_cloud = np.random.rand(100,3)
    plot_point_cloud(test_cloud)