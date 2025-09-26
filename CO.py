import time
import numpy as np

def CO(current_position,fobj,lower_limit,upper_limit,its):
    n,d = current_position.shape[0],current_position.shape[1]
    m = d / 4
    g21 = m + 1
    g22 = 2 * m
    g31 = (2 * m) + 1
    g32 = 3 * m
    g41 = (3 * m) + 1
    current_fitness = 0 * np.ones((d,n))
    gx = []
    g1 = []
    g2 = []
    g3 = []
    g4 = []
    Bbest = []
    best_position = []
    it = 1
    Random = 0 + np.multiply((1 - 0),np.random.rand(d,n))
    current_position = (Random * (upper_limit - lower_limit)) + lower_limit
    for j in np.arange(1,d+1).reshape(-1):
        for i in np.arange(1,n+1).reshape(-1):
            gx[j,i] =fobj(current_position[i, :])
            current_fitness[j,i] = (gx[j,i] * (1 - current_position(j,i))) + (120 * current_position(j,i))

    for j in np.arange(1,d+1).reshape(-1):
        Bbestj,best_pointj = np.amin(current_fitness[j,:])
        Bbest = np.array([[Bbest],[Bbestj]])
        best_position = np.array([[best_position],[current_position(j,best_pointj)]])

    for i in np.arange(1,n+1).reshape(-1):
        g1 = np.array([g1,current_position(np.arange(1,m+1),i)])
        g2 = np.array([g2,current_position(np.arange(g21,g22+1),i)])
        g3 = np.array([g3,current_position(np.arange(g31,g32+1),i)])
        g4 = np.array([g4,current_position(np.arange(g41,d+1),i)])
    local_Bbest = Bbest
    local_best_position = best_position
    reflection_g1 = np.zeros((n,m))
    visibility_g1 = np.zeros((n,m))
    ct = time.time()
    while (it < its):
        it = it + 1
        AVbest = np.mean(best_position)
        r1 = 2
        r2 = - 1
        V = 1
        R = ((0 + np.multiply((1 - 0),np.random.rand(1,1))) * (r1 - r2)) + r2
        for i in np.arange(1,m+1).reshape(-1):
            for j in np.arange(1,n+1).reshape(-1):
                reflection_g1[i,j] = R * g1[i,j]
                visibility_g1[i,j] = V * (best_position[j] - g1[i,j])
        g1_new = reflection_g1 + visibility_g1
        #------------------------------------------------#
    #        Studying Group 2 of population          #
    #------------------------------------------------#
        v1 = 1.5
        v2 = - 1.5
        R = 1
        V = ((0 + np.multiply((1 - 0),np.random.rand(1,1))) * (v1 - v2)) + v2
        reflection_g2 = np.zeros((n))
        reflection_g2_1 = np.zeros((n))
        visibility_g2 = np.zeros((n))
        for i in np.arange(1,m+1).reshape(-1):
            for j in np.arange(1,n+1).reshape(-1):
                reflection_g2_1[j] = R * best_position[j]
                visibility_g2[i,j] = V * (best_position[j] - g2[i,j])
            reflection_g2 = np.array([[reflection_g2],[reflection_g2_1]])
        g2_new = reflection_g2 + visibility_g2

        v1 = 1
        v2 = - 1
        R = 1
        V = ((0 + np.multiply((1 - 0),np.random.rand(1,1))) * (v1 - v2)) + v2
        g3_new = []
        reflection_g3 = np.zeros((n))
        visibility_g3 = np.zeros((n))
        for i in np.arange(1,m+1).reshape(-1):
            for j in np.arange(1,n+1).reshape(-1):
                reflection_g3[j] = R * best_position[j]
                visibility_g3[j] = V * (best_position[j] - AVbest)
            g3_new_1 = reflection_g3 + visibility_g3
            g3_new = np.array([[g3_new],[g3_new_1]])

        g4_new = []
        for i in np.arange(1,m+1).reshape(-1):
            Random = 0 + np.multiply((1 - 0),np.random.rand(1,n))
            g4_new_1 = (Random * (upper_limit - lower_limit)) + lower_limit
            g4_new = np.array([[g4_new],[g4_new_1]])

        current_position = np.array([[g1_new],[g2_new],[g3_new],[g4_new]])

        for j in np.arange(1,d+1).reshape(-1):
            for i in np.arange(1,n+1).reshape(-1):
                gx[j,i] = (30 * np.exp(- 100 * current_position[j,i])) + 50
                current_fitness[j,i] = (gx[j,i] * (1 - current_position[j,i])) + (120 * current_position[j,i])
        for j in np.arange(1,d+1).reshape(-1):
            Bbestj,best_pointj = np.amin(current_fitness[j,:])
            Bbest = np.array([[Bbest],[Bbestj]])
            best_position = np.array([[best_position],[current_position[j,best_pointj]]])

        for i in np.arange(1,n+1).reshape(-1):
            g1 = np.array([g1,current_position[(np.arange(1,m+1),i)]])
            g2 = np.array([g2,current_position[(np.arange(g21,g22+1),i)]])
            g3 = np.array([g3,current_position[(np.arange(g31,g32+1),i)]])
            g4 = np.array([g4,current_position[(np.arange(g41,d+1),i)]])

        local_Bbest = np.array([[local_Bbest],[Bbest]])
        local_best_position = np.array([[local_best_position],[best_position]])


    best_fitness,best_fitness_pos = np.amin(local_Bbest)
    ct =time.time()-ct

    Best_position_for_best_fitness = local_best_position[best_fitness_pos]
    return best_fitness,best_fitness_pos,Best_position_for_best_fitness,ct

