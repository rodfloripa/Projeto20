#!/usr/bin/python
# -*- coding: utf-8 -*-


from collections import namedtuple
import numpy as np
import math
from scipy.spatial import distance
from numpy import linalg as la
import cvxpy as cp


Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])
    
    
    customers = []
    for i in range(1, customer_count+1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer(i-1, int(parts[0]), float(parts[1]), float(parts[2])))
    cust_pts = [(i.x,i.y) for i in customers]
    #the depot is always the first customer in the input
    depot = customers[0] 

    def solve_cvxpy(demand_lst,points_visited,capacity,vehicle_count):
        # Returns all feasible solutions that respect the vehicle capacity
        # and attend a fraction of total customers(points_visited)
        # Points_visited varies from >=1 to a number where the problem is not feasible anymore,
        # because it exceeds the vehicle capacity.
        n = len(demand_lst)
        m = vehicle_count
        StateMatrix = cp.Variable((m,n),boolean = True)
        Route_demand = StateMatrix@np.array(demand_lst)

        objective = cp.Maximize( cp.sum(Route_demand)  ) 
        constraints = [  StateMatrix.T @ np.ones(m) == 1, StateMatrix @ demand_lst  <= capacity,
                    StateMatrix @ np.ones(n) >= points_visited, StateMatrix[0,0] == 1 ]

        # solve the problem            
        prob = cp.Problem(objective, constraints)
        result = prob.solve(cp.SCIP,verbose =False)
        return(StateMatrix.value)
    
    def return_distance(state_matrix):
        # Returns the total distance from the state matrix
        route_lst = [[] for i in range(0,vehicle_count)]
        for i in customers_lst2:
            i.distance = 0
        
        for i in range(0,vehicle_count):
            route_lst[i].append((np.where(state_matrix[i] > 0)[0]))
            route_lst[i] = route_lst[i][0]
            
        route_lst[0] = np.delete(route_lst[0], 0)
        depot.distance = 0
        
        for i in range(0,vehicle_count):
            actual_customer = depot
        
            for j in route_lst[i]:
                next = [k for k in customers_lst2 if k.index == j]
                actual_customer.connect(next[0])
                
                actual_customer = next[0]
                
            actual_customer.connect(depot)
            
        return(state_matrix,depot.distance)
    
    def order_pts(state_vector,cust_pts):
        # Get a list of all customer coordinates and a state vector selecting a fraction
        # of customers. Returns the state matrix with the order that gives the smallest
        # route length for the selected customers.
        # Miller–Tucker–Zemlin formulation- https://en.wikipedia.org/wiki/Travelling_salesman_problem
        num_pts = sum([i>0.5 for i in state_vector])
        route_lst = [[] for i in range(0,num_pts)]
        for i in range(0,1):
            route_lst[i].append((np.where(state_vector > 0.5)[0]))
            route_lst[i] = route_lst[i][0]
        cust_pts2 = []
        if route_lst[0][0] != 0:
           cust_pts2.append(cust_pts[0])
        [cust_pts2.append(cust_pts[i]) for i in route_lst[0]]
        distances = distance.cdist(cust_pts2, cust_pts2, 'euclidean')
        
        N = len(cust_pts2)
        # VARS
        x = cp.Variable((N, N), boolean=True)
        u = cp.Variable(N, integer=True)

        # CONSTRAINTS
        constraints = []

        for j in range(N):
            indices = np.hstack((np.arange(0, j), np.arange(j + 1, N)))
            constraints.append(sum(x[indices, j]) == 1)
        for i in range(N):
            indices = np.hstack((np.arange(0, i), np.arange(i + 1, N)))
            constraints.append(sum(x[i, indices]) == 1)

        for i in range(1, N):
            for j in range(1, N):
                if i != j:
                    constraints.append(u[i] - u[j] + N*x[i, j] <= N-1)
        constraints.append(x @ np.ones(N) == 1)

        # OBJ
        obj = cp.Minimize(cp.sum(cp.multiply(distances, x)))

        # SOLVE
        prob = cp.Problem(obj, constraints)
        result = prob.solve(cp.GUROBI,verbose = False)
              
        return(x.value,cust_pts2)
    
    

    def order_pts2(state_vector,cust_pts):
        # Calls order_pts and returns the select customer points by state_vector
        # ordered to give the smallest route length for selected customers.
        num_pts = sum([i>0.5 for i in state_vector])
        route_lst = [[] for i in range(0,num_pts)]
        route_lst[0].append(0)
        for i in range(0,1):
                route_lst[i].append((np.where(state_vector > 0)[0]))
                route_lst[i] = route_lst[i][1]
                
        # order points
        state_matrix,cust_pts = order_pts(state_vector,cust_pts)
        # prepare route data
        final_lst = []
        if route_lst[0][0] != 0:
           route_lst[0] = np.insert(route_lst[0], 0, 0)
        for i in range(0,len(cust_pts)):
            final_lst.append((i,(np.where(state_matrix[i] > 0.5)[0])[0]))
        aux_list2 = []
        aux_list3 = []
        j = 0
        for i in range(1,len(final_lst)):
            [aux_list2.append(i) for i in final_lst if i[0]==j]
            j = aux_list2[-1][1]
            aux_list3.append(aux_list2[-1][1])
        final_lst = aux_list3.copy()
        final_lst.insert(0,0)
        # calculating route length
        dist = 0
        for i in range(0,len(final_lst)-1): 
            dist = dist+np.linalg.norm(np.array(cust_pts[final_lst[i]])-np.array(cust_pts[final_lst[i+1]]))
        dist = dist+np.linalg.norm(np.array(cust_pts[final_lst[0]])-np.array(cust_pts[final_lst[i+1]]))
        obj = dist
        final_lst2 = []
        [final_lst2.append(route_lst[0][i]) for i in final_lst]
        
        if final_lst2[0] != 0:
            final_lst2.insert(0,0)
        
        return(final_lst2,obj)

    class Node(object):
        # A point with route distance accumulation,demand, index, point coordinates.

        # point attributes
        def __init__(self, demand, index, point):
            self.demand = demand
            self.distance = 0
            self.index = index
            self.point = (point)
            
        # connecting points, summing distance,demand, and assigning these values to the next point
        def connect(self,next_node):
            next_node.distance = self.distance+la.norm(np.array(next_node.point)-np.array(self.point))
            next_node.demand = self.demand + next_node.demand
        
        # constraints
        def constraints(self):
            return [self.demand <= vehicle_capacity]
    
    customers_lst = []
    for i in customers:
        customers_lst.append(Node(i.demand,i.index,(i.x,i.y)))
    demand_lst = []
    # Make a copy of customers list,because the first will be modified 
    customers_lst2 = customers_lst.copy()
    for i in range(0,customer_count):
        demand_lst.append([j.demand for j in customers_lst if j.index == i][0])
     
    constraints = []
    route_lst = [[] for i in range(0,vehicle_count)]
    # start route with depot
    depot = Node(customers[0].demand,customers[0].index,(customers[0].x,customers[0].y))
    actual_customer = customers_lst[0]
    dist = 0
    new_vehicle_capacity = vehicle_capacity
    total_distance = 0
    
    for k in range(0,vehicle_count):
        vehicle_capacity = new_vehicle_capacity
        actual_customer = depot
        for j in range(0,len(customers_lst)):
            # find eligible points that respect the capacity
            eligible_pts = [i for i in customers_lst if vehicle_capacity-i.demand >= 0]
            if len(eligible_pts) == 0:
                break
            
            eligible_pts = [i.point for i in eligible_pts]
            dist_vec = distance.cdist([actual_customer.point],eligible_pts)
            # don't use the 0 distance
            dist_vec[dist_vec == 0] = 1000000
            min_ =  np.argmin(dist_vec[0][:])
            arr = np.array(dist_vec[0][:])
            # find the second nearest point
            point = eligible_pts[arr.argsort()[:1][0]]
                        
            next = [i for i in customers_lst if i.point == point]
            if len(next) >1:
                min_cap = max([i.demand for i in next ])
                next = [i for i in next if i.demand == min_cap]
            # subctract demand from vehicle capacity
            vehicle_capacity = vehicle_capacity-next[0].demand
            # connect to second nearest point
            actual_customer.connect(next[0])
            # remove actual point from list
            [customers_lst.remove(i) for i in customers_lst if i.point == actual_customer.point]
            
            # add point to route_lst
            for i in customers_lst:
                if i.point == point:
                    actual_customer = i
                    route_lst[k].append(i.index)
                    
                    break
        # end route, obj has the total route length
        actual_customer.connect(depot)
        [customers_lst.remove(i) for i in customers_lst if i.point == actual_customer.point]
        obj = depot.distance 
       
    route_lst_2 = route_lst.copy()
    route_lst_2 = [item for sublist in route_lst_2 for item in sublist]
    
    # Minimize the route length, reordering points
    state_vector = []
    [state_vector.append(0) for i in range(0,len(cust_pts))]
    state_vector = np.array(state_vector)
    route_lst2 = []
    obj = 0
    for j in range(0,vehicle_count):
        state_vector = []
        [state_vector.append(0) for i in range(0,len(cust_pts))]
        state_vector = np.array(state_vector)
        for i in route_lst[j]:
            state_vector[i] = 1
        
        cst_pts = []
        [cst_pts.append(cust_pts[i] for i in route_lst[j])]
        if sum(state_vector) >=1:
            lst,obj1 = order_pts2(state_vector ,cust_pts)
            route_lst2.append(lst)
            obj = obj1 + obj 
        else:
            route_lst2.append([0]) 
    cvxpy = 0
    
    # Use CVXPY if the first solution doesn't visit all the customers.
    # First, solve_cvxpy gives the feasible solutions that respect the vehicle
    # capacity.The function cvxpy_results gives the length for all feasible solutions,
    # then order_pts2 minimizes the routes length for the feasible solution 
    # that gave the smallest route length.
    if len([item for sublist in route_lst for item in sublist])+1 != customer_count:
        cvxpy = 1
        print('using cvxpy')

        cvxpy_results = []
        for points_visited in range(0,customer_count):
            try:
                state_matrix = solve_cvxpy(demand_lst,points_visited,new_vehicle_capacity,vehicle_count)
                cvxpy_results.append([return_distance(state_matrix)])
                               
            except:
                break
        # Select the feasible solution that gave the smallest route length
        min_ = [i[0][1] for i in cvxpy_results]
        state_matrix = cvxpy_results[min_.index(sorted(min_)[0])][0][0]
        route_lst = [[] for i in range(0,vehicle_count)]
        obj = 0
        # Order points to give smallest route length
        for i in range(0,vehicle_count):
            print('ordering route ',i)
            state_matrix_l = state_matrix[i]
            if sum(state_matrix_l) > 0.5:
                route_lst[i],obj1 = order_pts2(state_matrix_l,cust_pts)
                dist = 0
                for j in range(0,len(route_lst[i])-1): 
                    dist = dist+np.linalg.norm(np.array(cust_pts[route_lst[i][j]])-np.array(cust_pts[route_lst[i][j+1]]))
                    
                dist = dist+np.linalg.norm(np.array(cust_pts[0])-np.array(cust_pts[route_lst[i][j+1]]))
                #print('total',dist)
                obj = obj+obj1
                #print(obj) 
            else:
                pass
    
    # If second solution was not selected, use the results from first    
    if not cvxpy:
        route_lst = route_lst2.copy()
    vehicle_tours = {}
    for i in range(0,vehicle_count):
        try:
            route_lst[i].remove(0)
        except:
            pass
        
        vehicle_tours[i] = route_lst[i]
       
    # prepare the solution in the specified output format
    outputData = '%.2f' % obj + ' ' + str(0) + '\n'
    for v in range(0, vehicle_count):
        outputData += str(depot.index) + ' ' + ' '.join([str(customer) for customer in vehicle_tours[v]]) + ' ' + str(depot.index) + '\n'

    return outputData


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:

        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

