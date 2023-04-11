import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
import time
import torch
import math

def map_inference_dpp_greedy(kernel_matrix, max_length, epsilon=1E-10):
    """
    Our proposed fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(max(0, di2s[selected_item]))
        if di_optimal == 0:
            di_optimal = epsilon
        # start_x_time = time.time()
        elements = kernel_matrix[selected_item, :]
        # print("elements: ", time.time() - start_x_time)
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        # if di2s[selected_item] < epsilon:
            # break
        selected_items.append(selected_item)
 
    S = np.sort(np.array(selected_items))
    return S, np.linalg.det(kernel_matrix[S.reshape(-1, 1), S.reshape(1, -1)]) 

def map_inference_dpp_local_search_2(L, k, verbose=False):
    start_time = time.time()
    greedy_sol, greedy_prob = map_inference_dpp_greedy(L, k)
    greedy_time = time.time() - start_time

    if verbose:
        print("Prob: ", greedy_prob)

    cur_sol = greedy_sol.copy()
    cur_prob = greedy_prob
    obj_greedy = greedy_prob

    N = L.shape[0]
    all_idx = np.array(range(N))
    ns_idx = np.setdiff1d(all_idx, cur_sol)

    # L = np.arange(100).reshape(10, 10)
    L_S = L[cur_sol[:, np.newaxis], cur_sol]
    it = 0

    while True:
        start_iter_time = time.time()

        idx = np.array(range(len(cur_sol)))
        best_removal_idx = 0
        best_removal_prob = 0

        for i in range(len(cur_sol)):
            # cur_sol[i], cur_sol[-1] = cur_sol[-1], cur_sol[i]
            idx[i], idx[-1] = idx[-1], idx[i]
            L_Se = L_S[idx[:-1, np.newaxis], idx[:-1]]
            prob = np.linalg.det(L_Se)

            if prob > best_removal_prob:
                best_removal_idx = i
                best_removal_prob = prob
        obj_loc = best_removal_prob

        brid = best_removal_idx
        br = cur_sol[brid]

        best_neighbors = cur_sol.copy()
        best_add = -1
        best_neighbors_prob = cur_prob
        localopt = True

        for v in ns_idx:
            cur_sol[brid] = v
            L_S[brid, :] = L[v, cur_sol]
            L_S[:, brid] = L[cur_sol, v]
            prob = np.linalg.det(L_S)

            if prob > best_neighbors_prob:
                best_neighbors_prob = prob
                best_add = v
                localopt = False

        if verbose:
            print("Iter {}:".format(it))
            print("remove item: ", br)
            print("add item: ", best_add)
            print("best_neighbors_prob: ", best_neighbors_prob)

        if not localopt:
            cur_sol[brid] = best_add
            cur_prob = best_neighbors_prob
            L_S[brid, :] = L[best_add, cur_sol]
            L_S[:, brid] = L[cur_sol, best_add]
            ns_idx = np.setdiff1d(all_idx, cur_sol)
        else:
            cur_sol = best_neighbors
            cur_prob = best_neighbors_prob
            break
        it += 1

    ls_time = time.time() - start_time
    return cur_sol, obj_loc, ls_time, greedy_sol, greedy_prob, greedy_time

