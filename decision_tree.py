####################################################################################
# Name: Kaushik Nadimpalli
# Course: CS6375.002
# Assignment 2: Fixed-Depth choice Tree Implementation
####################################################################################

import os
import math
import pandas as pd
import numpy as np
import graphviz
from matplotlib import pyplot as plt

# Paritioning data/dictionary
def partition(x):
    output_dict = {}
    for i in np.unique(x):
        output_dict.update({i : (x == i).nonzero()[0]})
    return output_dict

# Calculating Entropy
def entropy(y):
    h_z = 0.0
    y_size = len(y)
    dict_y = partition(y)

    for key in list(dict_y.keys()):
        p_z = dict_y[key].size/y_size
        h_z -= p_z*math.log2(p_z)
    return h_z

# Calculating Information Gain i.e. mutual information
def mutual_information(x, y):
    h_y = entropy(y)
    x_val, x_count = np.unique(x,return_counts = True)

    h_y_x = 0.0
    for i in range(len(x_val)):
        p_x = x_count[i].astype('float')/len(x)
        h_y_x += p_x * entropy(y[x==x_val[i]])
    return h_y - h_y_x

# Implementing the ID3 algorithm below
def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    my_tree = {}

    if attribute_value_pairs is None:
        attribute_value_pairs = np.vstack([[(i, v) for v in np.unique(x[:, i])] for i in range(x.shape[1])])

    y_vals, y_count = np.unique(y, return_counts=True)

    if len(y_vals) == 1:
        return y_vals[0]

    if len(attribute_value_pairs) == 0 or depth == max_depth:
        return y_vals[np.argmax(y_count)]

    info_gain = np.array([mutual_information(np.array(x[:, i] == v).astype(int), y) for (i, v) in attribute_value_pairs])
    (attr, val) = attribute_value_pairs[np.argmax(info_gain)]
    p = partition(np.array(x[:, attr] == val).astype(int))
    del_ind = np.all(attribute_value_pairs == (attr, val), axis=1)
    attribute_value_pairs = np.delete(attribute_value_pairs, np.argwhere(del_ind), 0)

    for key, indices in p.items():
        subsetx = x.take(indices, axis=0)
        subsety = y.take(indices, axis=0)
        choice = bool(key)
        my_tree[(attr, val, choice)] = id3(subsetx, subsety, attribute_value_pairs=attribute_value_pairs, max_depth=max_depth, depth=depth + 1)

    return my_tree

# Predict classification label and return the predicted label of x
def predict_example(x, tree):
    for val_split, sub_tree in tree.items():
        attr_num = val_split[0]
        attr_val = val_split[1]
        choice = val_split[2]

        if choice == (x[attr_num] == attr_val):
            if type(sub_tree) is dict:
                label = predict_example(x, sub_tree)
            else:
                label = sub_tree
            return label

# Computes averge error (will be later used for plotting)
def compute_error(y_true, y_pred):
    length = len(y_true)
    error = [y_true[i] != y_pred[i] for i in range(length)]
    return sum(error) / length

def pretty_print(tree, depth=0):
    """
    Pretty prints the choice tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))

def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for choice trees produced by scikit-learn
        * to_graphviz() (function is in this file) for choice trees produced by  your code.
    DO NOT MODIFY THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a choice tree.\nUse tree.export_graphviz()'
                        'for choice trees produced by scikit-learn and to_graphviz() for choice trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)

def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_choice = split_criterion[2]

        if not split_choice:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_choice:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_choice:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid


if __name__ == '__main__':

# Monk Dataset Below
# Different test/train versions for Monk are tested (for trees and plots, please refer report)
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    training_error = {}
    testing_error = {}

    for val in range(1,11):

        choice_tree = id3(Xtrn, ytrn, max_depth=3)
        print(choice_tree)
        pretty_print(choice_tree)

        # PNG Visualization
        dot_str = to_graphviz(choice_tree)
        render_dot_file(dot_str, './my_tree_depth_3')

        # Error Computation
        y_pred = [predict_example(x, choice_tree) for x in Xtst]
        testing_error[val] = compute_error(ytst, y_pred)
        training_error[val] = compute_error(ytrn, y_pred)

    print("Training Error is:")
    print(training_error[val])
    print("Testing Error is:")
    print(testing_error[val])

"""
# Part C - Printing the confusion matrix and checking the tree's accuracy

    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(ytst, y_pred))
    print(classification_report(ytst, y_pred))
"""


"""
# Part B - Printing the plots - Avg test/train error over different depths 1-10

    fig = plt.figure()
    plt.plot(list(tst_err.keys()), list(tst_err.values()), marker='v', linewidth=3, markersize=12)
    plt.plot(list(trn_err.keys()), list(trn_err.values()), marker='o', linewidth=3, markersize=12)
    plt.xlabel('values of depth',fontsize=16)
    plt.ylabel('training error/test error',fontsize=16)
    plt.xticks(list(trn_err.keys()), fontsize=12)
    plt.legend(['Test Error Avg','Training Error Avg'],fontsize=16)
    fig.savefig('Monk-1.png', dpi=fig.dpi)
"""


"""
# Part E - Anotehr Dataset - Hayes Roth (5 attributes and 160 instances)

# https://archive.ics.uci.edu/ml/datasets/Hayes-Roth
    M = pd.read_csv('hayes-roth.data',delimiter=',',header=None,dtype=int)
    T = pd.read_csv('hayes-roth.test',delimiter=',',header=None,dtype=int)

    Xtrn = M.iloc[:, 1:-1].values
    ytrn = M.iloc[:, -1].values
    Xtst = T.iloc[:,:-1].values
    ytst = T.iloc[:,-1].values

    training_error = {}
    testing_error = {}
"""
