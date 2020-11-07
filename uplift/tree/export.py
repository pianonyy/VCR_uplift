"""
This module defines export functions for decision trees.
"""



import numpy as np

from . import _criterion
from . import _tree

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_consistent_length

def error(group_type, X, Y):
    Y=np.array(Y)
    
    X['uplift_score'] = (X['uplift_score'] > 0).astype(int)
    
    if group_type == 'treat':
        X['error'] = (X['uplift_score'] != Y).astype(int)
    if group_type == 'control':
        X['error'] = (X['uplift_score'] == Y).astype(int)

    return X['error']
""" _____________________________________ADABOOST IMPLEMENTATION____________________________________________"""

def uplift_adaboost(n_estimators, estimator, X_train, Y_train, treatment_train):
    
    train_size = X_train.shape[0]
    

    check_consistent_length(X_train, Y_train, treatment_train)
    

    treat_records_x = X_train[treatment_train == 1] 
    control_records_x = X_train[treatment_train == 0]
    treat_records_y = Y_train[treatment_train == 1] 
    control_records_y = Y_train[treatment_train == 0]
    
    # 1 intialize weights
    treat_weights = np.ones(shape = treat_records_x.shape[0])
    control_weights = np.ones(shape = control_records_x.shape[0])

    # 2 for
    weight_array_list_treatment =[]
    weight_array_list_control = []
    
    total_betas = []

    #weight_array_list_control.append(control_weights)
    #weight_array_list_treatment.append(treatment_weights)
    
    for i in range(0,n_estimators):
        
        treat_weights = treat_weights / (treat_weights.sum() + control_weights.sum())
        control_weights = control_weights / (treat_weights.sum() + control_weights.sum())
        #add weights to X_train
        
        treat_records_x['weights'] = treat_weights
        control_records_x['weights'] = control_weights
        
        #print(treat_records_x)
        X_train = pd.concat([treat_records_x, control_records_x])
        Y_train = np.concatenate((treat_records_y, control_records_y), axis=None)
        
        estimator.fit(X_train, treatment_train, Y_train)
        
        uplift_score_t = estimator.predict_uplift(treat_records_x)
        uplift_score_c = estimator.predict_uplift(control_records_x)

        treat_records_x['uplift_score'] = uplift_score_t
        control_records_x['uplift_score'] = uplift_score_c
        
        error_T = error(group_type = 'treat',X = treat_records_x, Y = treat_records_y)
        error_C = error(group_type ='control', X = control_records_x,Y =control_records_x)
        

        #compute treat and control errors (5)
        treat_error = treat_records_x.loc[error_T == 1,'weights'].sum() / treat_records_x['weights'].sum()
        control_error = control_records_x.loc[error_C == 1,'weights'].sum() / control_records_x['weights'].sum()
        
        #compute relative sizes (1)
        p_t = treat_records_x['weights'].sum() / (treat_records_x['weights'].sum() + control_records_x['weights'].sum())
        p_c = control_records_x['weights'].sum() / (treat_records_x['weights'].sum() + control_records_x['weights'].sum())
        
        #total error (5)
        total_error = p_t * treat_error + p_c * control_error
        
        #compute betas (d)
        beta = total_error / (1 - total_error)
        if (beta == 1) or (treat_error <=0 or treat_error >= 0.5) or (treat_error <= 0 or treat_error >= 0.5):
            treat_weights = np.random.normal(loc=0.5, scale=0.5,size = treat_records_x.shape[0])
            control_weights = np.random.normal(loc=0.5, scale=0.5,size = control_records_x.shape[0])
            continue
            
        #update weights (f) (g)
        
        treat_weights = treat_weights * (beta ** ((treat_records_x['uplift_score'] == treat_records_y).astype(int)))
        control_weights = control_weights * (beta ** ((control_records_x['uplift_score'] == (1 - control_records_y)).astype(int)))
        
        total_betas.append(beta)
    print(total_betas)

def feature_importance(df_features_imp, features, num_bins,regularization, penalty):
    # Analyse given features in df_dataset and makes descending list of them according to their importance score.
    
    # Parameters
    # ----------
    # df_dataset : pandas dataframe, using following style:
    # (features(size: n_d),target(size: 1_d),treatmet(size: 1_d))

    # features : list, names of using features
    # num_bins : int, for continious features, making steps like (0,10)...(90,100) if num_bins = 10
    # and we have feature, measuring in (0,100)
    # Returns:
    # dict of scoref features and draw the plot

    dict_feat ={}

    indices_train, indices_test = train_test_split(
         df_features_imp.index,
         test_size=0.3,
         random_state = 12
    )
    

    for name_feature in features:
        df_features_imp = df_features_imp.sort_values(name_feature)
        df_features_imp = df_features_imp.reset_index(drop=True)
        
        indices_train_bins = np.array_split(indices_train, num_bins)
        indices_test_bins = np.array_split(indices_test, num_bins)
        NIV = 0
        penalty = 0
        for i in range(0, num_bins):
            group_train = df_features_imp.iloc[indices_train_bins[i]]
            group_test = df_features_imp.iloc[indices_test_bins[i]]

            number_of_treat = (group_train.loc[ (group_train['treatment'] == 1) ]).shape[0]
            number_of_control = (group_train.loc[ (group_train['treatment'] == 0) ]).shape[0]

            number_of_succ_treat = (group_train.loc[(group_train['visit'] == 1) & (group_train['treatment'] == 1)]).shape[0]
            number_of_succ_control = (group_train.loc[(group_train['visit'] == 1) & (group_train['treatment'] == 0)]).shape[0]

            number_of_fail_control = (group_train.loc[(group_train['visit'] == 0) & (group_train['treatment'] == 0)]).shape[0]
            number_of_fail_treat = (group_train.loc[(group_train['visit'] == 0) & (group_train['treatment'] == 1)]).shape[0]
            #print("treat",number_of_treat,"control",number_of_control,"succ_treat", number_of_succ_treat, "succ_control", number_of_succ_control,"fail_treat", number_of_fail_treat,"fail_control",number_of_fail_control)

            succ_rate_treat = number_of_succ_treat / number_of_treat
            fail_rate_treat = number_of_fail_treat / number_of_treat

            succ_rate_control = number_of_succ_control / number_of_control 
            fail_rate_control = number_of_fail_control / number_of_control 
            
            up = (succ_rate_treat + regularization) / (fail_rate_treat + (2 * regularization))
            down = (succ_rate_control + regularization) / (fail_rate_control + (2 * regularization))
            
            NWOE_train = np.log(up / down)
            
            NIV += (succ_rate_treat * fail_rate_control - fail_rate_treat * succ_rate_control) * NWOE_train
            
            number_of_treat = (group_test.loc[ (group_test['treatment'] == 1) ]).shape[0]
            number_of_control = (group_test.loc[ (group_test['treatment'] == 0) ]).shape[0]

            number_of_succ_treat = (group_test.loc[(group_test['visit'] == 1) & (group_test['treatment'] == 1)]).shape[0]
            number_of_succ_control = (group_test.loc[(group_test['visit'] == 1) & (group_test['treatment'] == 0)]).shape[0]

            number_of_fail_control = (group_test.loc[(group_test['visit'] == 0) & (group_test['treatment'] == 0)]).shape[0]
            number_of_fail_treat = (group_test.loc[(group_test['visit'] == 0) & (group_test['treatment'] == 1)]).shape[0]
            #print("treat",number_of_treat,"control",number_of_control,"succ_treat", number_of_succ_treat, "succ_control", number_of_succ_control,"fail_treat", number_of_fail_treat,"fail_control",number_of_fail_control)

            succ_rate_treat = number_of_succ_treat / number_of_treat
            fail_rate_treat = number_of_fail_treat / number_of_treat

            succ_rate_control = number_of_succ_control / number_of_control 
            fail_rate_control = number_of_fail_control / number_of_control 
            
            up = (succ_rate_treat + regularization) / (fail_rate_treat + (2 * regularization))
            down = (succ_rate_control + regularization) / (fail_rate_control + (2 * regularization))
            
            NWOE_test = np.log(up / down)
            
            w = abs(NWOE_train-NWOE_test)
            penalty += (succ_rate_treat * fail_rate_control - fail_rate_treat * succ_rate_control) * w
            
        if penalty:
            dict_feat[name_feature] = NIV - penalty
        else:
            dict_feat[name_feature] = NIV

    height = []
    features =[]
    list_d = list(dict_feat.items())
    list_d.sort(key=lambda i: i[1],reverse=True)
    for i in list_d:
        height.append(i[1])
        features.append(i[0])
    
    y_pos = np.arange(len(features))
 
    plt.barh(y_pos, height)
    plt.yticks(y_pos, features)
    plt.show()

    return dict_feat


def _color_brew(n):
    """Generate n colors with equally spaced hues.

    Parameters
    ----------
    n : int
        The number of colors required.

    Returns
    -------
    color_list : list, length n
        List of n tuples of form (R, G, B) being the components of each color.
    """
    color_list = []

    # Initialize saturation & value; calculate chroma & value shift
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    for h in np.arange(25, 385, 360. / n).astype(int):
        # Calculate some intermediate values
        h_bar = h / 60.
        x = c * (1 - abs((h_bar % 2) - 1))
        # Initialize RGB with same hue & chroma as our color
        rgb = [(c, x, 0),
               (x, c, 0),
               (0, c, x),
               (0, x, c),
               (x, 0, c),
               (c, 0, x),
               (c, x, 0)]
        r, g, b = rgb[int(h_bar)]
        # Shift the initial RGB values to match value and store
        rgb = [(int(255 * (r + m))),
               (int(255 * (g + m))),
               (int(255 * (b + m)))]
        color_list.append(rgb)

    return color_list


def export_graphviz(decision_tree, out_file="tree.dot", max_depth=None,
                    feature_names=None, class_names=None, label='all',
                    filled=False, leaves_parallel=False, impurity=True,
                    node_ids=False, proportion=False, rotate=False,
                    rounded=False, special_characters=False):
    """Export a decision tree in DOT format.

    This function generates a GraphViz representation of the decision tree,
    which is then written into `out_file`. Once exported, graphical renderings
    can be generated using, for example::

        $ dot -Tps tree.dot -o tree.ps      (PostScript format)
        $ dot -Tpng tree.dot -o tree.png    (PNG format)

    The sample counts that are shown are weighted with any sample_weights that
    might be present.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    decision_tree : decision tree classifier
        The decision tree to be exported to GraphViz.

    out_file : file object or string, optional (default="tree.dot")
        Handle or name of the output file.

    max_depth : int, optional (default=None)
        The maximum depth of the representation. If None, the tree is fully
        generated.

    feature_names : list of strings, optional (default=None)
        Names of each of the features.

    class_names : list of strings, bool or None, optional (default=None)
        Names of each of the target classes in ascending numerical order.
        Only relevant for classification and not supported for multi-output.
        If ``True``, shows a symbolic representation of the class name.

    label : {'all', 'root', 'none'}, optional (default='all')
        Whether to show informative labels for impurity, etc.
        Options include 'all' to show at every node, 'root' to show only at
        the top root node, or 'none' to not show at any node.

    filled : bool, optional (default=False)
        When set to ``True``, paint nodes to indicate majority class for
        classification, extremity of values for regression, or purity of node
        for multi-output.

    leaves_parallel : bool, optional (default=False)
        When set to ``True``, draw all leaf nodes at the bottom of the tree.

    impurity : bool, optional (default=True)
        When set to ``True``, show the impurity at each node.

    node_ids : bool, optional (default=False)
        When set to ``True``, show the ID number on each node.

    proportion : bool, optional (default=False)
        When set to ``True``, change the display of 'values' and/or 'samples'
        to be proportions and percentages respectively.

    rotate : bool, optional (default=False)
        When set to ``True``, orient tree left to right rather than top-down.

    rounded : bool, optional (default=False)
        When set to ``True``, draw node boxes with rounded corners and use
        Helvetica fonts instead of Times-Roman.

    special_characters : bool, optional (default=False)
        When set to ``False``, ignore special characters for PostScript
        compatibility.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn import tree

    >>> clf = tree.DecisionTreeClassifier()
    >>> iris = load_iris()

    >>> clf = clf.fit(iris.data, iris.target)
    >>> tree.export_graphviz(clf,
    ...     out_file='tree.dot')                # doctest: +SKIP
    """

    def get_color(value):
        # Find the appropriate color & intensity for a node
        if colors['bounds'] is None:
            # Classification tree
            color = list(colors['rgb'][np.argmax(value)])
            sorted_values = sorted(value, reverse=True)
            if len(sorted_values) == 1:
                alpha = 0
            else:
                alpha = int(np.round(255 * (sorted_values[0] - sorted_values[1]) /
                                           (1 - sorted_values[1]), 0))
        else:
            # Regression tree or multi-output
            color = list(colors['rgb'][0])
            alpha = int(np.round(255 * ((value - colors['bounds'][0]) /
                                        (colors['bounds'][1] -
                                         colors['bounds'][0])), 0))

        # Return html color code in #RRGGBBAA format
        color.append(alpha)
        hex_codes = [str(i) for i in range(10)]
        hex_codes.extend(['a', 'b', 'c', 'd', 'e', 'f'])
        color = [hex_codes[c // 16] + hex_codes[c % 16] for c in color]

        return '#' + ''.join(color)

    def node_to_str(tree, node_id, criterion):
        # Generate the node content string
        if tree.n_outputs == 1:
            value = tree.value[node_id][0, :]
        else:
            value = tree.value[node_id]

        # Should labels be shown?
        labels = (label == 'root' and node_id == 0) or label == 'all'

        # PostScript compatibility for special characters
        if special_characters:
            characters = ['&#35;', '<SUB>', '</SUB>', '&le;', '<br/>', '>']
            node_string = '<'
        else:
            characters = ['#', '[', ']', '<=', '\\n', '"']
            node_string = '"'

        # Write node ID
        if node_ids:
            if labels:
                node_string += 'node '
            node_string += characters[0] + str(node_id) + characters[4]

        # Write decision criteria
        if tree.children_left[node_id] != _tree.TREE_LEAF:
            # Always write node decision criteria, except for leaves
            if feature_names is not None:
                feature = feature_names[tree.feature[node_id]]
            else:
                feature = "X%s%s%s" % (characters[1],
                                       tree.feature[node_id],
                                       characters[2])
            node_string += '%s %s %s%s' % (feature,
                                           characters[3],
                                           round(tree.threshold[node_id], 4),
                                           characters[4])

        # Write impurity
        if impurity:
            if isinstance(criterion, _criterion.FriedmanMSE):
                criterion = "friedman_mse"
            elif not isinstance(criterion, str):
                criterion = "impurity"
            if labels:
                node_string += '%s = ' % criterion
            node_string += (str(round(tree.impurity[node_id], 4)) +
                            characters[4])

        # Write node sample count
        if labels:
            node_string += 'samples = '
        if proportion:
            percent = (100. * tree.n_node_samples[node_id] /
                       float(tree.n_node_samples[0]))
            node_string += (str(round(percent, 1)) + '%' +
                            characters[4])
        else:
            node_string += (str(tree.n_node_samples[node_id]) +
                            characters[4])

        # Write node class distribution / regression value
        if proportion and tree.n_classes[0] != 1:
            # For classification this will show the proportion of samples
            value = value / tree.weighted_n_node_samples[node_id]
        if labels:
            node_string += 'value = '
        if tree.n_classes[0] == 1:
            # Regression
            value_text = np.around(value, 4)
        elif proportion:
            # Classification
            value_text = np.around(value, 2)
        elif np.all(np.equal(np.mod(value, 1), 0)):
            # Classification without floating-point weights
            value_text = value.astype(int)
        else:
            # Classification with floating-point weights
            value_text = np.around(value, 4)
        # Strip whitespace
        value_text = str(value_text.astype('S32')).replace("b'", "'")
        value_text = value_text.replace("' '", ", ").replace("'", "")
        if tree.n_classes[0] == 1 and tree.n_outputs == 1:
            value_text = value_text.replace("[", "").replace("]", "")
        value_text = value_text.replace("\n ", characters[4])
        node_string += value_text + characters[4]

        # Write node majority class
        if (class_names is not None and
                tree.n_classes[0] != 1 and
                tree.n_outputs == 1):
            # Only done for single-output classification trees
            if labels:
                node_string += 'class = '
            if class_names is not True:
                class_name = class_names[np.argmax(value)]
            else:
                class_name = "y%s%s%s" % (characters[1],
                                          np.argmax(value),
                                          characters[2])
            node_string += class_name

        # Clean up any trailing newlines
        if node_string[-2:] == '\\n':
            node_string = node_string[:-2]
        if node_string[-5:] == '<br/>':
            node_string = node_string[:-5]

        return node_string + characters[5]

    def recurse(tree, node_id, criterion, parent=None, depth=0):
        if node_id == _tree.TREE_LEAF:
            raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        # Add node with description
        if max_depth is None or depth <= max_depth:

            # Collect ranks for 'leaf' option in plot_options
            if left_child == _tree.TREE_LEAF:
                ranks['leaves'].append(str(node_id))
            elif str(depth) not in ranks:
                ranks[str(depth)] = [str(node_id)]
            else:
                ranks[str(depth)].append(str(node_id))

            out_file.write('%d [label=%s'
                           % (node_id,
                              node_to_str(tree, node_id, criterion)))

            if filled:
                # Fetch appropriate color for node
                if 'rgb' not in colors:
                    # Initialize colors and bounds if required
                    colors['rgb'] = _color_brew(tree.n_classes[0])
                    if tree.n_outputs != 1:
                        # Find max and min impurities for multi-output
                        colors['bounds'] = (np.min(-tree.impurity),
                                            np.max(-tree.impurity))
                    elif tree.n_classes[0] == 1 and len(np.unique(tree.value)) != 1:
                        # Find max and min values in leaf nodes for regression
                        colors['bounds'] = (np.min(tree.value),
                                            np.max(tree.value))
                if tree.n_outputs == 1:
                    node_val = (tree.value[node_id][0, :] /
                                tree.weighted_n_node_samples[node_id])
                    if tree.n_classes[0] == 1:
                        # Regression
                        node_val = tree.value[node_id][0, :]
                else:
                    # If multi-output color node by impurity
                    node_val = -tree.impurity[node_id]
                out_file.write(', fillcolor="%s"' % get_color(node_val))
            out_file.write('] ;\n')

            if parent is not None:
                # Add edge to parent
                out_file.write('%d -> %d' % (parent, node_id))
                if parent == 0:
                    # Draw True/False labels if parent is root node
                    angles = np.array([45, -45]) * ((rotate - .5) * -2)
                    out_file.write(' [labeldistance=2.5, labelangle=')
                    if node_id == 1:
                        out_file.write('%d, headlabel="True"]' % angles[0])
                    else:
                        out_file.write('%d, headlabel="False"]' % angles[1])
                out_file.write(' ;\n')

            if left_child != _tree.TREE_LEAF:
                recurse(tree, left_child, criterion=criterion, parent=node_id,
                        depth=depth + 1)
                recurse(tree, right_child, criterion=criterion, parent=node_id,
                        depth=depth + 1)

        else:
            ranks['leaves'].append(str(node_id))

            out_file.write('%d [label="(...)"' % node_id)
            if filled:
                # color cropped nodes grey
                out_file.write(', fillcolor="#C0C0C0"')
            out_file.write('] ;\n' % node_id)

            if parent is not None:
                # Add edge to parent
                out_file.write('%d -> %d ;\n' % (parent, node_id))

    own_file = False
    try:
        if isinstance(out_file, str):
            out_file = open(out_file, "w", encoding="utf-8")
            own_file = True

        # The depth of each node for plotting with 'leaf' option
        ranks = {'leaves': []}
        # The colors to render each node with
        colors = {'bounds': None}

        out_file.write('digraph Tree {\n')
        
        # Specify node aesthetics
        out_file.write('node [shape=box')
        rounded_filled = []
        if filled:
            rounded_filled.append('filled')
        if rounded:
            rounded_filled.append('rounded')
        if len(rounded_filled) > 0:
            out_file.write(', style="%s", color="black"'
                           % ", ".join(rounded_filled))
        if rounded:
            out_file.write(', fontname=helvetica')
        out_file.write('] ;\n')

        # Specify graph & edge aesthetics
        if leaves_parallel:
            out_file.write('graph [ranksep=equally, splines=polyline] ;\n')
        if rounded:
            out_file.write('edge [fontname=helvetica] ;\n')
        if rotate:
            out_file.write('rankdir=LR ;\n')

        # Now recurse the tree and add node & edge attributes
        if isinstance(decision_tree, _tree.Tree):
            recurse(decision_tree, 0, criterion="impurity")
            
        else:
            recurse(decision_tree.tree_, 0, criterion=decision_tree.criterion)
            

        # If required, draw leaf nodes at same depth as each other
        if leaves_parallel:
            for rank in sorted(ranks):
                out_file.write("{rank=same ; " +
                               "; ".join(r for r in ranks[rank]) + "} ;\n")
        out_file.write("}")

    finally:
        if own_file:
            out_file.close()
