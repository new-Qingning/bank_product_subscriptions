# -*- coding: utf-8 -*-
# Author: lkm
# date: 2023/7/29 23:35

import time
import graphviz
import warnings
import pandas as pd
from sklearn.tree import export_graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')


def transform(feature, target,
              criterion='gini',
              n_bins: int = 2,
              feature_name: str = None,
              class_names: list = None,
              fontname: str = 'KaiTi',
              filled: bool = True,
              rounded: bool = True,
              special_characters: bool = True,
              out_file: str = None,
              format: str = 'png',
              is_displayed: bool = True,
              view: bool = True):
    # 构建决策树
    model = DecisionTreeClassifier(criterion=criterion, max_leaf_nodes=n_bins)
    model.fit(feature, target)

    if feature_name is None:
        feature_name = 'Feature'

    if class_names is None:  # 判断是否自定义目标名称，默认为['0', '1']
        dot_tree = export_graphviz(model, out_file=out_file,
                                   feature_names=[feature_name], class_names=['0', '1'],
                                   fontname=fontname, filled=filled, rounded=rounded,
                                   special_characters=special_characters)
    else:
        dot_tree = export_graphviz(model, out_file=out_file,
                                   feature_names=[feature_name], class_names=class_names,
                                   fontname=fontname, filled=filled, rounded=rounded,
                                   special_characters=special_characters)
    # 使用graphviz加载
    graph = graphviz.Source(dot_tree, format=format)

    # 显示决策树图
    if is_displayed:
        graph.render('decision_tree_{}'.format(int(time.time())), directory='.', view=view, cleanup=True)

    # 输出分箱结果
    leaf_nodes = model.apply(feature)
    mapped_list = LabelEncoder().fit_transform(leaf_nodes)

    # 获取决策树的结构信息
    tree_structure = model.tree_
    # 获取叶节点的id
    leaf_ids = [node_id for node_id in range(tree_structure.node_count) if
                tree_structure.children_left[node_id] == tree_structure.children_right[node_id]]
    thresholds = [tree_structure.threshold[i - 1] for i in leaf_ids[:-1]]
    thresholds.append(thresholds[-1])
    # 输出最终分箱的节点
    for i, leaf_id in enumerate(leaf_ids):
        if leaf_id != leaf_ids[-1]:
            print('叶子节点:', leaf_id, '分箱结果值:', i, '分箱节点：<=', thresholds[i])
        else:
            print('叶子节点:', leaf_id, '分箱结果值:', i, '分箱节点：>', thresholds[i])

    return mapped_list


if __name__ == '__main__':
    df1 = pd.DataFrame({'age': [25, 35, 42, 48, 55, 60, 65, 70, 75, 80],
                        'label': [0, 1, 0, 1, 1, 0, 0, 1, 0, 1]})
    a = transform(df1[['age']], df1['label'], n_bins=3, feature_name='Age', is_displayed=False, format='png')
    df1['age_bin'] = a
    print(df1)