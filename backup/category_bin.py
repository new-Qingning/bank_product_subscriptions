# -*- coding: utf-8 -*-
# Author: lkm
# date: 2023/7/31 0:01

import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')


def cate_transform(feature, target, criterion='gini', n_bins=3):
    encoder = LabelEncoder()
    feature = encoder.fit_transform(feature).reshape(-1, 1)
    encoding_mapping = dict(zip(range(len(encoder.classes_)), encoder.classes_))

    # 构建决策树
    model = DecisionTreeClassifier(criterion=criterion, max_leaf_nodes=n_bins)
    model.fit(feature, target)

    # 获取决策树的结构信息
    tree_structure = model.tree_

    # 获取叶节点的id
    leaf_ids = [node_id for node_id in range(tree_structure.node_count) if
                tree_structure.children_left[node_id] == tree_structure.children_right[node_id]]
    thresholds = [tree_structure.threshold[i - 1] for i in leaf_ids[:-1]]
    thresholds.append(thresholds[-1])

    # 输出最终分箱的节点
    res = []
    for i, leaf_id in enumerate(leaf_ids):
        if leaf_id != leaf_ids[-1]:
            res.append(encoding_mapping[i])
            print('分箱值', i, '叶子节点:', leaf_id, '分箱节点:', encoding_mapping[i])
        else:
            print('分箱值', i, '叶子节点:', leaf_id, '分箱节点:',
                  ', '.join([str(i) for i in encoding_mapping.values() if i not in res]))

    output = LabelEncoder().fit_transform(model.apply(feature))
    print(output)


if __name__ == '__main__':
    import seaborn as sns

    data = sns.load_dataset('titanic')
    x = data['embarked']
    y = data['survived']
    cate_transform(x, y, criterion='gini', n_bins=4)
