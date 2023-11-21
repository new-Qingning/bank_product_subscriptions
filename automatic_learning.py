# -*- coding: utf-8 -*-
# Author: 梁开孟
# date: 2023/11/18 20:14

import os
import re
import time
import json
import joblib
import logging
import warnings
import traceback
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")


class AutoLearningClassifier:
    def __init__(self,
                 dataset: DataFrame,
                 estimator: str,
                 feature_pipeline: str,
                 feature_attribute: str,
                 log_file: str = None,
                 error_file: str = None,
                 train_log: str = None):
        self.feature_pipeline = feature_pipeline
        self.estimator = joblib.load(estimator)
        self.formatted_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

        if not log_file:
            self.log_file = "logging/best_score_log.json"
        else:
            self.log_file = log_file

        if not error_file:
            self.error_file = "logging/error_log.log"
        else:
            self.error_file = error_file

        if not train_log:
            self.train_log = "logging/train_log.log"
        else:
            self.train_log = train_log

        try:
            with open(feature_attribute, "r") as json_file:
                feature_attributes = json.load(json_file)
        except Exception:
            logging.basicConfig(filename=self.error_file, level=logging.ERROR)
            error_message = traceback.format_exc()
            logging.error(f"{self.formatted_date} 异常日志如下：\n{error_message}\n")

        self.features = dataset[feature_attributes["ALL_FEATURES"][:-1]]
        self.target = dataset[feature_attributes["ALL_FEATURES"][-1]]

    def feature_engineer(self):
        try:
            preprocessing_pipeline = joblib.load(self.feature_pipeline)
            return preprocessing_pipeline.transform(self.features)
        except Exception:
            logging.basicConfig(filename=self.error_file, level=logging.ERROR)
            error_message = traceback.format_exc()
            logging.error(f"{self.formatted_date} 异常日志如下：\n{error_message}\n")

    def train_model(self):
        X_transformed = self.feature_engineer()
        target = self.target.apply(lambda x: 1 if x == "yes" else 0)
        x_train, x_valid, y_train, y_valid = train_test_split(X_transformed, target, test_size=0.3, random_state=12)
        lgbm_model = LGBMClassifier(**self.estimator.get_params())
        lgbm_model.fit(x_train, y_train,
                       eval_set=[(x_train, y_train), (x_valid, y_valid)],
                       eval_metric=['binary_logloss', 'auc'],
                       early_stopping_rounds=10,
                       init_model=self.estimator)

        try:
            filename = os.path.join("models", "best_model_" + re.sub(r"-", "", self.formatted_date[:10]) + ".pkl")
            joblib.dump(lgbm_model, filename)
        except Exception:
            logging.basicConfig(filename=self.error_file, level=logging.ERROR)
            error_message = traceback.format_exc()
            logging.error(f"{self.formatted_date} 异常日志如下：\n{error_message}\n")

        # 获取模型最优分数
        best_score = lgbm_model.best_score_
        # 获取模型的最优迭代次数
        best_iteration = lgbm_model.best_iteration_

        score_dict = {
            self.formatted_date[:10]: {
                "TRAIN_AUC": best_score["training"]["auc"],
                "TRAIN_LOSS": best_score["training"]["binary_logloss"],
                "VALID_AUC": best_score["valid_1"]["auc"],
                "VALID_LOSS": best_score["valid_1"]["binary_logloss"]
            }
        }

        with open(self.log_file, "w") as json_file:
            json.dump(score_dict, json_file)

        log_text = f"{self.formatted_date} best_iteration={best_iteration} best_score={best_score}"
        logging.basicConfig(filename=self.train_log, level=logging.INFO)
        logging.info(log_text)

        return lgbm_model


if __name__ == "__main__":
    train = pd.read_csv("dataset/train.csv")
    test = pd.read_csv("dataset/test.csv")
    pipeline_file = "models/preprocessing_pipeline.pkl"
    feature_file = "feature_attribute.json"
    model_file = "models/best_model.pkl"
    auto_learning = AutoLearningClassifier(dataset=train, estimator=model_file,
                                           feature_pipeline=pipeline_file,
                                           feature_attribute=feature_file)
    clf = auto_learning.train_model()
    print(clf.best_iteration_, clf.best_score_)
