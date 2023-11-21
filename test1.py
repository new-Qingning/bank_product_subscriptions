# -*- coding: utf-8 -*-
# Author: 梁开孟
# date: 2023/11/20 21:31

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


class Initialize:
    DEFAULT_LOG_FILE = "logging/best_score_log.json"
    DEFAULT_ERROR_FILE = "logging/error_log.log"
    DEFAULT_TRAIN_LOG = "logging/train_log.log"

    def __init__(self, log_file=None, error_file=None, train_log=None):
        """
        初始化日志文件，错误文件和训练日志文件.
        """
        if log_file is None:
            log_file = self.DEFAULT_LOG_FILE
        if not isinstance(log_file, str):
            raise ValueError("log_file must be a string")

        if error_file is None:
            error_file = self.DEFAULT_ERROR_FILE
        if not isinstance(error_file, str):
            raise ValueError("error_file must be a string")

        if train_log is None:
            train_log = self.DEFAULT_TRAIN_LOG
        if not isinstance(train_log, str):
            raise ValueError("train_log must be a string")

        self.log_file = log_file
        self.error_file = error_file
        self.train_log = train_log


class AutoLearningClassifier(Initialize):
    def __init__(self,
                 dataset: DataFrame,
                 estimator: str,
                 feature_pipeline: str,
                 feature_attribute: str,
                 log_file: str = None,
                 error_file: str = None,
                 train_log: str = None):
        """
        初始化AutoLearningClassifier类.

        参数:
            dataset: DataFrame类型, 数据集.
            estimator: str类型, 估计器.
            feature_pipeline: str类型, 特征管道.
            feature_attribute: str类型, 特性属性.
            log_file: str类型, 日志文件路径（默认为"logging/best_score_log.json"）.
            error_file: str类型, 错误文件路径（默认为"logging/error_log.log"）.
            train_log: str类型, 训练日志文件路径（默认为"logging/train_log.log"）.
        """
        super().__init__(log_file, error_file, train_log)
        self.formatted_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        self.estimator = joblib.load(estimator)
        self._load_feature_attributes(feature_attribute)
        self._feature_engineer(feature_pipeline)
        self.features = dataset[self.feature_attributes["ALL_FEATURES"][:-1]]
        self.target = dataset[self.feature_attributes["ALL_FEATURES"][-1]]

    def _load_estimator(self, estimator):
        """
        加载估计器模型.
        """
        self.estimator = joblib.load(estimator)

    def _load_feature_attributes(self, feature_attribute):
        """
        加载特性属性.
        """
        try:
            with open(feature_attribute, "r") as json_file:
                self.feature_attributes = json.load(json_file)
        except Exception as e:
            logging.basicConfig(filename=self.error_file, level=logging.ERROR)
            error_message = traceback.format_exc()
            logging.error(f"{self.formatted_date} 异常日志如下：\n{error_message}\n")
            raise e  # 重新抛出捕获到的异常以便外部处理

    def _feature_engineer(self, feature_pipeline):
        """
        加载数据处理与特征工程管道，转换特征值
        """
        try:
            preprocessing_pipeline = joblib.load(feature_pipeline)
            self.X_transform = preprocessing_pipeline.transform(self.features)
        except Exception:
            logging.basicConfig(filename=self.error_file, level=logging.ERROR)
            error_message = traceback.format_exc()
            logging.error(f"{self.formatted_date} 异常日志如下：\n{error_message}\n")

    def train_model(self):
        target = self.target.apply(lambda x: 1 if x == "yes" else 0)
        x_train, x_valid, y_train, y_valid = train_test_split(self.X_transform, target, test_size=0.3, random_state=12)
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
