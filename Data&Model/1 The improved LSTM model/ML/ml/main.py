# -*- coding: utf-8 -*-

import config as C
import model as M
import utils as U
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    """
    Main execution function.
    Loads dataset, trains selected models, 
    and records training/testing performance.
    """

    # Load training and testing data
    x, y, x_test, y_test = U.main()

    print(f"Number of training samples: {len(x)}")
    print(f"Number of training labels: {len(y)}")
    print(f"Type of training data x: {type(x)}")

    model_list = C.model_list

    # Open result files for recording model performance
    with open("TRAIN_result.txt", "w", encoding="utf-8") as f_train, \
         open("TEST_result.txt", "w", encoding="utf-8") as f_test:

        for model_name in model_list:

            print(f"\nTraining model: {model_name}")

            # Train model and obtain predictions
            y_pred_train, y_pred_test = M.train(
                x, y, x_test, f_train, f_test, model_name
            )

            # Record training performance
            U.get_train_performance(
                y_pred_train, y, model_name, f_train
            )

            # Record testing performance
            total1, right1, hx, precision_1, recall_1, precision_0, recall_0 = \
                U.get_test_performance(
                    y_pred_test, y_test, model_name, f_test
                )

            print(
                hx,
                "Class 1 Precision:", precision_1,
                "Class 0 Precision:", precision_0,
                f"Model: {model_name}"
            )


if __name__ == "__main__":
    main()