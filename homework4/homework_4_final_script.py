import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


def check_response_type(response):

    if len(set(response)) <= 3:  # Assuming there could be 3 classes (iris)
        var = "Boolean"
        print("Response variable is boolean!")
    else:
        var = "Continuous"
        print(len(set(response)))
        print("Response variable is continuous!")
    return var


def check_predictor_type(predictor):
    # all object types are categorical, int/float types with less than 10 unique values are categorical as well (size)
    if (predictor.dtype == "object") or (
        predictor.dtype != "object" and len(set(predictor)) < 10
    ):
        return "Categorical"
    else:
        return "Continuous"


def r_continuous_p_continuous(r, p, response, predictor):

    print("Response:", r, "Predictor:", p)
    fig = px.scatter(x=response, y=predictor, trendline="ols")
    fig.update_layout(
        title="Continuous Response by Continuous Predictor",
        xaxis_title=p,
        yaxis_title=r,
    )
    fig.show()


def Linear_Regression(response, predictor):
    linear_regression_model = statsmodels.api.OLS(response, predictor)

    linear_regression_model_fitted = linear_regression_model.fit()

    print(linear_regression_model_fitted.summary())

    # Get the stats
    t_value = round(linear_regression_model_fitted.tvalues[0], 6)
    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[0])
    print("t_value is", t_value)
    print("p-value is", p_value)


# Mean of response for categorical
def mean_cat_diff(df, predictor, response):
    w = df.groupby(predictor).count()[response] / sum(
        df.groupby(predictor).count()[response]
    )
    w = w.reset_index()

    _mean = np.mean(df[response])
    bin = df.groupby(predictor).apply(np.mean)
    response_mean = bin[response].reset_index(name="Mean: response")
    unweigh_diff = np.sum(np.square(response_mean.iloc[:, 1] - _mean)) / len(
        response_mean
    )
    weigh_diff = np.sum(
        w.iloc[:, 1] * np.square(response_mean.iloc[:, 1] - _mean)
    ) / len(
        response_mean
    )  # weighted difference with mean
    print(f"unweighted difference with mean : {unweigh_diff}")
    print(f"unweighted difference with mean : {weigh_diff}")


# Mean of response for continuous
def mean_cont_diff(predictor, response):
    bin = pd.cut(predictor, 10)
    w = bin.value_counts() / sum(bin.value_counts())
    w = w.reset_index()
    _mean = np.mean(response)
    response_mean = (
        response.groupby(bin).apply(np.mean).reset_index(name="Mean-response")
    )
    diff = np.square(response_mean.iloc[:, 1] - _mean)
    unweigh_diff = np.sum(np.square(response_mean.iloc[:, 1] - _mean)) / len(
        response_mean
    )
    weigh_diff = np.sum(
        w.iloc[:, 1] * np.square(response_mean.iloc[:, 1] - _mean)
    ) / len(
        response_mean
    )  # weighted difference with mean
    print(f"unweighted difference with mean: {unweigh_diff}")
    print(f"unweighted difference with mean: {weigh_diff}")


# random forest feature importance
def random_forest(df, p, predictor, response):
    rf = RandomForestClassifier()
    rf.fit(predictor, response)
    print(p, rf.feature_importances_)


def logistic_regression(p, predictor, response):
    logisticRegr = LogisticRegression(solver="lbfgs")
    logisticreg = logisticRegr.fit(predictor, response)
    plt.plot(p, response, "-o")

    # Uncomment to see logistic Regression plots
    # plt.show()


def main():

    # df = px.data.iris()
    cancer = load_breast_cancer()
    df = pd.DataFrame(
        np.c_[cancer["data"], cancer["target"]],
        columns=np.append(cancer["feature_names"], ["target"]),
    )

    X = df.loc[:, df.columns != "target"]
    y = df["target"]
    response = "target"
    # df = px.data.tips()

    # response = "species"
    # predictors = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    # response = "tip"
    # predictors = ["total_bill", "sex", "smoker", "day", "time", "size"]

    # y = df[response]
    # X = df[predictors]

    # Dataframes for cat and cont predictors
    cat_predictors = []
    cont_predictors = []
    for i in X:
        if check_predictor_type(X[i]) == "Categorical":
            cat_predictors.append(i)
        elif check_predictor_type(X[i]) == "Continuous":
            cont_predictors.append(i)

    response_type = check_response_type(y)

    # encoding categorical predictors
    encoder = LabelEncoder()
    df_cat_predictors = df[cat_predictors].apply(encoder.fit_transform)
    df_cont_predictors = df[cont_predictors]
    # print(df_cont_predictors)

    # print(df_cont_predictors.shape,y.shape)

    # iterating over predictors
    for i in X:

        predictor_type = check_predictor_type(X[i])
        # Following code contains 4 different if conditions for 4 different combinations of (Categorical, Continuous)
        if predictor_type == "Categorical" and response_type == "Boolean":
            print("Categorical-", i)

            # calling functions
            mean_cat_diff(df, X[i], response)
            logistic_regression(df_cont_predictors[i], df_cont_predictors, y)
            conf_matrix = confusion_matrix(y, X[i])

            fig_no_relationship = go.Figure(
                data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
            )
            fig_no_relationship.update_layout(
                title="Categorical Predictor by Categorical Response (without relationship)",
                xaxis_title="Response",
                yaxis_title="Predictor",
            )
            fig_no_relationship.show()

        elif predictor_type == "Categorical" and response_type == "Continuous":
            print("Categorical-", i)
            mean_cat_diff(df, X[i], response)

            Linear_Regression(y, df_cat_predictors[i])

            fig_1 = ff.create_distplot([df_cat_predictors[i]], [i], bin_size=0.2)

            fig_1.show()

            fig_2 = go.Figure()
            for curr_hist, curr_group in zip([df_cat_predictors[i]], [i]):
                fig_2.add_trace(
                    go.Violin(
                        x=np.repeat(curr_group, len(df)),
                        y=curr_hist,
                        name=curr_group,
                        box_visible=True,
                        meanline_visible=True,
                    )
                )
            fig_2.show()

        elif predictor_type == "Continuous" and response_type == "Continuous":
            print("Continuous -", i)
            mean_cont_diff(X[i], y)
            r_continuous_p_continuous(response, i, y, X[i])
            Linear_Regression(y, X[i])
            random_forest(i, df_cont_predictors, y.astype("int"))

        elif predictor_type == "Continuous" and response_type == "Boolean":
            logistic_regression(df_cont_predictors[i], df_cont_predictors, y)

            print("Continuous -", i)
            mean_cont_diff(X[i], y)

            random_forest(i, df_cont_predictors, y.astype("int"))
            fig_1 = ff.create_distplot([df_cont_predictors[i]], [i], bin_size=0.2)

            fig_1.show()

            fig_2 = go.Figure()
            for curr_hist, curr_group in zip([df_cont_predictors[i]], [i]):
                fig_2.add_trace(
                    go.Violin(
                        x=np.repeat(curr_group, len(df)),
                        y=curr_hist,
                        name=curr_group,
                        box_visible=True,
                        meanline_visible=True,
                    )
                )
            fig_2.show()


if __name__ == "__main__":
    sys.exit(main())
