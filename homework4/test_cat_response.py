import io
import sys

import numpy as np
import pandas as pd
import requests
import statsmodels.api
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from sklearn import datasets
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder


def main():

    df = px.data.iris()
    response = "species"
    predictors = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    y = df[response]
    X = df[predictors]
    if len(set(y)) == 3:
        var = "Boolean"
        print("Response variable is boolean!")
    else:
        var = "Continuous"
        print(len(set(y)))
        print("Response variable is continuous!")

    labelEncoder = LabelEncoder()
    # x = labelEncoder.fit_transform(X)
    for i in X:
        print(i)
        if X[i].dtype == "object":
            print("Categorical variable")

        elif X[i].dtype != "object" and len(set(X[i])) < 10:

            print("Here")
            if var == "Continuous":

                fig_1 = ff.create_distplot([X[i]], [i], bin_size=0.2)
                fig_1.show()

                fig_2 = go.Figure()
                for curr_hist, curr_group in zip([X[i]], [i]):
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
                print("Categorical variable!")
        else:
            if var == "Continuous":
                linear_regression_model = statsmodels.api.OLS(y, X[i])

                linear_regression_model_fitted = linear_regression_model.fit()

                print(linear_regression_model_fitted.summary())

                # Get the stats
                t_value = round(linear_regression_model_fitted.tvalues[0], 6)
                p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[0])
                print(t_value)
                print(p_value)
                fig = px.scatter(x=y, y=X[i], trendline="ols")
                fig.update_layout(
                    title=f"Variable: {i}: (t-value={t_value}) (p-value={p_value})",
                    xaxis_title=f"Variable: {i}",
                    yaxis_title="y",
                )
                fig.show()
                print("Continuous variable")
            elif var == "Boolean":
                print("yoo")
                Logistic_regression_model = statsmodels.api.Logit(y, X[i])
                Logistic_regression_model_fitted = Logistic_regression_model.fit()

                # Get the stats
                t_value = round(Logistic_regression_model_fitted.tvalues[0], 6)
                p_value = "{:.6e}".format(Logistic_regression_model_fitted.pvalues[0])

                fig = px.histogram(y, X[i])
                fig.update_layout(
                    title=f"Variable: {i}: (t-value={t_value}) (p-value={p_value})",
                    xaxis_title=f"Variable: {i}",
                    yaxis_title="y",
                )
                fig.show()


if __name__ == "__main__":
    sys.exit(main())
