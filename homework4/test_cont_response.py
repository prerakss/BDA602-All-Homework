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
    url = "https://raw.githubusercontent.com/plotly/datasets/master/tips.csv"
    urlData = requests.get(url).content
    df = pd.read_csv(io.StringIO(urlData.decode("utf-8")))
    response_1 = "tip"
    # response = np.array(df['tip'])
    predictors_1 = ["total_bill", "sex", "smoker", "day", "time", "size"]
    # predictors = np.array(df[['total_bill','sex','smoker','day','time','size']])
    y = df[response_1]
    X = df[predictors_1]
    # print(y)
    if len(set(y)) == 2:
        var = "Boolean"
        print("Response variable is boolean!")
    else:
        var = "Continuous"
        print(len(set(y)))
        print("Response variable is continuous!")

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


if __name__ == "__main__":
    sys.exit(main())
