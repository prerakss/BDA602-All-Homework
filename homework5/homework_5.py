import os
import sys

import numpy
import numpy as np
import pandas
import pandas as pd
import seaborn
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors
import sklearn.neural_network
import sklearn.tree
import statsmodels.api
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from pyspark.sql import SparkSession
from scipy.stats import pearsonr


def model_evaluation(df, response, predictors, game_id):
    # The following train,test split yields 50% accuracy for both the models - KNN and Neural Networks

    x_train = df[df[game_id] <= 12930][predictors]
    y_train = df[df[game_id] <= 12930][response]
    x_test = df[df[game_id] > 12930][predictors]
    y_test = df[df[game_id] > 12930][response]

    # The following commented train,test split yields 52% accuracy for KNN, and 46% accuracy for Neural network

    # x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(df[predictors], df[response])

    # Model 1
    knn_clf = sklearn.neighbors.KNeighborsClassifier(50)
    knn_clf = knn_clf.fit(x_train, y_train)
    knn_predictions = knn_clf.predict(x_test)
    knn_accuracy = sklearn.metrics.accuracy_score(y_test, knn_predictions)
    print("K-nearest neighbors accuracy:", knn_accuracy)

    # Model 2
    nn_clf = sklearn.neural_network.MLPClassifier()
    nn_clf = nn_clf.fit(x_train, y_train)
    nn_predictions = nn_clf.predict(x_test)
    nn_accuracy = sklearn.metrics.accuracy_score(y_test, nn_predictions)
    print("Neural network accuracy:", nn_accuracy)


# CORRELATION MATRIX AND CORRELATION TABLES

# CONTINUOUS - CONTINUOUS PAIR CORRELATION MATRIX AND TABLE
def continuous_continuous_pairs(df, response, cont_predictors):
    # Correlation Matrix for cont-cont

    s = seaborn.heatmap(
        df[cont_predictors].corr(),
        annot=True,
        vmax=1,
        vmin=-1,
        center=0,
        cmap="vlag",
    )
    s = s.get_figure()
    s.savefig("Continuous_Continuous_correlation_matrix.png")

    # Correlation table logic
    corr_values = pd.DataFrame(
        columns=["predictor_1", "predictor_2", "corr_value", "abs_corr_value"]
    )
    temp_corr = []
    table = pd.DataFrame(columns=["predictor", "file_link"])

    for i in cont_predictors:

        for j in cont_predictors:
            if i == j:
                continue
            else:
                corr, _ = pearsonr(df[i], df[j])

                if corr not in temp_corr:
                    temp_corr.append(corr)
                    corr_values.loc[len(corr_values)] = [i, j, corr, abs(corr)]

        predictor_name = statsmodels.api.add_constant(df[i])
        y = df[response]
        y, temp_var = y.factorize()

        linear_regression_model = statsmodels.api.Logit(y, predictor_name)
        linear_regression_model_fitted = linear_regression_model.fit()
        t_value = round(linear_regression_model_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])
        # print(f"t value: {t_value}", f"p value: {p_value}")

        fig = px.scatter(x=df[i], y=df[response], trendline="ols")
        fig.update_layout(
            title=f"Variable: {i}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {i}",
            yaxis_title=f"{response}",
        )
        filename = f"{response}_{i}_plot.html"
        fig.write_html(file=filename, include_plotlyjs="cdn")

        table.loc[len(table)] = [i, filename]

    table["file_link"] = os.getcwd() + "/" + table["file_link"]
    temp_table = pd.merge(
        corr_values, table, how="inner", left_on="predictor_1", right_on="predictor"
    )
    final_table = pd.merge(
        temp_table, table, how="inner", left_on="predictor_2", right_on="predictor"
    )
    final_table = final_table[
        [
            "predictor_1",
            "predictor_2",
            "corr_value",
            "abs_corr_value",
            "file_link_x",
            "file_link_y",
        ]
    ]
    final_table = final_table.rename(
        columns={
            "file_link_x": "regression_plot_link_predictor_1",
            "file_link_y": "regression_plot_link_predictor_2",
        }
    )
    final_table.sort_values(by="corr_value", ascending=False, inplace=True)

    final_table["regression_plot_link_predictor_1"] = final_table.apply(
        lambda x: '<a href="{}">{}</a>'.format(
            x["regression_plot_link_predictor_1"], x["regression_plot_link_predictor_1"]
        ),
        axis=1,
    )
    final_table["regression_plot_link_predictor_2"] = final_table.apply(
        lambda x: '<a href="{}">{}</a>'.format(
            x["regression_plot_link_predictor_2"], x["regression_plot_link_predictor_2"]
        ),
        axis=1,
    )
    final_table.to_html(
        "Continuous continuous rankings and regression plot.html",
        escape=False,
        render_links=True,
    )


# BRUTE FORCE TABLES
def two_variable_mean_response_brute_force(df, response, cont_predictors):
    cont_cont_brute_force_table = pd.DataFrame(
        columns=[
            "predictor_1",
            "predictor_2",
            "unweighted mean of response",
            "weighted mean of response",
            "unweighted plot link",
            "weighted plot link",
        ]
    )

    y = response

    # Cont-Cont Brute force
    for p1 in cont_predictors:
        for p2 in cont_predictors:
            df_temp = df  # [[p1,p2]]
            if p1 != p2:
                p1_binning, p2_binning = "Bins:" + p1, "Bins:" + p2

                df_temp[p1_binning] = pd.cut(x=df_temp[p1], bins=10)
                df_temp[p2_binning] = pd.cut(x=df_temp[p1], bins=10)

                mean = {y: np.mean}
                length = {y: np.size}
                mean_values = (
                    df_temp.groupby([p1_binning, p2_binning]).agg(mean).reset_index()
                )
                lengths = (
                    df_temp.groupby([p1_binning, p2_binning]).agg(length).reset_index()
                )

                mean_values_merged = pd.merge(
                    mean_values,
                    lengths,
                    how="left",
                    left_on=[p1_binning, p2_binning],
                    right_on=[p1_binning, p2_binning],
                )

                mean_values_merged["population_mean"] = df_temp[response].mean()
                mean_values_merged["population"] = mean_values_merged[
                    response + "_y"
                ].sum()

                mean_values_merged["diff with mean of response"] = (
                    mean_values_merged["population_mean"]
                    - mean_values_merged[response + "_x"]
                )

                mean_values_merged["squared diff"] = (
                    1
                    / 100
                    * np.power(mean_values_merged["diff with mean of response"], 2)
                )

                mean_values_merged["weighted diff with mean of response"] = (
                    np.power((mean_values_merged["diff with mean of response"]), 2)
                    * mean_values_merged[response + "_y"]
                    / mean_values_merged["population"]
                )

                unweigh_diff = [mean_values_merged["squared diff"].sum()]
                weigh_diff = [
                    mean_values_merged["weighted diff with mean of response"].sum()
                ]

                fig = go.Figure(
                    data=go.Heatmap(
                        {
                            "z": mean_values_merged[
                                "diff with mean of response"
                            ].tolist(),
                            "x": mean_values_merged[p1_binning].astype(str).tolist(),
                            "y": mean_values_merged[p2_binning].astype(str).tolist(),
                        }
                    )
                )
                fig.update_layout(
                    title="Unweighted Difference with Mean of Response for" + y,
                    xaxis_title=p1 + " bins",
                    yaxis_title=p2 + " bins",
                )

                fig2 = go.Figure(
                    data=go.Heatmap(
                        {
                            "z": mean_values_merged[
                                "weighted diff with mean of response"
                            ].tolist(),
                            "x": mean_values_merged[p1_binning].astype(str).tolist(),
                            "y": mean_values_merged[p2_binning].astype(str).tolist(),
                        }
                    ),
                )
                fig2.update_layout(
                    title="Weighted Difference with Mean of Response for" + y,
                    xaxis_title=p1 + " bins",
                    yaxis_title=p2 + " bins",
                )
                filename = f"{p1}_{p2}_unweighted_difference_with_mean_of_response.html"
                filename_2 = f"{p1}_{p2}_weighted_difference_with_mean_of_response.html"
                fig.write_html(file=filename, include_plotlyjs="cdn")
                fig2.write_html(file=filename_2, include_plotlyjs="cdn")

                cont_cont_brute_force_table.loc[len(cont_cont_brute_force_table)] = [
                    p1,
                    p2,
                    unweigh_diff,
                    weigh_diff,
                    os.getcwd() + "/" + filename,
                    os.getcwd() + "/" + filename_2,
                ]

    cont_cont_brute_force_table[
        "unweighted plot link"
    ] = cont_cont_brute_force_table.apply(
        lambda x: '<a href="{}">{}</a>'.format(
            x["unweighted plot link"], x["unweighted plot link"]
        ),
        axis=1,
    )

    cont_cont_brute_force_table[
        "weighted plot link"
    ] = cont_cont_brute_force_table.apply(
        lambda x: '<a href="{}">{}</a>'.format(
            x["weighted plot link"], x["weighted plot link"]
        ),
        axis=1,
    )

    cont_cont_brute_force_table.to_html(
        "cont_cont_brute_force_table.html", escape=False, render_links=True
    )


def normal_mean_of_response(df, response, cont_predictors):
    cont_diff_table = pd.DataFrame(
        columns=["predictor", "unweighted diff", "weighted diff"]
    )
    y = response
    df_temp = df
    for p in cont_predictors:
        bin = pd.cut(df[p], 10)
        w = bin.value_counts() / sum(bin.value_counts())
        w = w.reset_index()
        _mean = np.mean(df[y])
        response_mean = (
            df_temp[y].groupby(bin).apply(np.mean).reset_index(name="Mean-response")
        )

        # diff = np.square(response_mean.iloc[:, 1] - _mean)
        unweigh_diff = np.sum(np.square(response_mean.iloc[:, 1] - _mean)) / len(
            response_mean
        )
        weigh_diff = (
            np.sum(w.iloc[:, 1] * (np.square(response_mean.iloc[:, 1] - _mean)))
            / len(response_mean)
            * response_mean["Mean-response"].count()
        )

        cont_diff_table.loc[len(cont_diff_table)] = [p, unweigh_diff, weigh_diff]
        cont_diff_table.to_html("normal_mean_of_response_continuous.html")


def homework_4_plots(df, response, cont_predictors):
    plots_df = pd.DataFrame(columns=["response", "predictor", "plot_link"])

    # Cat response, cont pred
    for p in cont_predictors:
        hist_data = [df[p]]
        group_labels = [p]
        fig_1 = ff.create_distplot(hist_data, group_labels)
        fig_1.update_layout(
            title=f"Distribution for {p}",
            xaxis_title="Predictor",
            yaxis_title="Distribution",
        )

        filename = f"{response}_{p}_distributionPlot_hw4.html"
        fig_1.write_html(file=filename, include_plotlyjs="cdn")
        plots_df.loc[len(plots_df)] = [
            response,
            p,
            os.getcwd() + "/" + filename,
        ]

        fig_2 = go.Figure(
            data=go.Violin(
                y=df[response],
                x=df[p],
                line_color="black",
                box_visible=True,
                meanline_visible=True,
            )
        )
        fig_2.update_layout(
            title=f"Violin plot for {response} and {p}",
            xaxis_title=response,
            yaxis_title=p,
        )
        filename = f"{response}_{p}_violin_hw4.html"
        fig_2.write_html(file=filename, include_plotlyjs="cdn")
        plots_df.loc[len(plots_df)] = [
            response,
            p,
            os.getcwd() + "/" + filename,
        ]

    plots_df["plot_link"] = plots_df.apply(
        lambda x: '<a href="{}">{}</a>'.format(x["plot_link"], x["plot_link"]), axis=1
    )

    plots_df.to_html("All_Homework_4_plots.html", escape=False, render_links=True)


def check_response_type(response):
    if len(set(response)) < 3:
        var = "Categorical"
        print("Response variable is boolean!")
    else:
        var = "Continuous"
        print("Response variable is continuous!")
    return var


def check_predictor_type(predictor):
    if predictor.dtype == "object":
        return "Categorical"
    else:
        return "Continuous"


def fill_na(data):
    if isinstance(data, pandas.Series):
        return data.fillna(0)
    else:
        return numpy.array([value if value is not None else 0 for value in data])


def make_clickable(val):
    return '<a href="{}">{}</a>'.format(val, val)


def main():
    appName = "Homework 5 Baseball"
    master = "local"

    spark = (
        SparkSession.builder.appName(appName)
        .master(master)
        .config("spark.jars", "mariadb-java-client-3.0.8.jar")
        .enableHiveSupport()
        .getOrCreate()
    )

    # Fetching tables from mariadb database
    sql = "select * from baseball.baseball_features"

    database = "baseball"
    user = "root"
    df_pass = "root"
    server = "localhost"
    port = 3306
    jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"

    spark_df = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("query", sql)
        .option("user", user)
        .option("password", df_pass)
        .option("driver", jdbc_driver)
        .load()
    )

    df = spark_df.toPandas()

    response = "HomeTeamWins"
    predictors = [
        "H_rolling_strikeout",
        "H_rolling_walk",
        "H_rolling_hits",
        "H_rolling_outs_played",
        "H_rolling_pitches_thrown",
        "A_rolling_strikeout",
        "A_rolling_walk",
        "A_rolling_hits",
        "A_rolling_outs_played",
        "A_rolling_pitches_thrown",
        "H_rolling_batting_avg",
        "H_total_bases",
        "A_rolling_batting_avg",
        "A_total_bases",
    ]
    game_id = "game_id"

    df[response] = fill_na(df[response])
    df[response] = pd.to_numeric(df[response])
    for i in predictors:
        df[i] = fill_na(df[i])
        if df[i].dtype == "object":
            df[i] = df[i].astype("float64")

    print(check_response_type(df[response]))

    # List of categorical and continuous predictors separated in two lists
    cat_predictors = []
    cont_predictors = []

    for i in predictors:
        if check_predictor_type(df[i]) == "Categorical":
            cat_predictors.append(i)
        elif check_predictor_type(df[i]) == "Continuous":
            cont_predictors.append(i)

    print("Categorical predictors are -", cat_predictors)
    print("Continuous predictors are -", cont_predictors)

    # Calling functions

    # Evaluate two ML models
    model_evaluation(df, response, predictors, game_id)

    # Build correlation matrix and correlation value table
    continuous_continuous_pairs(df, response, cont_predictors)

    # Brute force table
    two_variable_mean_response_brute_force(df, response, cont_predictors)

    # Normal mean of response
    normal_mean_of_response(df, response, cont_predictors)

    # Re-build hw4 plots
    homework_4_plots(df, response, cont_predictors)


if __name__ == "__main__":
    sys.exit(main())
