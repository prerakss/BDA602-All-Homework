import os
import random
import sys
import warnings
from typing import List

import numpy
import numpy as np
import pandas
import pandas as pd
import seaborn
import statsmodels.api
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from pyspark.sql import SparkSession
from scipy import stats
from scipy.stats import pearsonr
from sklearn import datasets

TITANIC_PREDICTORS = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "embarked",
    "parch",
    "fare",
    "who",
    "adult_male",
    "deck",
    "embark_town",
    "alone",
    "class",
]


def get_test_data_set(data_set_name: str = None) -> (pandas.DataFrame, List[str], str):
    """Function to load a few test data sets

    :param:
    data_set_name : string, optional
        Data set to load

    :return:
    data_set : :class:`pandas.DataFrame`
        Tabular data, possibly with some preprocessing applied.
    predictors :list[str]
        List of predictor variables
    response: str
        Response variable
    """
    seaborn_data_sets = ["mpg", "tips", "titanic", "titanic_2"]
    sklearn_data_sets = ["boston", "diabetes", "breast_cancer"]
    all_data_sets = seaborn_data_sets + sklearn_data_sets

    if data_set_name is None:
        data_set_name = random.choice(all_data_sets)
    else:
        if data_set_name not in all_data_sets:
            raise Exception(f"Data set choice not valid: {data_set_name}")

    if data_set_name in seaborn_data_sets:
        if data_set_name == "mpg":
            data_set = seaborn.load_dataset(name="mpg").dropna().reset_index()
            predictors = [
                "cylinders",
                "displacement",
                "horsepower",
                "weight",
                "acceleration",
                "origin",
                "name",
            ]
            response = "mpg"
        elif data_set_name == "tips":
            data_set = seaborn.load_dataset(name="tips").dropna().reset_index()
            predictors = [
                "total_bill",
                "sex",
                "smoker",
                "day",
                "time",
                "size",
            ]
            response = "tip"
        elif data_set_name == "titanic":
            data_set = seaborn.load_dataset(name="titanic").dropna()
            predictors = TITANIC_PREDICTORS
            response = "survived"
        elif data_set_name == "titanic_2":
            data_set = seaborn.load_dataset(name="titanic").dropna()
            predictors = TITANIC_PREDICTORS
            response = "alive"
    elif data_set_name in sklearn_data_sets:
        if data_set_name == "boston":
            print("lol")
            """data = datasets.load_boston()
            data_set = pandas.DataFrame(data.data, columns=data.feature_names)
            data_set["CHAS"] = data_set["CHAS"].astype(str)"""
        elif data_set_name == "diabetes":
            data = datasets.load_diabetes()
            data_set = pandas.DataFrame(data.data, columns=data.feature_names)
            data_set["gender"] = ["1" if i > 0 else "0" for i in data_set["sex"]]
        elif data_set_name == "breast_cancer":
            data = datasets.load_breast_cancer()
            data_set = pandas.DataFrame(data.data, columns=data.feature_names)

        data_set["target"] = data.target
        predictors = data.feature_names
        response = "target"

    print(f"Data set selected: {data_set_name}")
    return data_set, predictors, response


def check_response_type(response):

    if len(set(response)) < 3:  # Assuming there could be 3 classes (iris)
        var = "Categorical"
        print("Response variable is boolean!")
    else:
        var = "Continuous"
        print(len(set(response)))
        print("Response variable is continuous!")
    return var


def check_predictor_type(predictor):
    # all object types are categorical, int/float types with less than 10 unique values are categorical as well (size)
    if predictor.dtype == "object":
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


def fill_na(data):
    if isinstance(data, pandas.Series):
        return data.fillna(0)
    else:
        return numpy.array([value if value is not None else 0 for value in data])


def cat_cont_correlation_ratio(categories, values):
    """
    Correlation Ratio: https://en.wikipedia.org/wiki/Correlation_ratio
    SOURCE:
    1.) https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    :param categories: Numpy array of categories
    :param values: Numpy array of values
    :return: correlation
    """
    f_cat, _ = pandas.factorize(categories)
    cat_num = numpy.max(f_cat) + 1
    y_avg_array = numpy.zeros(cat_num)
    n_array = numpy.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values[numpy.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = numpy.average(cat_measures)
    y_total_avg = numpy.sum(numpy.multiply(y_avg_array, n_array)) / numpy.sum(n_array)
    numerator = numpy.sum(
        numpy.multiply(
            n_array, numpy.power(numpy.subtract(y_avg_array, y_total_avg), 2)
        )
    )
    denominator = numpy.sum(numpy.power(numpy.subtract(values, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numpy.sqrt(numerator / denominator)
    return eta


def cat_correlation(x, y, bias_correction=True, tschuprow=False):

    corr_coeff = numpy.nan
    try:
        x, y = fill_na(x), fill_na(y)
        crosstab_matrix = pandas.crosstab(x, y)
        n_observations = crosstab_matrix.sum().sum()

        yates_correct = True
        if bias_correction:
            if crosstab_matrix.shape == (2, 2):
                yates_correct = False

        chi2, _, _, _ = stats.chi2_contingency(
            crosstab_matrix, correction=yates_correct
        )
        phi2 = chi2 / n_observations

        # r and c are number of categories of x and y
        r, c = crosstab_matrix.shape
        if bias_correction:
            phi2_corrected = max(0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1))
            r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
            c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
            if tschuprow:
                corr_coeff = numpy.sqrt(
                    phi2_corrected / numpy.sqrt((r_corrected - 1) * (c_corrected - 1))
                )
                return corr_coeff
            corr_coeff = numpy.sqrt(
                phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
            )
            return corr_coeff
        if tschuprow:
            corr_coeff = numpy.sqrt(phi2 / numpy.sqrt((r - 1) * (c - 1)))
            return corr_coeff
        corr_coeff = numpy.sqrt(phi2 / min((r - 1), (c - 1)))
        return corr_coeff
    except Exception as ex:
        print(ex)
        if tschuprow:
            warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
        else:
            warnings.warn("Error calculating Cramer's V", RuntimeWarning)
        return corr_coeff


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

    df[response] = fill_na(df[response])
    df[response] = pd.to_numeric(df[response])
    for i in predictors:
        df[i] = fill_na(df[i])
        if df[i].dtype == "object":
            df[i] = df[i].astype("float64")

    print(check_response_type(df[response]))

    cat_predictors = []
    cont_predictors = []

    for i in predictors:
        if check_predictor_type(df[i]) == "Categorical":
            cat_predictors.append(i)
        elif check_predictor_type(df[i]) == "Continuous":
            cont_predictors.append(i)

    print("Categorical predictors are -", cat_predictors)
    print("Continuous predictors are -", cont_predictors)

    # CORRELATION MATRIX AND CORRELATION TABLES

    # CONTINUOUS - CONTINUOUS PAIR CORRELATION MATRIX AND TABLE

    def continuous_continuous_pairs():

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
        print(corr_values)

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

            linear_regression_model = statsmodels.api.OLS(df[response], predictor_name)
            linear_regression_model_fitted = linear_regression_model.fit()
            t_value = round(linear_regression_model_fitted.tvalues[1], 6)
            p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])
            print(f"t value: {t_value}", f"p value: {p_value}")

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
        final_table.to_csv("Cont_Cont_Correlation_table.csv")
        print("final table --", final_table)

    # BRUTE FORCE TABLES

    def two_variable_mean_response_brute_force():

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
                        df_temp.groupby([p1_binning, p2_binning])
                        .agg(mean)
                        .reset_index()
                    )
                    lengths = (
                        df_temp.groupby([p1_binning, p2_binning])
                        .agg(length)
                        .reset_index()
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
                                "x": mean_values_merged[p1_binning]
                                .astype(str)
                                .tolist(),
                                "y": mean_values_merged[p2_binning]
                                .astype(str)
                                .tolist(),
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
                                "x": mean_values_merged[p1_binning]
                                .astype(str)
                                .tolist(),
                                "y": mean_values_merged[p2_binning]
                                .astype(str)
                                .tolist(),
                            }
                        )
                    )
                    fig2.update_layout(
                        title="Weighted Difference with Mean of Response for" + y,
                        xaxis_title=p1 + " bins",
                        yaxis_title=p2 + " bins",
                    )
                    filename = (
                        f"{p1}_{p2}_unweighted_difference_with_mean_of_response.html"
                    )
                    filename_2 = (
                        f"{p1}_{p2}_weighted_difference_with_mean_of_response.html"
                    )
                    fig.write_html(file=filename, include_plotlyjs="cdn")
                    fig2.write_html(file=filename_2, include_plotlyjs="cdn")

                    cont_cont_brute_force_table.loc[
                        len(cont_cont_brute_force_table)
                    ] = [
                        p1,
                        p2,
                        unweigh_diff,
                        weigh_diff,
                        os.getcwd() + "/" + filename,
                        os.getcwd() + "/" + filename_2,
                    ]

        print(cont_cont_brute_force_table.to_csv("cont_cont_brute_force_table.csv"))

    def normal_mean_of_response():

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
            cont_diff_table.to_csv("normal_mean_of_response_continuous.csv")

    def homework_4_plots():

        response_type = check_response_type(df[response])
        plots_df = pd.DataFrame(columns=["response", "predictor", "plot_link"])

        if response_type == "Categorical":

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
                filename = f"{response}_{p}distributionPlot_hw4.html"
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

        plots_df.to_csv("All_Homework_4_plots.csv")

        print(plots_df)

    # Calling all the methods written inside main func

    continuous_continuous_pairs()
    two_variable_mean_response_brute_force()
    normal_mean_of_response()
    homework_4_plots()


if __name__ == "__main__":
    sys.exit(main())
