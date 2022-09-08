import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df_iris = pd.read_csv("iris.data")

df_iris.columns = [
    "sepal length",
    "sepal width",
    "petal length",
    "petal width",
    "class",
]

print("Statistical summary", df_iris.describe())

arr_iris = df_iris[
    ["sepal length", "sepal width", "petal length", "petal width"]
].values

print("Mean: ", np.mean(arr_iris, axis=0))
print("Min: ", np.min(arr_iris, axis=0))
print("Max: ", np.max(arr_iris, axis=0))
print("25% Quantile: ", np.quantile(arr_iris, q=0.25, axis=0))
print("50% Quantile: ", np.quantile(arr_iris, q=0.5, axis=0))
print("75% Quantile: ", np.quantile(arr_iris, q=0.75, axis=0))

fig_scatter = px.scatter(
    df_iris, x="sepal width", y="sepal length", color="class", symbol="class"
)
fig_scatter.show()

fig = px.pie(
    df_iris,
    values="petal width",
    names="class",
    color_discrete_sequence=px.colors.sequential.RdBu,
)
fig.show()

fig_violin = px.violin(df_iris, box=True, y="sepal width", color="class")
fig_violin.show()

aggregations = {
    "sepal length": lambda x: np.max(x),
    "sepal width": lambda x: np.max(x),
    "petal length": lambda x: np.mean(x),
    "petal width": lambda x: np.mean(x),
}
df_iris_grouped = df_iris.groupby("class", as_index=False).agg(aggregations)

fig_bar = px.bar(df_iris_grouped, x="class", y="sepal length", color="class")
fig_bar.show()

fig_bar_2 = px.bar(df_iris_grouped, x="sepal width", y="class", color="class")
fig_bar_2.show()


X_orig = df_iris[["sepal length", "sepal width", "petal length", "petal width"]].values
y = df_iris["class"].values

standard_scaler = StandardScaler()
standard_scaler.fit(X_orig)

X = standard_scaler.transform(X_orig)

random_forest = RandomForestClassifier(random_state=1234)
random_forest.fit(X, y)
prediction = random_forest.predict(X)
probability = random_forest.predict_proba(X)
print(prediction, probability)


svc = SVC()
svc.fit(X, y)
prediction_2 = svc.predict(X)
print(svc.score(X, y))


pipeline = Pipeline(
    [
        ("StandardScaler", StandardScaler()),
        ("RandomForest", RandomForestClassifier(random_state=1234)),
    ]
)


pipeline2 = Pipeline([("StandardScaler", StandardScaler()), ("SVC", SVC())])
pipeline.fit(X_orig, y)
pipeline2.fit(X_orig, y)

prediction_3 = pipeline.predict(X)
