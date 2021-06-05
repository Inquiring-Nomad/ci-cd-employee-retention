import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load data
df = pd.read_csv("data/HR_comma_sep.csv")
X = df.drop(['left'], axis=1)
y = df['left']
# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ordinal = ['salary']
onehot = ['sales']
numeric = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']

# Data transformation
ct = ColumnTransformer(
    [
        ("ordinal", OrdinalEncoder(categories=[['low', 'medium', 'high']]), ordinal),
        ("onehot", OneHotEncoder(), onehot),
        ("stdscal", StandardScaler(), numeric)
    ], remainder='passthrough')

df_tr = ct.fit_transform(X_train)


# Build the model
def build_nn_binclassifier():
    model = Sequential()
    model.add(Dense(20, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


nn_binaryclassifier = KerasClassifier(build_fn=build_nn_binclassifier, nb_epoch=10)

# Cross-validation
scores = cross_val_score(estimator=nn_binaryclassifier, X=df_tr, y=y_train, cv=5, n_jobs=-1)
print(f'Scores: {scores.mean()}')

# Evaluate model on test set
nn_binaryclassifier.fit(df_tr, y_train)
X_test_tr = ct.transform(X_test)
test_predictions = nn_binaryclassifier.predict(X_test_tr)
print(nn_binaryclassifier.score(X_test_tr, y_test))
