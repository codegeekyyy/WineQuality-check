import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def main():
    csv_path = "winequality-red.csv"
    try:
        wine_dataset = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read {csv_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Prepare features and label
    X = wine_dataset.drop('quality', axis=1)
    Y = wine_dataset['quality'].apply(lambda y: 1 if y >= 7 else 0)

    # Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

    # Train model
    model = RandomForestClassifier(random_state=3)
    model.fit(X_train, Y_train)

    # Evaluate
    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    print(f"Test accuracy: {test_data_accuracy:.4f}")

    # Example prediction
    input_data = (7.5, 0.5, 0.36, 6.1, 0.071, 17.0, 102.0, 0.9978, 3.35, 0.8, 10.5)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    print("Prediction for sample input:", int(prediction[0]))
    print("Good Quality Wine" if prediction[0] == 1 else "Bad Quality Wine")


if __name__ == "__main__":
    main()
