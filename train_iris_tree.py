from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


def main():
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test, predictions, target_names=iris.target_names, zero_division=0
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    output_path = Path(__file__).resolve().parent / "iris_decision_tree.png"
    plt.figure(figsize=(14, 8))
    plot_tree(
        model,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True,
        rounded=True,
    )
    plt.title("Decision Tree for the Iris Dataset")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nDecision tree visualization saved to: {output_path}")


if __name__ == "__main__":
    main()
