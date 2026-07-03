SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def print_all_sizes(comment, X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"{comment} split size...")
    print(f"Train: {len(X_train)}")
    print("Empty diagnosis in y_train:", (y_train.str.len() == 0).sum())
    print(f"Val:   {len(X_val)}")
    print("Empty diagnosis in y_val:", (y_val.str.len() == 0).sum())
    print(f"Test:  {len(X_test)}")
    print("Empty diagnosis in y_test:", (y_test.str.len() == 0).sum())
    print()


def remove_empty_diagnosis(X, Y):
    non_empty_train_mask = Y.str.len() > 0
    X = X[non_empty_train_mask]
    Y = Y[non_empty_train_mask]
    return X, Y


def print_superclass_distribution_statistics(X_train, y_train, X_val, y_val, X_test, y_test):
    total_rows = len(y_train) + len(y_val) + len(y_test)

    total_counts = {}
    for superclass in SUPERCLASSES:
        total_counts[superclass] = sum(superclass in labels for labels in y_train) + sum(superclass in labels for labels in y_val) + sum(superclass in labels for labels in y_test)

    print("Total superclass distribution...")
    for superclass in SUPERCLASSES:
        print(f"{superclass:<6} {total_counts[superclass]:5}   {total_counts[superclass]/total_rows*100:.2f}% of dataset")
    print()

    splits = [
        ("y_train", y_train),
        ("y_val", y_val),
        ("y_test", y_test),
    ]

    for split_name, labels in splits:
        print(split_name)
        for superclass in SUPERCLASSES:
            count = sum(superclass in x for x in labels)
            print(f"{superclass:<6} {count:5}    {count/len(labels)*100:.2f}%   {count/total_counts[superclass]*100:.2f}% of total")
        print()