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

    total_NORM = sum("NORM" in labels for labels in y_train) + sum("NORM" in labels for labels in y_val) + sum("NORM" in labels for labels in y_test)
    total_MI = sum("MI" in labels for labels in y_train) + sum("MI" in labels for labels in y_val) + sum("MI" in labels for labels in y_test)
    total_STTC = sum("STTC" in labels for labels in y_train) + sum("STTC" in labels for labels in y_val) + sum("STTC" in labels for labels in y_test)
    total_CD = sum("CD" in labels for labels in y_train) + sum("CD" in labels for labels in y_val) + sum("CD" in labels for labels in y_test)
    total_HYP = sum("HYP" in labels for labels in y_train) + sum("HYP" in labels for labels in y_val) + sum("HYP" in labels for labels in y_test)

    print("Total superclass distribution...")
    print("NORM   ", total_NORM, "  ", round(total_NORM/total_rows*100, 2), "% of dataset")
    print("MI     ", total_MI, "  ", round(total_MI/total_rows*100, 2), "% of dataset")
    print("STTC   ", total_STTC, "  ", round(total_STTC/total_rows*100, 2), "% of dataset")
    print("CD     ", total_CD, "  ", round(total_CD/total_rows*100, 2), "% of dataset")
    print("HYP    ", total_HYP, "  ", round(total_HYP/total_rows*100, 2), "% of dataset")
    print()

    print("Per split superclass distribution...")
    print(f"y_train")
    sum_NORM = sum("NORM" in labels for labels in y_train)
    print("NORM   ", sum_NORM, "  ", round(sum_NORM/len(y_train)*100, 2), "%   ", round(sum_NORM/total_NORM*100, 2), "% of total")
    sum_MI = sum("MI" in labels for labels in y_train)
    print("MI     ", sum_MI, "  ", round(sum_MI/len(y_train)*100, 2), "%   ", round(sum_MI/total_MI*100, 2), "% of total")
    sum_STTC = sum("STTC" in labels for labels in y_train)
    print("STTC   ", sum_STTC, "  ", round(sum_STTC/len(y_train)*100, 2), "%   ", round(sum_STTC/total_STTC*100, 2), "% of total")
    sum_CD = sum("CD" in labels for labels in y_train)
    print("CD     ", sum_CD, "  ", round(sum_CD/len(y_train)*100, 2), "%   ", round(sum_CD/total_CD*100, 2), "% of total")
    sum_HYP = sum("HYP" in labels for labels in y_train)
    print("HYP    ", sum_HYP, "  ", round(sum_HYP/len(y_train)*100, 2), "%   ", round(sum_HYP/total_HYP*100, 2), "% of total")
    print()

    print(f"y_val")
    sum_NORM = sum("NORM" in labels for labels in y_val)
    print("NORM   ", sum_NORM, "  ", round(sum_NORM/len(y_val)*100, 2), "%   ", round(sum_NORM/total_NORM*100, 2), "% of total")
    sum_MI = sum("MI" in labels for labels in y_val)
    print("MI     ", sum_MI, "  ", round(sum_MI/len(y_val)*100, 2), "%   ", round(sum_MI/total_MI*100, 2), "% of total")
    sum_STTC = sum("STTC" in labels for labels in y_val)
    print("STTC   ", sum_STTC, "  ", round(sum_STTC/len(y_val)*100, 2), "%   ", round(sum_STTC/total_STTC*100, 2), "% of total")
    sum_CD = sum("CD" in labels for labels in y_val)
    print("CD     ", sum_CD, "  ", round(sum_CD/len(y_val)*100, 2), "%   ", round(sum_CD/total_CD*100, 2), "% of total")
    sum_HYP = sum("HYP" in labels for labels in y_val)
    print("HYP    ", sum_HYP, "  ", round(sum_HYP/len(y_val)*100, 2), "%   ", round(sum_HYP/total_HYP*100, 2), "% of total")
    print()

    print(f"y_test")
    sum_NORM = sum("NORM" in labels for labels in y_test)
    print("NORM   ", sum_NORM, "  ", round(sum_NORM/len(y_test)*100, 2), "%   ", round(sum_NORM/total_NORM*100, 2), "% of total")
    sum_MI = sum("MI" in labels for labels in y_test)
    print("MI     ", sum_MI, "  ", round(sum_MI/len(y_test)*100, 2), "%   ", round(sum_MI/total_MI*100, 2), "% of total")
    sum_STTC = sum("STTC" in labels for labels in y_test)
    print("STTC   ", sum_STTC, "  ", round(sum_STTC/len(y_test)*100, 2), "%   ", round(sum_STTC/total_STTC*100, 2), "% of total")
    sum_CD = sum("CD" in labels for labels in y_test)
    print("CD     ", sum_CD, "  ", round(sum_CD/len(y_test)*100, 2), "%   ", round(sum_CD/total_CD*100, 2), "% of total")
    sum_HYP = sum("HYP" in labels for labels in y_test)
    print("HYP    ", sum_HYP, "  ", round(sum_HYP/len(y_test)*100, 2), "%   ", round(sum_HYP/total_HYP*100, 2), "% of total")
    print()