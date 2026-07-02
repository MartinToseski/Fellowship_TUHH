import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import ast


matplotlib.use("Agg")
DATASET_PATH = "../../data/ptb-xl/"

# ----- PTBXL_DATABASE.CSV -----
print("= = = PTBXL_DATABASE.CSV = = =")
main_csv = pd.read_csv(DATASET_PATH + "ptbxl_database.csv")

rows = main_csv.shape[0]
cols = main_csv.shape[1]
print("Rows:", rows)
print("Columns:", cols, "\n")

num_cols = ['age', 'height', 'weight']
cat_cols = ['sex', 'device', 'heart_axis', 'infarction_stadium1', 'second_opinion', 'validated_by_human', 'static_noise', 'burst_noise', 'baseline_drift', 'electrodes_problems', 'extra_beats', 'pacemaker']

print("Data Quality Report (Numerical Columns):")
for col in num_cols:
    print("-", col.upper(), "-")
    vals = main_csv[col]
    print("Count:", vals.count())
    print("Missing %:", (rows-vals.count())/rows*100)
    print("Cardinality:", len(vals.unique()))
    print("Min:", vals.min())
    print("Q1:", vals.quantile(0.25))
    print("Mean:", vals.mean())
    print("Median:", vals.median())
    print("Q3:", vals.quantile(0.75))
    print("Max:", vals.max())    
    print("Standard Deviation:", vals.std())
    print()

    plt.figure()
    plt.boxplot(vals.dropna())
    plt.xticks([])
    plt.title(col)
    plt.savefig(f"vis/numerical/{col}_boxplot.png")
    plt.close()
    
    plt.figure()
    counts, bins, patches = plt.hist(vals.dropna())
    bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    plt.xticks(bins, bin_labels + [f"{bins[-1]:.1f}"], rotation=45, ha='right')
    plt.title(col)
    plt.tight_layout()
    plt.savefig(f"vis/numerical/{col}_histogram.png")
    plt.close()

print("Data Quality Report (Categorical Columns):")
for col in cat_cols:
    print("-", col.upper(), "-")
    vals = main_csv[col]
    print("Count:", vals.count())
    print("Missing %:", (rows-vals.count())/rows*100)
    print("Cardinality:", len(vals.unique()))
    
    counts = vals.value_counts()
    print("1st Mode:", counts.index[0])
    print("1st Mode Frequency:", counts.iloc[0])
    print("1st Mode %:", counts.iloc[0]/rows*100)
    print("2nd Mode:", counts.index[1])
    print("2nd Mode Frequency:", counts.iloc[1])
    print("2nd Mode %:", counts.iloc[1]/rows*100)
    print()

    if (len(vals.unique()) < 20):
        plt.figure()
        bars = plt.bar(counts.index.astype(str), counts.values)

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                str(int(height)),
                ha='center',
                va='bottom'
            )

        plt.xticks(rotation=45, ha='right')
        plt.title(col)
        plt.tight_layout()
        plt.savefig(f"vis/categorical/{col}_barchart.png")
        plt.close()
    else:
        counts = vals.value_counts().head(10)
        plt.figure()
        plt.bar(counts.index.astype(str), counts.values)

        bars = plt.bar(counts.index.astype(str), counts.values)

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                str(int(height)),
                ha='center',
                va='bottom'
            )

        plt.xticks(rotation=45, ha='right')
        plt.title(col + " top 10")
        plt.tight_layout()
        plt.savefig(f"vis/categorical/{col}_barchart.png")
        plt.close()

main_csv['scp_codes'] = main_csv['scp_codes'].apply(lambda x: ast.literal_eval(x))
print(main_csv['scp_codes'].head(), "\n")



print("= = = scp_statements.csv = = =")
codes_csv = pd.read_csv(DATASET_PATH + "scp_statements.csv", index_col=0)

def aggregate_diagnostic(y_dic):
    tmp = []
    for key, value in y_dic.items():
        if value > 0 and key in codes_csv.index:
            cls = codes_csv.loc[key, 'diagnostic_class']
            if pd.notna(cls):   
                tmp.append(cls)
    return list(set(tmp))

main_csv['superclasses'] = main_csv['scp_codes'].apply(aggregate_diagnostic)
superclass_len = main_csv['superclasses'].apply(len).value_counts().sort_index()

plt.figure()

bars = plt.bar(superclass_len.index.astype(str), superclass_len.values)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        str(int(height)),
        ha='center',
        va='bottom'
    )

plt.xticks()
plt.title("Superclass Codes Length Distribution")
plt.tight_layout()
plt.savefig(f"vis/superclass/superclass_length_distribution.png")
plt.close()

superclass_counts = main_csv['superclasses'].explode().dropna().value_counts()
print(superclass_counts)

plt.figure()
bars = plt.bar(superclass_counts.index.astype(str), superclass_counts.values)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        str(int(height)),
        ha='center',
        va='bottom'
    )

plt.xticks()
plt.title("Superclass Count Distribution")
plt.tight_layout()
plt.savefig(f"vis/superclass/superclass_count_distribution.png")
plt.close()


age_by_class = {}
weight_by_class = {}
height_by_class = {}
sex_by_class = {}
device_by_class = {}

for i in range(len(main_csv)):
    row = main_csv.iloc[i]
    classes = row['superclasses']

    for cls in classes:
        age_by_class.setdefault(cls, []).append(row['age'])
        weight_by_class.setdefault(cls, []).append(row['weight'])
        height_by_class.setdefault(cls, []).append(row['height'])
        sex_by_class.setdefault(cls, []).append(row['sex'])
        device_by_class.setdefault(cls, []).append(row['device'])

plt.figure(figsize=(10,6))

for i, cls in enumerate(sorted(age_by_class.keys())):
    plt.hist(
        age_by_class[cls],
        bins=20,
        alpha=0.5,
        histtype='step',
        label=cls
    )

plt.title("Age by Superclass")
plt.legend()
plt.tight_layout()
plt.savefig("vis/cross/age_by_superclass.png")
plt.close()


plt.figure(figsize=(10,6))

for i, cls in enumerate(sorted(weight_by_class.keys())):
    plt.hist(
        weight_by_class[cls],
        bins=20,
        alpha=0.5,
        histtype='step',
        label=cls
    )

plt.title("Weight by Superclass")
plt.legend()
plt.tight_layout()
plt.savefig("vis/cross/weight_by_superclass.png")
plt.close()


plt.figure(figsize=(10,6))

for i, cls in enumerate(sorted(height_by_class.keys())):
    plt.hist(
        height_by_class[cls],
        bins=20,
        alpha=0.5,
        histtype='step',
        label=cls
    )

plt.title("Height by Superclass")
plt.legend()
plt.tight_layout()
plt.savefig("vis/cross/height_by_superclass.png")
plt.close()



sex_percent = {}

total_patients = len(main_csv)

for cls in sex_by_class:
    vals = sex_by_class[cls]

    male = vals.count(0)
    female = vals.count(1)

    sex_percent[cls] = {
        "male": male / total_patients * 100,
        "female": female / total_patients * 100
    }

classes = sorted(sex_percent.keys())

male_vals = [sex_percent[c]["male"] for c in classes]
female_vals = [sex_percent[c]["female"] for c in classes]

x = np.arange(len(classes))
width = 0.35

plt.figure(figsize=(10, 6))

bars1 = plt.bar(x - width/2, male_vals, width,
                label="Male", color="tab:blue")

bars2 = plt.bar(x + width/2, female_vals, width,
                label="Female", color="tab:orange")

for bars in (bars1, bars2):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.xticks(x, classes)
plt.ylabel("Percentage of All Patients (%)")
plt.title("Sex Distribution by Diagnostic Superclass")
plt.legend()

plt.tight_layout()
plt.savefig("vis/cross/sex_by_superclass.png")
plt.close()


device_percent = {}

for cls in device_by_class:
    vals = device_by_class[cls]
    total = len(vals)

    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1

    device_percent[cls] = {k: v / total * 100 for k, v in counts.items()}

classes = sorted(device_percent.keys())
devices = sorted({d for x in device_percent.values() for d in x})

device_totals = {}
for dev in devices:
    device_totals[dev] = sum(device_percent[c].get(dev, 0) for c in classes)

devices = sorted(devices, key=lambda d: device_totals[d], reverse=True)
colors = plt.cm.tab10(np.linspace(0, 1, len(devices)))
bottom = np.zeros(len(classes))

plt.figure(figsize=(12, 6))

for color, dev in zip(colors, devices):
    vals = [device_percent[c].get(dev, 0) for c in classes]

    plt.bar(
        classes,
        vals,
        bottom=bottom,
        label=dev,
        color=color
    )

    bottom += np.array(vals)

plt.ylabel("Percentage")
plt.title("Device Distribution by Superclass")
plt.legend(title="Device", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("vis/cross/device_by_superclass.png")
plt.close()


age_bins = [0, 20, 40, 60, 80, 100]
age_labels = ['0-19', '20-39', '40-59', '60-79', '80+']

main_csv['age_group'] = pd.cut(
    main_csv['age'],
    bins=age_bins,
    labels=age_labels,
    include_lowest=True
)

age_group_counts = {}

for cls in sorted(age_by_class.keys()):
    age_group_counts[cls] = [0] * len(age_labels)

for i in range(len(main_csv)):
    group = main_csv.iloc[i]['age_group']

    if pd.isna(group):
        continue

    idx = age_labels.index(str(group))

    for cls in main_csv.iloc[i]['superclasses']:
        age_group_counts[cls][idx] += 1

x = np.arange(len(age_labels))
width = 0.15

plt.figure(figsize=(12,6))

classes = sorted(age_group_counts.keys())

for i, cls in enumerate(classes):
    plt.bar(
        x + (i - 2) * width,
        age_group_counts[cls],
        width,
        label=cls
    )

plt.xticks(x, age_labels)
plt.xlabel("Age Group")
plt.ylabel("Number of Patients")
plt.title("Diagnostic Superclasses by Age Group")
plt.legend()
plt.tight_layout()
plt.savefig("vis/cross/superclass_by_age_group.png")
plt.close()


# ----- Age Group by Sex -----

age_bins = [0, 20, 40, 60, 80, 100]
age_labels = ['0-19', '20-39', '40-59', '60-79', '80+']

main_csv['age_group'] = pd.cut(
    main_csv['age'],
    bins=age_bins,
    labels=age_labels,
    include_lowest=True
)

male_counts = []
female_counts = []

for group in age_labels:
    subset = main_csv[main_csv['age_group'] == group]

    male_counts.append((subset['sex'] == 0).sum())
    female_counts.append((subset['sex'] == 1).sum())

x = np.arange(len(age_labels))
width = 0.35

plt.figure(figsize=(10,6))

bars1 = plt.bar(
    x - width/2,
    male_counts,
    width,
    label='Male',
    color='tab:blue'
)

bars2 = plt.bar(
    x + width/2,
    female_counts,
    width,
    label='Female',
    color='tab:orange'
)

for bars in (bars1, bars2):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height,
            str(int(height)),
            ha='center',
            va='bottom',
            fontsize=9
        )

plt.xticks(x, age_labels)
plt.xlabel("Age Group")
plt.ylabel("Number of Patients")
plt.title("Patient Distribution by Age Group and Sex")
plt.legend()

plt.tight_layout()
plt.savefig("vis/cross/age_group_sex_distribution.png")
plt.close()


# ----- Superclass Correlation Heatmap -----

classes = sorted(superclass_counts.index)

corr = pd.DataFrame(
    0,
    index=classes,
    columns=classes
)

for superclasses in main_csv['superclasses']:
    for c1 in superclasses:
        for c2 in superclasses:
            corr.loc[c1, c2] += 1

plt.figure(figsize=(7,6))

plt.imshow(corr, cmap='Blues')

plt.xticks(range(len(classes)), classes)
plt.yticks(range(len(classes)), classes)

plt.colorbar(label="Co-occurrences")

for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(
            j,
            i,
            corr.iloc[i, j],
            ha='center',
            va='center',
            fontsize=9,
            color='black'
        )

plt.title("Diagnostic Superclass Co-occurrence")
plt.tight_layout()
plt.savefig("vis/cross/superclass_correlation.png")
plt.close()