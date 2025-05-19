import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("results.csv")

# Set style
sns.set(style="whitegrid")

# Group data by algorithm (method)
grouped = (
    df.groupby("method")
    .agg(
        {
            "cost": "mean",
            "total_distance": "mean",
            "processing_time": "mean",
            "distance_per_second": "mean",
            "cost_per_second": "mean",
        }
    )
    .reset_index()
)

# Plot: Average cost
plt.figure(figsize=(10, 6))
sns.barplot(x="cost", y="method", data=grouped, palette="Blues_d")
plt.title("Average Route Cost by Algorithm")
plt.xlabel("Cost")
plt.ylabel("Algorithm")
plt.tight_layout()
plt.savefig("avg_cost_per_algorithm.png")
plt.close()

# Plot: Average total distance
plt.figure(figsize=(10, 6))
sns.barplot(x="total_distance", y="method", data=grouped, palette="Greens_d")
plt.title("Average Total Distance by Algorithm")
plt.xlabel("Total Distance")
plt.ylabel("Algorithm")
plt.tight_layout()
plt.savefig("avg_distance_per_algorithm.png")
plt.close()

# Plot: Processing time (log scale)
plt.figure(figsize=(10, 6))
sns.barplot(x="processing_time", y="method", data=grouped, palette="Reds_d")
plt.xscale("log")
plt.title("Average Processing Time (Log Scale) by Algorithm")
plt.xlabel("Processing Time [s] (log scale)")
plt.ylabel("Algorithm")
plt.tight_layout()
plt.savefig("avg_processing_time_per_algorithm.png")
plt.close()

# Plot: Distance per second
plt.figure(figsize=(10, 6))
sns.barplot(x="distance_per_second", y="method", data=grouped, palette="Purples_d")
plt.title("Average Distance per Second by Algorithm")
plt.xlabel("Distance per Second")
plt.ylabel("Algorithm")
plt.tight_layout()
plt.savefig("distance_per_sec_per_algorithm.png")
plt.close()

# Plot: Cost per second
plt.figure(figsize=(10, 6))
sns.barplot(x="cost_per_second", y="method", data=grouped, palette="Oranges_d")
plt.title("Average Cost per Second by Algorithm")
plt.xlabel("Cost per Second")
plt.ylabel("Algorithm")
plt.tight_layout()
plt.savefig("cost_per_sec_per_algorithm.png")
plt.close()

print("✅ Charts saved as PNG files.")
