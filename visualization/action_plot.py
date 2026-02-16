import matplotlib.pyplot as plt

 # TODO this code is not working well
def plot_action_counts(counts):
    actions = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(8, 5))
    plt.bar(actions, values, color=["green", "blue", "gray"])
    plt.title("Action Frequency After 1000 Steps")
    plt.xlabel("Actions")
    plt.ylabel("Times Chosen")
    plt.tight_layout()
    plt.savefig("action_frequency.png")
    print("📊 Saved action_frequency.png")
