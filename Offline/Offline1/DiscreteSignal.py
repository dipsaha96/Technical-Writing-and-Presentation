import numpy as np

import matplotlib.pyplot as plt

class DiscreteSignal:
    def __init__(self, INF):
        self.INF = INF
        self.values = np.zeros(2 * INF + 1)

    def set_value_at_time(self, time: int, value):
        if 0 <= time <= 2*self.INF+1:
            self.values[time] = value
        else:
            raise ValueError("Time index out of range")

    def shift_signal(self, shift: int):
        new_signal = DiscreteSignal(self.INF)
        if(shift >= 0):
            new_signal.values = np.concatenate((np.zeros(shift), self.values[: len(self.values) - shift]))
        elif(shift < 0):
            new_signal.values = np.concatenate((self.values[-shift:], np.zeros(-shift)))
        return new_signal

    def add(self, other):
        new_signal = DiscreteSignal(self.INF)
        new_signal.values = self.values + other.values
        return new_signal

    def multiply(self, other):
        if self.INF != other.INF:
            raise ValueError("Signals must have the same INF value")
        new_signal = DiscreteSignal(self.INF)
        new_signal.values = self.values * other.values
        return new_signal

    def multiply_const_factor(self, scaler):
        new_signal = DiscreteSignal(self.INF)
        new_signal.values = self.values * scaler
        return new_signal

    def check_plot(self):
        time_indices = np.arange(-self.INF, self.INF + 1,1)
        plt.figure(figsize=(8,3))
        plt.xticks(time_indices)
        plt.stem(time_indices, self.values)
        y_range = (-1, max(np.max(self.values), 3) + 1)
        plt.ylim(*y_range)
        plt.xlabel("n (Time Index)")
        plt.ylabel("x[n]")
        plt.title("Discrete Signal")
        plt.grid(True)
        plt.show()

    def plot(
        self,
        DiscreteSignals: list["DiscreteSignal"],
        title,
        subTitle,
        subplotTitles,
        saveTo,
    ):
        final_response = DiscreteSignals[len(DiscreteSignals) - 1]
        DiscreteSignals = DiscreteSignals[: len(DiscreteSignals) - 1]

        # Create a figure with multiple subplots (4 rows, 3 columns)
        fig, axs = plt.subplots(4, 3, figsize=(10, 10))
        y_range = (-1, max(np.max(self.values), 3) + 1)

        # Title for the entire figure
        fig.suptitle(subTitle, fontsize=16)

        # Plot the individual impulses δ[n-k] * x[k]
        row, col = 0, 0
        for DiscreteSignal, subplotTitle in zip(DiscreteSignals, subplotTitles):
            axs[row, col].stem(
                np.arange(-self.INF, self.INF + 1, 1),
                DiscreteSignal.values,
                basefmt="r-",
            )
            axs[row, col].set_xticks(np.arange(-self.INF, self.INF + 1, 1))
            axs[row, col].set_ylim(*y_range)
            axs[row, col].set_title(subplotTitle)
            axs[row, col].set_xlabel("n (Time Index)")
            axs[row, col].set_ylabel("x[n]")
            axs[row, col].grid(True)
            col += 1
            if col == 3:
                col = 0
                row += 1

        # Plot the sum of all impulse responses in the last subplot
        axs[row, col].stem(
            np.arange(-self.INF, self.INF + 1, 1), final_response.values, basefmt="r-"
        )
        axs[row, col].set_xticks(np.arange(-self.INF, self.INF + 1, 1))
        axs[row, col].set_ylim(*y_range)
        axs[row, col].set_title(subplotTitles[len(subplotTitles) - 1])
        axs[row, col].set_xlabel("n(Time Index)")
        axs[row, col].set_ylabel("x[n]")
        axs[row, col].grid(True)

        # Adjust layout to prevent overlapping of plots
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.1)

        # Add a caption below the figure
        fig.text(
            0.5,
            0.01,
            title,
            ha="center",
            fontsize=12,
        )

        # Save figure
        plt.savefig(saveTo)

        # Display the plot
        # plt.show()


if __name__ == "__main__":
    INF = 5
    x = DiscreteSignal(INF)
    y= DiscreteSignal(INF)
    y.set_value_at_time( INF - 1, 1)
    y.set_value_at_time( INF, 2)
    x.set_value_at_time( INF + 0, 0.5)
    x.set_value_at_time( INF + 1, 1)
    x.add(y).check_plot()