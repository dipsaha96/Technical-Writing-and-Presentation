import numpy as np

import matplotlib.pyplot as plt

class ContinuousSignal:
    def __init__(self, func, INF=5):
        self.func = func
        self.INF = INF

    def shift(self, shift):
        def shifted_func(t):
            return self.func(t - shift)
        return ContinuousSignal(shifted_func)

    def add(self, other):
        def added_func(t):
            return self.func(t) + other.func(t)
        return ContinuousSignal(added_func)

    def multiply(self, other):
        def multiplied_func(t):
            return self.func(t) * other.func(t)
        return ContinuousSignal(multiplied_func)

    def multiply_const_factor(self, scaler):
        def scaled_func(t):
            return self.func(t) * scaler
        return ContinuousSignal(scaled_func)

    # def plot(self, t_range=(-INF, INF), num_points=1000):
    #     t = np.linspace(t_range[0], t_range[1], num_points)
    #     y = self.func(t)
    #     plt.plot(t, y)
    #     plt.xlabel('t')
    #     plt.ylabel('x(t)')
    #     plt.title('Continuous Signal')
    #     plt.grid(True)
    #     plt.show()

    def check_plot(self, minheight=-2, maxheight=2, y_tick_spacing=0.5, color="blue"):
        t = np.linspace(-self.INF, self.INF + 0.01, 1000)
        time_indices = np.arange(-self.INF, self.INF + 1,1)
        plt.figure(figsize=(8, 3))
        plt.xticks(time_indices)
        plt.plot(t, self.func(t), color=color)
        plt.ylim([minheight - 0.1, maxheight + 0.3])
        plt.yticks(np.arange(minheight, maxheight + y_tick_spacing, y_tick_spacing))
        plt.title("Continuous Signal")
        plt.xlabel("t(Time)")
        plt.ylabel("x(t)")
        plt.grid(True)
        plt.show()

    def plot(
        self,
        continuousSignals: list["ContinuousSignal"],
        title,
        subTitle,
        subplotTitles,
        rows,
        columns,
        saveTo,
        minheight=0,
        maxheight=1,
        y_tick_spacing=0.5,
        samePlot=False,
        label1="",
        label2="",
    ):
        t = np.linspace(-self.INF, self.INF + 0.01, 1000)
        reconstructed_signal = continuousSignals[len(continuousSignals) - 1]
        continuousSignals = continuousSignals[: len(continuousSignals) - 1]

        # Create a figure with multiple subplots (4 rows, 3 columns)
        fig, axs = plt.subplots(rows, columns, figsize=(10, 10))

        # Title for the entire figure
        fig.suptitle(subTitle, fontsize=16)

        # Plot the individual impulses δ/h[t-k▽] * x[t] * ▽
        row, col = 0, 0
        for continuousSignal, subplotTitle in zip(continuousSignals, subplotTitles):
            axs[row, col].set_xticks(np.arange(-self.INF, self.INF + 1, 1))
            axs[row, col].set_yticks(
                np.arange(0, maxheight + y_tick_spacing, y_tick_spacing)
            )
            if samePlot:
                axs[row, col].plot(t, continuousSignal.func(t), label=label1)
                axs[row, col].plot(t, self.func(t), color="red", label=label2)
            else:
                axs[row, col].plot(t, continuousSignal.func(t))
            axs[row, col].set_ylim([minheight, maxheight])
            axs[row, col].set_title(subplotTitle)
            axs[row, col].set_xlabel("t(Time)")
            axs[row, col].set_ylabel("x[t]")
            if samePlot:
                axs[row, col].legend()
            axs[row, col].grid(True)
            col += 1
            if col == columns:
                col = 0
                row += 1

        # Plot the sum of all impulse responses in the last subplot
        axs[row, col].set_xticks(np.arange(-self.INF, self.INF + 1, 1))
        axs[row, col].set_yticks(
            np.arange(0, maxheight + y_tick_spacing, y_tick_spacing)
        )
        if samePlot:
            axs[row, col].plot(t, reconstructed_signal.func(t), label=label1)
            axs[row, col].plot(t, self.func(t), color="red", label=label2)
        else:
            axs[row, col].plot(t, reconstructed_signal.func(t))
        axs[row, col].set_ylim([minheight, maxheight])
        axs[row, col].set_title(subplotTitles[len(subplotTitles) - 1])
        axs[row, col].set_xlabel("t(Time)")
        axs[row, col].set_ylabel("x[t]")
        if samePlot:
            axs[row, col].legend()
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
    