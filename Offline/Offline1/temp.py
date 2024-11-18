import numpy as np
import matplotlib.pyplot as plt
import os

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
        if self.INF != other.INF:
            raise ValueError("Signals must have the same INF value")
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

    # def check_plot(self):
    #     time_indices = np.arange(-self.INF, self.INF + 1,1)
    #     plt.figure(figsize=(8,3))
    #     plt.xticks(time_indices)
    #     plt.stem(time_indices, self.values)
    #     y_range = (-1, max(np.max(self.values), 3) + 1)
    #     plt.ylim(*y_range)
    #     plt.xlabel("n (Time Index)")
    #     plt.ylabel("x[n]")
    #     plt.title("Discrete Signal")
    #     plt.grid(True)
    #     plt.show()

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


    # def check_plot(self, minheight=-2, maxheight=2, y_tick_spacing=0.5, color="blue"):
    #     t = np.linspace(-self.INF, self.INF + 0.01, 1000)
    #     time_indices = np.arange(-self.INF, self.INF + 1,1)
    #     plt.figure(figsize=(8, 3))
    #     plt.xticks(time_indices)
    #     plt.plot(t, self.func(t), color=color)
    #     plt.ylim([minheight - 0.1, maxheight + 0.3])
    #     plt.yticks(np.arange(minheight, maxheight + y_tick_spacing, y_tick_spacing))
    #     plt.title("Continuous Signal")
    #     plt.xlabel("t(Time)")
    #     plt.ylabel("x(t)")
    #     plt.grid(True)
    #     plt.show()

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

        # Create a figure with multiple subplots (rows, columns)
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


class LTI_Discrete:
    def __init__(self, impulse_response: DiscreteSignal):
        self.impulse_response = impulse_response

    def linear_combination_of_impulses(self, input_signal: DiscreteSignal):
        INF = input_signal.INF
        unit_impulses = []
        coefficients = []
        for i in range(-INF, INF + 1):
            coefficient = input_signal.values[INF + i]
            unit_impulse = DiscreteSignal(INF)
            unit_impulse.set_value_at_time(INF, 1)
            unit_impulses.append(unit_impulse.shift_signal(i))
            coefficients.append(coefficient)

        return unit_impulses, coefficients

    def output(self, input_signal: DiscreteSignal):
        INF = input_signal.INF
        output_signal = DiscreteSignal(INF)
        coefficients=[]
        constituent_impulses = []
        for i in range(-INF, INF + 1):
            coefficients.append(input_signal.values[INF + i])
            response = self.impulse_response.shift_signal(i)
            constituent_impulses.append(response)
            output_signal = output_signal.add(
                response.multiply_const_factor(input_signal.values[INF + i])
            )
        return output_signal, constituent_impulses, coefficients
    


class LTI_Continuous:
    def __init__(self, impulse_response: ContinuousSignal):
        self.impulse_response = impulse_response

    def linear_combination_of_impulses(self, input_signal:ContinuousSignal, delta: float):
        INF = input_signal.INF
        # t_values = np.arange(-input_signal.INF, input_signal.INF, delta)
        t_values = np.array(
            [
                -input_signal.INF + i * delta
                for i in range(int((input_signal.INF - (-input_signal.INF)) / delta))
            ]
        )
        impulses = []
        coefficients = []
        for t in t_values:

            impulse = ContinuousSignal(lambda tau,t=t: 1/delta * ((tau>=t) & (tau<=t+delta)), INF)
            coefficient = input_signal.func(t)
            impulses.append(impulse)
            coefficients.append(coefficient)
        return impulses, coefficients

    def output_approx(self, input_signal:ContinuousSignal, delta:float):
        INF = input_signal.INF
        t_values = np.array(
            [
                -input_signal.INF + i * delta
                for i in range(int((input_signal.INF - (-input_signal.INF)) / delta))
            ]
        )
        constituent_impulses = []
        coefficients = []
        output_signal = ContinuousSignal(lambda t: 0, input_signal.INF)
        for t in t_values:
            coefficient = input_signal.func(t) * delta
            coefficients.append(coefficient)
            h_t = self.impulse_response.shift(t)
            constituent_impulses.append(h_t)
            output_signal = output_signal.add(h_t.multiply_const_factor(coefficient))
        return output_signal, constituent_impulses, coefficients


def main():
    #discrete portion
    folder = "Discrete"
    if not os.path.exists(folder):
        os.makedirs(folder)

    INF = 5
    impulse_response = DiscreteSignal(INF)
    impulse_response.set_value_at_time(INF + 0, 1)
    impulse_response.set_value_at_time(INF + 1, 1)
    impulse_response.set_value_at_time(INF + 2, 1)
    lti = LTI_Discrete(impulse_response)

    input_signal = DiscreteSignal(INF)
    input_signal.set_value_at_time(INF + 0, 0.5)
    input_signal.set_value_at_time(INF + 1, 2)
    # input_signal.check_plot()
    #input_sction
    final_response = DiscreteSignal(INF)
    input_plot =[]
    unit_impulses, coefficients = lti.linear_combination_of_impulses(input_signal)
    for unit_impulse, coefficient in zip(unit_impulses, coefficients):
        input_plot.append(unit_impulse.multiply_const_factor(coefficient))
        final_response = final_response.add(
            unit_impulse.multiply_const_factor(coefficient)
        )
    subplot_titles = []
    for i in range(-INF, INF + 1):
        subplot_titles.append(f"δ[n - ({i})]x[{i}]")

    subplot_titles.append("Sum")
    input_plot.append(final_response)
    input_signal.plot(
        input_plot,
        "Figure: Returned impulses multiplied by respective coefficients",
        "Impulses multiplied by coefficients",
        subplot_titles,
        f"{folder}/input.png",
    )
    # Output Sectiomn
    output_plot = []
    output_signal, constituent_impulses, coefficients = lti.output(input_signal)

    for constituent_impulse, coefficient in zip(constituent_impulses, coefficients):
        output_plot.append(constituent_impulse.multiply_const_factor(coefficient))

    subplot_titles = []
    for k in range(-INF, INF + 1):
        subplot_titles.append(f"h[n - ({k})]x[{k}]")

    subplot_titles.append("Output = Sum")

    output_plot.append(output_signal)
    output_signal.plot(
        output_plot,
        "Figure: Output",
        "Response of Input Signal",
        subplot_titles,
        f"{folder}/output.png",
    )
# Continuous Portion


    folder = "Continuous"

    if not os.path.exists(folder):
        os.makedirs(folder)

    INF = 3
    delta = 0.5
    impulse_response = ContinuousSignal(
        lambda t: np.piecewise(t, [t < 0, t >= 0], [0, 1]), INF
    )

    input_signal = ContinuousSignal(
        lambda t: np.piecewise(t, [t < 0, t >= 0], [0, lambda t: np.exp(-t)]), INF
    )

    lti = LTI_Continuous(impulse_response)

    input_plot = []
    reconstructed_signal = ContinuousSignal(lambda t: 0, INF)
    impulses, coefficients = lti.linear_combination_of_impulses(input_signal, delta)

    for impulse, coefficient in zip(impulses, coefficients):
        reconstructed_signal = reconstructed_signal.add(
            impulse.multiply_const_factor(coefficient*delta)
        )
        input_plot.append(impulse.multiply_const_factor(coefficient*delta))

    subplotTitles = []
    for k in range(-2 * INF, 2 * INF + 1):
        subplotTitles.append(f"δ(t - ({k}∇))x({k}∇)∇")

    subplotTitles.append("Reconstructed Signal")

    input_plot.append(reconstructed_signal)
    input_signal.plot(
        input_plot,
        "Figure: Returned impulses multiplied by their coefficients",
        "Impulses multiplied by coefficients",
        subplotTitles,
        5,
        3,
        f"{folder}/input.png",
        -0.1,
        1.1
    )
    # --Reconstructed Signal with varying Delta--
    Deltas = [0.5, 0.1, 0.05, 0.01]
    reconstructed_signals = []
    for Delta in Deltas:
        reconstructed_signal = ContinuousSignal(lambda t: 0, INF)
        impulses, coefficients = lti.linear_combination_of_impulses(input_signal, Delta)
        for impulse, coefficient in zip(impulses, coefficients):
            reconstructed_signal = reconstructed_signal.add(
                impulse.multiply_const_factor(coefficient*Delta)
            )
            # input_plot.append(impulse.multiply_const_factor(coefficient*delta))
        reconstructed_signals.append(reconstructed_signal)

    subplotTitles = []
    for Delta in Deltas:
        subplotTitles.append(f"∇ = {Delta}")

    input_signal.plot(
        reconstructed_signals,
        "Figure: Reconstruction of input signal with varying delta",
        "",
        subplotTitles,
        2,
        2,
        f"{folder}/input_varying_delta.png",
        -0.1,
        1.1,
        0.2,
        True,
        "Reconstructed",
        "x(t)",
    )

    # --Output Portion--
    output_portion = []
    output_signal, constituent_impulses, coefficients = lti.output_approx(
        input_signal, delta
    )

    for constituent_impulse, coefficient in zip(constituent_impulses, coefficients):
        output_portion.append(constituent_impulse.multiply_const_factor(coefficient))

    subplotTitles = []
    for k in range(-2 * INF, 2 * INF + 1):
        subplotTitles.append(f"h(t - ({k}∇))x({k}∇)∇")

    subplotTitles.append("Output = Sum")

    output_portion.append(output_signal)
    output_signal.plot(
        output_portion,
        "Figure: Returned impulses multiplied by their coefficients",
        "Response of Impulse Signal",
        subplotTitles,
        5,
        3,
        f"{folder}/output.png",
        -0.1,
        1.3,
    )

    # --Output Signal with varying Delta--
    Deltas = [0.5, 0.1, 0.05, 0.01]
    reconstructed_signals = []
    for Delta in Deltas:
        output_signal, impulses, coefficients = lti.output_approx(input_signal, Delta)
        reconstructed_signals.append(output_signal)

    subplotTitles = []
    for Delta in Deltas:
        subplotTitles.append(f"∇ = {Delta}")

    output_signal_varying_delta = ContinuousSignal(
        lambda t: np.piecewise(t, [t < 0, t >= 0], [0, lambda t: 1 - np.exp(-t)]), INF
    )

    output_signal_varying_delta.plot(
        reconstructed_signals,
        "Figure: Approximate output signal with varying delta",
        "Approximate output as ∇ tends to 0",
        subplotTitles,
        2,
        2,
        f"{folder}/output_varying_delta.png",
        -0.1,
        1.3,
        0.2,
        True,
        "y_approx(t)",
        "y(t) = (1 - e^(-t))u(t)",
    )
if __name__ == "__main__":
    main()