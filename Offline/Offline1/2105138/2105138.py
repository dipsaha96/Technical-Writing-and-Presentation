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

    def plot(
    self,
    DiscreteSignals: list["DiscreteSignal"],
    title: str,
    supTitle: str,
    subplotTitles: list[str],
    saveTo: str,
    ):
         # Get the last element as final response
        final_response = DiscreteSignals[-1] 
        # All other signals before the last one
        intermediate_signals = DiscreteSignals[:-1]  

        # Create a figure with subplots (4 rows, 3 columns)
        fig, axs = plt.subplots(4, 3, figsize=(10, 10))
        y_range = (-1, max(np.max(self.values), 3) + 1)

        # Set the overall title for the figure
        fig.suptitle(supTitle, fontsize=16)

        # Define time range for x-axis
        time_range = np.arange(-self.INF, self.INF + 1)

        # Plot each intermediate signal
        for idx, (DiscreteSignal, subplotTitle) in enumerate(zip(intermediate_signals, subplotTitles)):
            row, col = divmod(idx, 3)  # Determine row and column for subplot
            ax = axs[row, col]
            ax.stem(time_range, DiscreteSignal.values, basefmt="r-")
            ax.set_xticks(time_range)
            ax.set_ylim(*y_range)
            ax.set_title(subplotTitle)
            ax.set_xlabel("n (Time Index)")
            ax.set_ylabel("x[n]")
            ax.grid(True)

        # Plot the final response in the next subplot position
        final_row, final_col = divmod(len(intermediate_signals), 3)
        ax = axs[final_row, final_col]
        ax.stem(time_range, final_response.values, basefmt="r-")
        ax.set_xticks(time_range)
        ax.set_ylim(*y_range)
        ax.set_title(subplotTitles[-1])
        ax.set_xlabel("n (Time Index)")
        ax.set_ylabel("x[n]")
        ax.grid(True)

        # Hide any unused subplots
        for idx in range(len(intermediate_signals) + 1, 12):
            row, col = divmod(idx, 3)
            fig.delaxes(axs[row, col])

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

        # Save the figure to the specified path
        plt.savefig(saveTo)
        plt.close(fig)



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


    def plot(
    self,
    continuousSignals: list["ContinuousSignal"],
    title: str,
    subTitle: str,
    subplotTitles: list[str],
    rows: int,
    columns: int,
    saveTo: str,
    minheight: float = 0,
    maxheight: float = 1,
    y_tick_spacing: float = 0.5,
    samePlot: bool = False,
    label1: str = "",
    label2: str = "",
    ):
        t = np.linspace(-self.INF, self.INF + 0.01, 1000)
        # Get the last element as the reconstructed signal
        reconstructed_signal = continuousSignals[-1] 
        # All other signals before the last one 
        intermediate_signals = continuousSignals[:-1]  
        # Create a figure with subplots (rows, columns)
        fig, axs = plt.subplots(rows, columns, figsize=(10, 10))
        # Set the overall title for the figure
        fig.suptitle(subTitle, fontsize=16)

        # Plot each intermediate signal
        for idx, (continuousSignal, subplotTitle) in enumerate(zip(intermediate_signals, subplotTitles)):
            row, col = divmod(idx, columns)
            ax = axs[row, col]

            # Set ticks and labels
            ax.set_xticks(np.arange(-self.INF, self.INF + 1, 1))
            ax.set_yticks(np.arange(0, maxheight + y_tick_spacing, y_tick_spacing))
            
            # Plot signal with optional comparison
            if samePlot:
                ax.plot(t, continuousSignal.func(t), label=label1)
                ax.plot(t, self.func(t), color="red", label=label2)
            else:
                ax.plot(t, continuousSignal.func(t))
                ax.set_xlim([-3, 3])

            
            ax.set_ylim([minheight, maxheight])
            ax.set_title(subplotTitle)
            ax.set_xlabel("t (Time)")
            ax.set_ylabel("x[t]")
            
            if samePlot:
                ax.legend()
            
            ax.grid(True)

        # Plot the reconstructed signal in the next available subplot
        final_row, final_col = divmod(len(intermediate_signals), columns)
        ax = axs[final_row, final_col]
        ax.set_xticks(np.arange(-self.INF, self.INF + 1, 1))
        ax.set_yticks(np.arange(0, maxheight + y_tick_spacing, y_tick_spacing))
        
        # Plot reconstructed signal with optional comparison
        if samePlot:
            ax.plot(t, reconstructed_signal.func(t), label=label1)
            ax.plot(t, self.func(t), color="red", label=label2)
        else:
            ax.plot(t, reconstructed_signal.func(t))
            ax.set_xlim([-3, 3])
        
        ax.set_ylim([minheight, maxheight])
        ax.set_title(subplotTitles[-1])
        ax.set_xlabel("t (Time)")
        ax.set_ylabel("x[t]")
        
        if samePlot:
            ax.legend()
        
        ax.grid(True)

        # Hide any unused subplots
        for idx in range(len(intermediate_signals) + 1, rows * columns):
            row, col = divmod(idx, columns)
            fig.delaxes(axs[row, col])

        # Adjust layout to prevent overlapping of plots
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.1)

        # Add a caption below the figure
        fig.text(0.5, 0.01, title, ha="center", fontsize=12)

        # Save the figure to the specified path
        plt.savefig(saveTo)
        plt.close(fig)


class LTI_Discrete:
    def __init__(self, impulse_response: DiscreteSignal):
        self.impulse_response = impulse_response

    def linear_combination_of_impulses(self, input_signal: DiscreteSignal):
        INF = input_signal.INF
        impulses, coeffs = [], []
        
        # Iterate through each time shift to extract coefficients and create unit impulses
        for shift in range(-INF, INF + 1):
            coeffs.append(input_signal.values[INF + shift])
            
            # Create a unit impulse shifted appropriately
            unit_impulse = DiscreteSignal(INF)
            unit_impulse.set_value_at_time(INF, 1)
            shifted_impulse = unit_impulse.shift_signal(shift)
            
            # Collect the shifted impulse
            impulses.append(shifted_impulse)

        return impulses, coeffs

    def output(self, input_signal: DiscreteSignal):
        INF = input_signal.INF
        output_signal = DiscreteSignal(INF)
        coeffs, impulse_responses = [], []

        # Construct the output by iterating over each time shift and scaling impulse responses
        for shift in range(-INF, INF + 1):
            coeff = input_signal.values[INF + shift]
            shifted_response = self.impulse_response.shift_signal(shift)
            
            # Store the coefficients and shifted responses
            coeffs.append(coeff)
            impulse_responses.append(shifted_response)
            
            # Scale the shifted response and add it to the output signal
            scaled_response = shifted_response.multiply_const_factor(coeff)
            output_signal = output_signal.add(scaled_response)

        return output_signal, impulse_responses, coeffs

    
class LTI_Continuous:
    def __init__(self, impulse_response: ContinuousSignal):
        self.impulse_response = impulse_response

    def linear_combination_of_impulses(self, input_signal: ContinuousSignal, delta: float):
        INF = input_signal.INF
        # Generate time values using delta increments
        # t_values = np.arange(-INF, INF, delta)
        num_points = int((2 * INF) / delta)  # Number of points from -INF to INF
        t_values = np.linspace(-INF, INF, num_points, endpoint=False)
        impulses, coefficients = [], []

        # Iterate over time values to create scaled unit impulses
        for t in t_values:
            # Define the unit impulse function over a small time range [t, t + delta]
            impulse = ContinuousSignal(
                lambda tau, t=t: 1/delta * ((tau >= t) & (tau <= t + delta)), INF
            )
            # Calculate the coefficient as the input signal value at time t
            coeff = input_signal.func(t)
            
            # Store the unit impulse and its coefficient
            impulses.append(impulse)
            coefficients.append(coeff)
        
        return impulses, coefficients

    def output_approx(self, input_signal: ContinuousSignal, delta: float):
        INF = input_signal.INF
        # Generate time values using delta increments
        # t_values = np.arange(-INF, INF, delta)
        num_points = int((2 * INF) / delta)  # Number of points from -INF to INF
        t_values = np.linspace(-INF, INF, num_points, endpoint=False)
        impulses, coefficients = [], []
        output_signal = ContinuousSignal(lambda t: 0, INF)

        # Iterate over time values to approximate the output signal
        for t in t_values:
            # Calculate the coefficient by multiplying input signal at t with delta
            coeff = input_signal.func(t) * delta
            coefficients.append(coeff)
            
            # Shift the impulse response by t and scale it by the coefficient
            shifted_response = self.impulse_response.shift(t)
            impulses.append(shifted_response)
            
            # Add the scaled shifted response to the output signal
            scaled_response = shifted_response.multiply_const_factor(coeff)
            output_signal = output_signal.add(scaled_response)
        
        return output_signal, impulses, coefficients


def create_output_directory(folder: str):
    # Create output directory if it doesn't exist.
    if not os.path.exists(folder):
        os.makedirs(folder)


def run_discrete_simulation(INF, impulse_response_values, input_signal_values, folder):
    # Run the discrete signal simulation and generate plots.
    impulse_response = DiscreteSignal(INF)
    set_signal_values(impulse_response, impulse_response_values)
    input_signal = DiscreteSignal(INF)
    set_signal_values(input_signal, input_signal_values)

    lti = LTI_Discrete(impulse_response)

    # Input Signal Decomposition and Plotting
    input_plot, final_response = decompose_and_plot_input_discrete(
        lti, input_signal, INF, folder
    )

    # Output Signal Computation and Plotting
    output_signal, output_plot = compute_and_plot_output_discrete(
        lti, input_signal, INF, folder
    )


def set_signal_values(signal, values_dict):
    # Set values at specified time indices for a signal.
    for k, v in values_dict.items():
        signal.set_value_at_time(signal.INF + k, v)


def decompose_and_plot_input_discrete(lti, input_signal, INF, folder):
    # Decompose the input signal and create corresponding plots.
    input_plot = []
    unit_impulses, coefficients = lti.linear_combination_of_impulses(input_signal)
    final_response = DiscreteSignal(INF)

    for unit_impulse, coefficient in zip(unit_impulses, coefficients):
        scaled_impulse = unit_impulse.multiply_const_factor(coefficient)
        input_plot.append(scaled_impulse)
        final_response = final_response.add(scaled_impulse)

    subplot_titles = [f"δ[n - ({i})]x[{i}]" for i in range(-INF, INF + 1)] + ["Sum"]
    input_plot.append(final_response)

    input_signal.plot(
        input_plot,
        "Figure: Returned impulses multiplied by respective coefficients",
        "Impulses multiplied by coefficients",
        subplot_titles,
        f"{folder}/input.png",
    )

    return input_plot, final_response


def compute_and_plot_output_discrete(lti, input_signal, INF, folder):
    # Compute the output signal using convolution and create plots.
    output_signal, constituent_impulses, coefficients = lti.output(input_signal)
    output_plot = []

    for impulse, coefficient in zip(constituent_impulses, coefficients):
        output_plot.append(impulse.multiply_const_factor(coefficient))

    subplot_titles = [f"h[n - ({k})]x[{k}]" for k in range(-INF, INF + 1)] + ["Output = Sum"]
    output_plot.append(output_signal)

    output_signal.plot(
        output_plot,
        "Figure: Output",
        "Response of Input Signal",
        subplot_titles,
        f"{folder}/output.png",
    )

    return output_signal, output_plot


def run_continuous_simulation(INF, delta, impulse_response_func, input_signal_func, folder, varying_deltas):
    # Run the continuous signal simulation and generate plots.
    impulse_response = ContinuousSignal(impulse_response_func, INF)
    input_signal = ContinuousSignal(input_signal_func, INF)

    lti = LTI_Continuous(impulse_response)

    # Input Signal Decomposition and Plotting
    input_plot, reconstructed_signal = decompose_and_plot_input_continuous(
        lti, input_signal, INF, delta, folder
    )

    # Reconstructed Signals with Varying Delta
    plot_varying_delta_reconstruction(lti, input_signal, INF, varying_deltas, folder)

    # Output Signal Computation and Plotting
    output_signal = compute_and_plot_output_continuous(
        lti, input_signal, INF, delta, folder
    )

    # Output Signal with Varying Delta
    plot_varying_delta_output(lti, input_signal, INF, varying_deltas, folder)


def decompose_and_plot_input_continuous(lti, input_signal, INF, delta, folder):
    # Decompose the input signal into impulses and plot.
    input_plot = []
    impulses, coefficients = lti.linear_combination_of_impulses(input_signal, delta)
    reconstructed_signal = ContinuousSignal(lambda t: 0, INF)

    for impulse, coefficient in zip(impulses, coefficients):
        scaled_impulse = impulse.multiply_const_factor(coefficient * delta)
        input_plot.append(scaled_impulse)
        reconstructed_signal = reconstructed_signal.add(scaled_impulse)

    subplot_titles = [f"δ(t - ({k}∇))x({k}∇)∇" for k in range(-2 * INF, 2 * INF + 1)] + ["Reconstructed Signal"]
    input_plot.append(reconstructed_signal)

    input_signal.plot(
        input_plot,
        "Figure: Returned impulses multiplied by their coefficients",
        "Impulses multiplied by coefficients",
        subplot_titles,
        5,
        3,
        f"{folder}/input.png",
        -0.1,
        1.1,
    )

    return input_plot, reconstructed_signal


def plot_varying_delta_reconstruction(lti, input_signal, INF, Deltas, folder):
    # Plot reconstructed signals with varying delta values.
    reconstructed_signals = []

    for delta in Deltas:
        impulses, coefficients = lti.linear_combination_of_impulses(input_signal, delta)
        reconstructed_signal = ContinuousSignal(lambda t: 0, INF)  # Initialize an empty signal for sum
        for impulse, coefficient in zip(impulses, coefficients):
            scaled_impulse = impulse.multiply_const_factor(coefficient * delta)
            reconstructed_signal = reconstructed_signal.add(scaled_impulse)
        
        reconstructed_signals.append(reconstructed_signal)

    subplot_titles = [f"∇ = {delta}" for delta in Deltas]

    input_signal.plot(
        reconstructed_signals,
        "Figure: Reconstruction of input signal with varying delta",
        "",
        subplot_titles,
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



def compute_and_plot_output_continuous(lti, input_signal, INF, delta, folder):
    # Compute and plot the output signal.
    output_signal, constituent_impulses, coefficients = lti.output_approx(input_signal, delta)
    output_plot = [
        impulse.multiply_const_factor(coefficient)
        for impulse, coefficient in zip(constituent_impulses, coefficients)
    ]

    subplot_titles = [f"h(t - ({k}∇))x({k}∇)∇" for k in range(-2 * INF, 2 * INF + 1)] + ["Output = Sum"]
    output_plot.append(output_signal)

    output_signal.plot(
        output_plot,
        "Figure: Returned impulses multiplied by their coefficients",
        "Response of Impulse Signal",
        subplot_titles,
        5,
        3,
        f"{folder}/output.png",
        -0.1,
        1.3,
    )

    return output_signal


def plot_varying_delta_output(lti, input_signal, INF, Deltas, folder):
    # Plot output signals with varying delta values.
    reconstructed_signals = [
        lti.output_approx(input_signal, delta)[0] for delta in Deltas
    ]

    subplot_titles = [f"∇ = {delta}" for delta in Deltas]

    output_signal_varying_delta = ContinuousSignal(
        lambda t: np.piecewise(t, [t < 0, t >= 0], [0, lambda t: 1 - np.exp(-t)]), INF
    )

    output_signal_varying_delta.plot(
        reconstructed_signals,
        "Figure: Approximate output signal with varying delta",
        "Approximate output as ∇ tends to 0",
        subplot_titles,
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
def main():
    # Create directories for saving outputs
        create_output_directory("Discrete")
        create_output_directory("Continuous")

        # Discrete Part
        run_discrete_simulation(
            INF=5,
            impulse_response_values={0: 1, 1: 1, 2: 1},
            input_signal_values={0: 0.5, 1: 2},
            folder="Discrete",
        )

        # Continuous Part
        run_continuous_simulation(
            INF=3,
            delta=0.5,
            impulse_response_func=lambda t: np.piecewise(t, [t < 0, t >= 0], [0, 1]),
            input_signal_func=lambda t: np.piecewise(t, [t < 0, t >= 0], [0, lambda t: np.exp(-t)]),
            folder="Continuous",
            varying_deltas=[0.5, 0.1, 0.05, 0.01],
        )
if __name__ == "__main__":
    main()