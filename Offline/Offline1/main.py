import numpy as np
import matplotlib.pyplot as plt
import DiscreteSignal as ds
import ContinuousSignal as cs
import DiscreteLTIS as dl
import ContinuousLTIS as cl
import os

def main():
    #discrete portion
    folder = "Discrete"
    if not os.path.exists(folder):
        os.makedirs(folder)

    INF = 5
    impulse_response = ds.DiscreteSignal(INF)
    impulse_response.set_value_at_time(INF + 0, 1)
    impulse_response.set_value_at_time(INF + 1, 1)
    impulse_response.set_value_at_time(INF + 2, 1)
    lti = dl.LTI_Discrete(impulse_response)

    input_signal = ds.DiscreteSignal(INF)
    input_signal.set_value_at_time(INF + 0, 0.5)
    input_signal.set_value_at_time(INF + 1, 2)
    input_signal.check_plot()
    #input_sction
    final_response = ds.DiscreteSignal(INF)
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
    impulse_response = cs.ContinuousSignal(
        lambda t: np.piecewise(t, [t < 0, t >= 0], [0, 1]), INF
    )

    input_signal = cs.ContinuousSignal(
        lambda t: np.piecewise(t, [t < 0, t >= 0], [0, lambda t: np.exp(-t)]), INF
    )

    lti = cl.LTI_Continuous(impulse_response)

    input_plot = []
    reconstructed_signal = cs.ContinuousSignal(lambda t: 0, INF)
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
        reconstructed_signal = cs.ContinuousSignal(lambda t: 0, INF)
        impulses, coefficients = lti.linear_combination_of_impulses(input_signal, Delta)
        for impulse, coefficient in zip(impulses, coefficients):
            reconstructed_signal = reconstructed_signal.add(
                impulse.multiply_const_factor(coefficient)
            )
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

    output_signal_varying_delta = cs.ContinuousSignal(
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