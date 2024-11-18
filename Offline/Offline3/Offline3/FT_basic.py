# import numpy as np
# import matplotlib.pyplot as plt

# # Define the interval and function and generate appropriate x values and y values
# x_values = 
# y_values = 

# # Plot the original function
# plt.figure(figsize=(12, 4))
# plt.plot(x_values, y_values, label="Original y = x^2")
# plt.title("Original Function (y = x^2)")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show()


# # Define the sampled times and frequencies
# sampled_times = x_values
# frequencies = 

# # Fourier Transform 
# def fourier_transform(signal, frequencies, sampled_times):
#     num_freqs = len(frequencies)
#     ft_result_real = np.zeros(num_freqs)
#     ft_result_imag = np.zeros(num_freqs)
    
#     # Store the fourier transform results for each frequency. Handle the real and imaginary parts separately
#     # use trapezoidal integration to calculate the real and imaginary parts of the FT

#     return ft_result_real, ft_result_imag

# # Apply FT to the sampled data
# ft_data = fourier_transform(y_values, frequencies, sampled_times)
# #  plot the FT data
# plt.figure(figsize=(12, 6))
# plt.plot(frequencies, np.sqrt(ft_data[0]**2 + ft_data[1]**2))
# plt.title("Frequency Spectrum of y = x^2")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude")
# plt.show()


# # Inverse Fourier Transform 
# def inverse_fourier_transform(ft_signal, frequencies, sampled_times):
#     n = len(sampled_times)
#     reconstructed_signal = np.zeros(n)
#     # Reconstruct the signal by summing over all frequencies for each time in sampled_times.
#     # use trapezoidal integration to calculate the real part
#     # You have to return only the real part of the reconstructed signal
    
#     return reconstructed_signal

# # Reconstruct the signal from the FT data
# reconstructed_y_values = inverse_fourier_transform(ft_data, frequencies, sampled_times)
# # Plot the original and reconstructed functions for comparison
# plt.figure(figsize=(12, 4))
# plt.plot(x_values, y_values, label="Original y = x^2", color="blue")
# plt.plot(sampled_times, reconstructed_y_values, label="Reconstructed y = x^2", color="red", linestyle="--")
# plt.title("Original vs Reconstructed Function (y = x^2)")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Define the extended interval for plotting
plot_range = np.linspace(-10, 10, 1000)

# Define the interval for computation
x_values = np.linspace(-2, 2, 500)

# 1. Function Definitions
def parabolic_function(x):
    return np.where((x >= -2) & (x <= 2), x**2, 0)

def triangular_function(x):
    return np.where((x >= -2) & (x <= 2), 1 - np.abs(x / 2), 0)

def sawtooth_function(x):
    return np.where((x >= -2) & (x <= 2), x, 0)

def rectangular_function(x):
    return np.where((x >= -2) & (x <= 2), 1, 0)

# Select the function to test (change to triangular, sawtooth, or rectangular as needed)
y_values = parabolic_function(x_values)

# Plot the original function over the extended range
plt.figure(figsize=(8, 4))
plt.plot(plot_range, parabolic_function(plot_range), label="Original Function", color="blue")
plt.title("Original Function (y = x^2)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

# 2. Fourier Transform
def fourier_transform(signal, frequencies, sampled_times):
    num_freqs = len(frequencies)
    ft_real = np.zeros(num_freqs)
    ft_imag = np.zeros(num_freqs)
    
    for i, freq in enumerate(frequencies):
        cos_component = np.cos(2 * np.pi * freq * sampled_times)
        sin_component = np.sin(2 * np.pi * freq * sampled_times)
        ft_real[i] = np.trapz(signal * cos_component, sampled_times)
        ft_imag[i] = -np.trapz(signal * sin_component, sampled_times)
    
    return ft_real, ft_imag

# 3. Inverse Fourier Transform
def inverse_fourier_transform(ft_signal, frequencies, sampled_times, extended_times):
    n = len(extended_times)
    reconstructed_signal = np.zeros(n)
    
    for i, t in enumerate(extended_times):
        for j, freq in enumerate(frequencies):
            cos_component = np.cos(2 * np.pi * freq * t)
            sin_component = np.sin(2 * np.pi * freq * t)
            reconstructed_signal[i] += (
                ft_signal[0][j] * cos_component - ft_signal[1][j] * sin_component
            )
    
    # Normalize the signal by the length of the frequency range
    return reconstructed_signal / len(frequencies)

# 4. Experiment with Frequency Limits
frequency_ranges = [
    np.linspace(-1, 1, 100),
    np.linspace(-2, 2, 200),
    np.linspace(-5, 5, 500),
]

for freq_range in frequency_ranges:
    sampled_times = x_values
    ft_real, ft_imag = fourier_transform(y_values, freq_range, sampled_times)
    ft_magnitude = np.sqrt(ft_real**2 + ft_imag**2)

    # Plot the frequency spectrum
    plt.figure(figsize=(12, 6))
    plt.plot(freq_range, ft_magnitude)
    plt.title(f"Frequency Spectrum (Frequency Range: {freq_range[0]} to {freq_range[-1]})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.show()

    # Reconstruct the signal
    ft_data = (ft_real, ft_imag)
    reconstructed_y_values = inverse_fourier_transform(ft_data, freq_range, sampled_times, plot_range)

    # Plot original vs reconstructed functions over the extended range
    plt.figure(figsize=(8, 4))
    plt.plot(plot_range, parabolic_function(plot_range), label="Original Function", color="blue")
    plt.plot(plot_range, reconstructed_y_values, label="Reconstructed Function", color="red", linestyle="--")
    plt.title(f"Reconstructed Signal (Frequency Range: {freq_range[0]} to {freq_range[-1]})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()
