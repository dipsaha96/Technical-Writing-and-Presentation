import numpy as np
import matplotlib.pyplot as plt

# Define the extended interval for plotting
plot_range = np.linspace(-10, 10, 1000)

# Define the interval for computation
x_values = np.linspace(-2, 2, 500)

# Function Definitions
def parabolic_function(x):
    return np.where((x >= -2) & (x <= 2), x**2, 0)

def triangular_function(x):
    return np.where((x >= -2) & (x <= 2), 1 - np.abs(x / 2), 0)

def sawtooth_function(x):
    return np.where((x >= -2) & (x <= 2), x, 0)

def rectangular_function(x):
    return np.where((x >= -2) & (x <= 2), 1, 0)

# Fourier Transform
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

# Inverse Fourier Transform
def inverse_fourier_transform(ft_signal, frequencies, sampled_times, extended_times):
    reconstructed_signal = np.zeros(len(extended_times))
    
    for i, t in enumerate(extended_times):
        for j, freq in enumerate(frequencies):
            cos_component = np.cos(2 * np.pi * freq * t)
            sin_component = np.sin(2 * np.pi * freq * t)
            reconstructed_signal[i] += (
                ft_signal[0][j] * cos_component - ft_signal[1][j] * sin_component
            )
    
    return reconstructed_signal

# Function to plot and process a signal
def process_function(func, function_name):
    y_values = func(x_values)

    # Plot the original function over the extended range
    plt.figure(figsize=(8, 4))
    plt.plot(plot_range, func(plot_range), label="Original Function", color="blue")
    plt.title(f"Original Function ({function_name})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

    # Frequency ranges to experiment with
    frequency_ranges = [
        np.linspace(-1, 1, 100),
        np.linspace(-2, 2, 200),
        np.linspace(-5, 5, 500),
    ]

    for freq_range in frequency_ranges:
        # Fourier Transform
        ft_real, ft_imag = fourier_transform(y_values, freq_range, x_values)
        ft_magnitude = np.sqrt(ft_real**2 + ft_imag**2)

        # Plot the frequency spectrum
        plt.figure(figsize=(8, 4))
        plt.plot(freq_range, ft_magnitude)
        plt.title(f"Frequency Spectrum ({function_name}) - Range: {freq_range[0]} to {freq_range[-1]}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid()
        plt.show()

        # Reconstruct the signal
        ft_data = (ft_real, ft_imag)
        reconstructed_y_values = inverse_fourier_transform(ft_data, freq_range, x_values, plot_range)

        # Scale reconstructed signal to match original amplitude
        reconstructed_y_values *= (x_values[-1] - x_values[0]) / len(freq_range)

        # Plot original vs reconstructed functions
        plt.figure(figsize=(8, 4))
        plt.plot(plot_range, func(plot_range), label="Original Function", color="blue")
        plt.plot(plot_range, reconstructed_y_values, label="Reconstructed Function", color="red", linestyle="--")
        plt.title(f"Reconstructed Signal ({function_name}) - Range: {freq_range[0]} to {freq_range[-1]}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid()
        plt.show()

# Main function for user interaction
def main():
    print("Choose a function to process:")
    print("1. Parabolic Function (y = x^2)")
    print("2. Triangular Function")
    print("3. Sawtooth Function")
    print("4. Rectangular Function")
    
    choice = int(input("Enter your choice (1-4): "))
    
    if choice == 1:
        process_function(parabolic_function, "y = x^2")
    elif choice == 2:
        process_function(triangular_function, "Triangular Function")
    elif choice == 3:
        process_function(sawtooth_function, "Sawtooth Function")
    elif choice == 4:
        process_function(rectangular_function, "Rectangular Function")
    else:
        print("Invalid choice. Please run the program again and choose a valid option.")

# Run the main function
if __name__ == "__main__":
    main()
