import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


def plot_xy(func, x_min=0, x_max=100, num_points=500, title=None, xlabel='x', ylabel='y'):
    """
    Plots a function y = f(x) over a specified x range.
    
    Parameters:
        func (callable): A Python function that takes a NumPy array x and returns y.
        x_min (float): Minimum x value.
        x_max (float): Maximum x value.
        num_points (int): Number of points to plot.
        title (str): Optional title for the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    if not isinstance(func, (list, tuple)):
        func = [func]
    # Generate x range
    x = np.linspace(x_min, x_max, num_points)
    print("################################################")
    plt.figure(figsize=(8, 5))
    for i, f in enumerate(func):
        y = f(x)
        max_idx = np.argmax(y)
        print(f'Function {i+1}: Maximum y value: {y[max_idx]} at x = {x[max_idx]}')
        plt.plot(x, y, label=f'Function {i+1}')
    # Plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title if title else 'Function Plot')
    plt.grid(True)
    plt.legend()
    plt.show()

# Example usage:
# Define a function y = 0.4x / (13 + x)
if __name__ == "__main__":
    def my_func(x):
        return 0.4 * x / (13 + x)

    # Call the plot function
    plot_xy(my_func, x_min=0, x_max=50, title=r'A graph of a function!!!')


