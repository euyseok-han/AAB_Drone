import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

def plot_xy(func, x_min=0, x_max=100, num_points=500, title=None, xlabel='x', ylabel='y', mark_x=None):
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
    mark_x (float): Optional x value to mark on the graph with its corresponding y value.
    """
    if not isinstance(func, (list, tuple)):
        func = [func]
    
    # Generate x range
    x = np.linspace(x_min, x_max, num_points)
    print("################################################")
    plt.figure(figsize=(8, 5))
    
    y_max = 0
    y_min = 10**10
    for i, f in enumerate(func):
        y = f(x)
        max_idx = np.argmax(y)
        print(f'Function {i+1}: Maximum y value: {y[max_idx]} at x = {x[max_idx]}')
        plt.plot(x, y, label=f'Function {i+1}')
        
        # Mark specific x value if provided
        if mark_x is not None:
            # Find the closest x value in our array
            closest_idx = np.argmin(np.abs(x - mark_x))
            mark_y = y[closest_idx]
            actual_x = x[closest_idx]
            y_max = max(y_max, mark_y)
            y_min = min(y_min, mark_y)
            # Plot the point
            plt.plot(actual_x, mark_y, 'ro', markersize=8, label=f'x={actual_x:.2f}')
            
            # Add annotation
            plt.annotate(f'({actual_x:.2f}, {mark_y:.2f})',
                        xy=(actual_x, mark_y),
                        xytext=(10, 10),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            print(f'Function {i+1}: At x = {actual_x:.2f}, y = {mark_y:.2f}')

    if y_max:
        print(f'max_thrust is {y_max/y_min:.2f}times of the drone weight')
    # Plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title if title else 'Function Plot')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return x[max_idx], y[max_idx]

# Example usage:
# Define a function y = 0.4x / (13 + x)
if __name__ == "__main__":
    def my_func(x):
        return 0.4 * x / (13 + x)

    # Call the plot function
    plot_xy(my_func, x_min=0, x_max=50, title=r'A graph of a function!!!')


