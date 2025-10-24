from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from test import plot_xy
from scipy.optimize import curve_fit


def fit_thrust_current(data):
    """
    Fit thrust = K * (I - a) to measured (I, thrust) data.
    Returns K, a.
    """
    data = np.array(data)
    I = data[:,0]
    T = data[:,1]

    # Linear regression: T = m*I + b
    A = np.vstack([I, np.ones(len(I))]).T
    m, b = np.linalg.lstsq(A, T, rcond=None)[0]

    K = m
    a = -b / m
    return K, a

def fit_thrust_current_linear(data):
    """
    Fit thrust = K * I to measured (I, thrust) data.
    Returns K.
    """
    data = np.array(data)
    I = data[:, 0]
    T = data[:, 1]

    # Least squares fit for T = K * I
    K = np.sum(I * T) / np.sum(I**2)
    return K

def plot_thrust_fit(data, K, a):
    """
    Plot measured thrust-current data and the fitted line y = K*(I - a).
    
    Parameters:
    - data: list of [I, T] pairs
    - K: fitted slope constant
    - a: fitted offset constant
    """
    data = np.array(data)
    I_data = data[:,0]
    T_data = data[:,1]

    # Create smooth current values for plotting the fitted line
    I_line = np.linspace(min(I_data)-0.1, max(I_data)+0.1, 200)
    T_line = K * (I_line - a)

    # Plot scatter of actual data
    plt.figure(figsize=(7,5))
    plt.scatter(I_data, T_data, color='red', label='Measured Data', zorder=3)

    # Plot fitted line
    plt.plot(I_line, T_line, color='blue', label=f'Fit: T = {K:.2f}(I - {a:.2f})')

    # Labels and grid
    plt.xlabel('Current (A)')
    plt.ylabel('Thrust (g)')
    plt.title('Thrust vs Current with Linear Fit')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_thrust_fit_with_percentage_error(data, K, a):
    """
    Plot thrust-current data with linear fit and percentage error bars.
    Each bar shows the percentage error with exact value on top.
    """
    data = np.array(data)
    I_data = data[:, 0]
    T_measured = data[:, 1]

    # Model prediction
    T_fit = K * (I_data - a)

    # Error calculation
    error_abs = T_measured - T_fit
    error_pct = (error_abs / T_measured) * 100

    # Smooth line for fit
    I_line = np.linspace(min(I_data)-0.1, max(I_data)+0.1, 200)
    T_line = K * (I_line - a)

    # --- Figure layout ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,8), sharex=True,
                                   gridspec_kw={'height_ratios':[3,1]})

    # Top plot: Thrust vs Current
    ax1.scatter(I_data, T_measured, color='red', label='Measured', zorder=3)
    ax1.plot(I_line, T_line, color='blue', label=f'Fit: T = {K:.2f}(I - {a:.2f})')
    ax1.set_ylabel('Thrust (g)')
    ax1.set_title('Thrust vs Current with Linear Fit & Percentage Error')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Bottom plot: % Error bars
    bar_width = 0.1
    bars = ax2.bar(I_data, error_pct, width=bar_width, color='gray', edgecolor='black')
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_xlabel('Current (A)')
    ax2.set_ylabel('Error (%)')
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Annotate bars with exact % error values
    for bar, err in zip(bars, error_pct):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height,
                 f"{err:.1f}%", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()

    # Print numeric table
    print("Index | Current(A) | Measured(g) | Fit(g) | Error(g) | Error(%)")
    print("-"*65)
    for i, (I, Tm, Tf, ea, ep) in enumerate(zip(I_data, T_measured, T_fit, error_abs, error_pct)):
        print(f"{i:5d} | {I:10.2f} | {Tm:11.2f} | {Tf:6.2f} | {ea:8.2f} | {ep:8.2f}%")

    return error_pct.tolist()

def plot_thrust_fit_thrust_per_current(data, K, a):
    """
    Plot thrust-current data with linear fit and percentage error bars.
    Also plot thrust/current (T/I) of the fitted line.
    """
    data = np.array(data)
    I_data = data[:, 0]
    T_measured = data[:, 1]

    # Model prediction
    T_fit = K * (I_data - a)

    # Error calculation
    error_abs = T_measured - T_fit
    error_pct = (error_abs / T_measured) * 100

    # Smooth line for fit
    I_line = np.linspace(min(I_data)-0.1, max(I_data)+0.1, 200)
    T_line = K * (I_line - a)

    # Thrust per current
    T_per_I = np.divide(T_line, I_line, out=np.zeros_like(T_line), where=I_line!=0)

    # --- Figure layout ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,8), sharex=True,
                                   gridspec_kw={'height_ratios':[3,1]})

    # Top plot: Thrust vs Current
    ax1.scatter(I_data, T_measured, color='red', label='Measured', zorder=3)
    ax1.plot(I_line, T_line, color='blue', label=f'Fit: T = {K:.2f}(I - {a:.2f})')
    ax1.set_ylabel('Thrust (g)')
    ax1.set_title('Thrust vs Current with Linear Fit & Percentage Error')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left')

    # Add T/I line on secondary y-axis
    ax1b = ax1.twinx()
    ax1b.plot(I_line, T_per_I, color='green', linestyle='--', label='Thrust/Current (T/I)')
    ax1b.set_ylabel('Thrust per Current (g/A)', color='green')
    ax1b.tick_params(axis='y', labelcolor='green')
    ax1b.legend(loc='upper right')

    # Bottom plot: Percentage error
    ax2.bar(I_data, error_pct, color='gray', alpha=0.7)
    for i, val in enumerate(error_pct):
        ax2.text(I_data[i], val, f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    ax2.set_ylabel('Error (%)')
    ax2.set_xlabel('Current (A)')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

def fit_to_cubic(data):


    # Separate into x and y
    x = np.array([d[0] for d in data_t_to_e_2])
    y = np.array([d[1] for d in data_t_to_e_2])

    # Fit a cubic polynomial (degree = 3)
    coeffs = np.polyfit(x, y, 3)

    # Create a polynomial function from coefficients
    poly = np.poly1d(coeffs)

    # Generate smooth curve for plotting
    x_fit = np.linspace(min(x), max(x), 200)
    y_fit = poly(x_fit)

    # Print the equation
    print("Cubic fit equation:")
    print(f"y = {coeffs[0]:.6f}x³ + {coeffs[1]:.6f}x² + {coeffs[2]:.6f}x + {coeffs[3]:.6f}")

    # Plot data and fitted curve
    plt.scatter(x, y, color='red', label='Data points')
    plt.plot(x_fit, y_fit, color='blue', label='Cubic fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Cubic Fit to Data')
    plt.grid(True)
    plt.show()
    return coeffs


def fit_to_parabola(data):


    # Separate into x and y
    x = np.array([d[0] for d in data_t_to_e_2])
    y = np.array([d[1] for d in data_t_to_e_2])

    # Fit a cubic polynomial (degree = 2)
    coeffs = np.polyfit(x, y, 2)

    # Create a polynomial function from coefficients
    poly = np.poly1d(coeffs)

    # Generate smooth curve for plotting
    x_fit = np.linspace(min(x), max(x), 200)
    y_fit = poly(x_fit)

    # Print the equation
    print("parabolic fit equation:")
    print(f"y = {coeffs[0]:.6f}x^2 + {coeffs[1]:.6f}x + {coeffs[2]:.6f} ")

    # Plot data and fitted curve
    plt.scatter(x, y, color='red', label='Data points')
    plt.plot(x_fit, y_fit, color='blue', label='parabola fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('parabolic Fit to Data')
    plt.grid(True)
    plt.show()
    return coeffs

def fit_concave_parabola(data, title='Downward-opening Parabolic Fit'):
    """
    Fits (x, y) data to a concave parabola y = a*x² + b*x + c, 
    enforcing a < 0 via a = -exp(A).

    Parameters:
        data (array-like): Nx2 array with columns [x, y].
        title (str): Title for the plot.

    Returns:
        list: [a, b, c] coefficients of the fitted parabola (a < 0).
    """
    # Ensure numpy array
    data = np.array(data)
    x = data[:, 0]
    y = data[:, 1]

    # Define concave parabola model (forces a < 0)
    def concave_parabola(x, A, b, c):
        a = -np.exp(A)
        return a * x**2 + b * x + c

    # Fit curve (initial guesses: A=0 → a=-1, b=0, c=mean(y))
    popt, pcov = curve_fit(concave_parabola, x, y, p0=[0.0, 0.0, np.mean(y)])
    A_opt, b_opt, c_opt = popt
    a_opt = -np.exp(A_opt)

    print(f"Fitted coefficients: a = {a_opt}, b = {b_opt:.6f}, c = {c_opt:.6f}")
    # Compute fitted curve for plotting
    x_fit = np.linspace(x.min(), x.max(), 400)
    y_fit = concave_parabola(x_fit, *popt)

    # Equation text
    eq_text = f"$y = {a_opt:.4f}x^2 + {b_opt:.4f}x + {c_opt:.4f}$"

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='red', label='Data points')
    plt.plot(x_fit, y_fit, color='blue', linewidth=2, label='Concave fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.text(0.05, 0.95, eq_text,
             transform=plt.gca().transAxes,
             fontsize=11, color='blue',
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.grid(True)
    plt.show()

    return [a_opt, b_opt, c_opt]



def fit_to_lin(data):
    # Separate into x and y
    x = np.array([d[0] for d in data_t_to_e_2])
    y = np.array([d[1] for d in data_t_to_e_2])

    # Fit a cubic polynomial (degree = 2)
    coeffs = np.polyfit(x, y, 1)

    # Create a polynomial function from coefficients
    poly = np.poly1d(coeffs)

    # Generate smooth curve for plotting
    x_fit = np.linspace(min(x), max(x), 200)
    y_fit = poly(x_fit)

    # Print the equation
    print("parabolic fit equation:")
    print(f"y = {coeffs[0]:.6f}x + {coeffs[1]:.6f}")

    # Plot data and fitted curve
    plt.scatter(x, y, color='red', label='Data points')
    plt.plot(x_fit, y_fit, color='blue', label='parabola fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('parabolic Fit to Data')
    plt.grid(True)
    plt.show()
    return coeffs


# Example dataset
data1 = [
    [0.52, 5.1],
    [1.00, 8.9],
    [1.53, 13.1],
    [1.99, 16.3],
    [2.50, 19.3],
    [2.86, 21.7]
]



data2 = [[1, 8.1],[1.5, 11.8],[2,15.6],[2.5,19.6],[3,22.4],[3.5, 25.6],[4, 27.1]]

# === 0802SE 19500KV ===

# GF31mm -3B 1219
data_GF31_19500 = [
    [0.5, 2.8],
    [1.0, 7.2],
    [1.5, 11.6],
    [2.0, 16.0],
    [2.5, 20.7],
    [3.0, 24.2],
    [3.1, 24.6]
]

# GF35mm -3B
data_GF35_19500 = [
    [0.5, 4.1],
    [1.0, 8.8],
    [1.5, 14.0],
    [2.0, 18.5],
    [2.5, 23.0],
    [3.0, 26.3],
    [3.4, 30.6]

]

# GF40mm -2B 1610
data_GF40_19500 = [
    [0.5,4.4],
    [1.0, 9.4],
    [1.5, 14.5],
    [2.0, 20.0],
    [2.5, 25.3],
    [3.0, 29.0],
    [3.5, 32.5],
    [3.7, 33.5]
]

# === 0802SE 23000KV ===

# GF31mm -3B 1219
data_GF31_23000 = [
    [0.5, 3.0],
    [1.0, 7.2],
    [1.5, 11.3],
    [2.0, 17.2],
    [2.5, 21.8],
    [3.0, 25.4],
    [3.3, 27.4]
]

# GF35mm -3B    from scipy.optimize import minimize

data_GF35_23000 = [
    [0.5, 3.9],
    [1.0, 8.6],
    [1.5, 12.9],
    [2.0, 18.4],
    [2.5, 24.2],
    [3.0, 27.3],
    [3.5, 30.6],
    [3.7, 32.6]
]

# GF40mm -2B 1610
data_GF40_23000 = [
    [0.5, 4.8],
    [1.0, 9.9],
    [1.5, 14.2],
    [2.0, 18.8],
    [2.5, 23.4],
    [3.0, 28.5],
    [3.5, 32.3],
    [4.0, 35.3]
]


data3 = [[0.5, 3.24],[1.0,6.57],[1.5,8.922],[2.0, 11.13],[2.5, 13.03],[3, 15.87],[3.5, 18.89],[4, 23.36],[4.5, 26.49],[5, 29.8],[5.5, 32.59],[6, 36.13],[6.5, 37.24],[7,40.02],[7.5, 41.23], [7.8, 44.02]]

data4 = [[0.5,3.655],[1,7.004],[1.5,10.71],[2,14.07]]

data5 = [[0.5,4.57],[1,8.755],[1.5, 13.39],[2,17.6],]

data6 = [[1.05, 11.9],[2.08, 20.5],[3.05, 27.1],[4.11, 32.3], [4.25, 33.9]]
data7=[[0.5, 4.4],[1,9.4],[1.5,14.5],[2,20],[2.5,25.3],[3,29],[3.5,32.5],[3.7,33.5]]

# Print one example to verify
datas = [data1, data2, data_GF31_19500, data_GF35_19500, data_GF40_19500,
         data_GF31_23000, data_GF35_23000, data_GF40_23000]

# for data in datas:
#     K, a = fit_thrust_current_linear(data), 0
#     print(f"Best fit: K = {K:.3f}, a = {a:.3f}")
#     plot_thrust_fit_with_percentage_error(data, K, a)
#     plot_thrust_fit_thrust_per_current(data, K, a)

data_t_to_e_1 = [[11.9,2.982],[20.5, 2.594],[27.1, 2.338],[32.3, 2.068],[33.9, 2.156],]
data_t_to_e_2 = [[7.6, 2.866],[13.2,2.551],[18.9,2.431,],[23.4, 2.261],[25.2,2.14],]
data_t_to_e_3 = [[5.1,3.55],[10.3, 3.58],[13.8, 3.3,],[17.9, 3.24],[22.7, 3.27, ],[27.7, 3.31],[32,3.29],[35.7,3.23],[39, 3.17],[40.6, 3.11],]
data_t_to_e_4 = [[23,2.905],[40.9, 2.607],[56.6, 2.413],[68.6, 2.289],[81.8,2.193],[95,2.074],[106.5,1.991],[115.3,1.884],[121.9, 1.791],] # happymodel ex1103 https://www.happymodel.cn/index.php/2022/09/05/bassline-spare-part-ex1103-kv11000-brushless-motor/
data=data7

data_cubic = data_t_to_e_4

cubic = fit_concave_parabola(data_cubic)
cubic_func = lambda x: cubic[0]*x**3 + cubic[1]*x**2 + cubic[2]*x + cubic[3]
parabolic_func = lambda x: cubic[0]*x**2 + cubic[1]*x + cubic[2]
lin_func = lambda x: cubic[0]*x + cubic[1]

def time_func(x):
    return 0.2 * x * parabolic_func((15+x)/4) / (15 + x) 

def battery_thrust_func(watt_per_gram, x): # Don't have to multiply 4, even with 4 motors.
    return watt_per_gram * x * parabolic_func((15+x)/4) 

def weight_func(x):
    return (15+x)

def parabolic(x):
    return parabolic_func((15+x)/4)

plot_xy([time_func], x_min=0, x_max=400)

for i in np.arange(0.5, 1, 0.05):
    print(f'=== Watt per gram: {i} ===')
    plot_xy([partial(battery_thrust_func, i), weight_func], x_min=0, x_max=400)

    print('---------------------------------------------------------------------------------')
# plot_xy([parabolic], x_min=0, x_max=200, title='Thrust per motor vs Battery Current', ylabel='Thrust per motor (g)')