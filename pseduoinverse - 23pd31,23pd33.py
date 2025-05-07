import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import json
import time
import os

def generate_random_data(num_points):
    np.random.seed(int(time.time()))
    X = np.random.rand(num_points, 1) * 10
    for i in range(num_points):
        func_type = np.random.choice([1, 2])

        if func_type == 1:  
            slope = np.random.uniform(0.5, 2)
            intercept = np.random.uniform(0, 3)
            noise = np.random.normal(0, 5, size=(num_points, 1))  
            Y = slope * X + intercept + noise
        elif func_type == 2:
            noise = np.random.normal(0, 5, size=(num_points, 1))  
            Y = 2.5 * X + 3 + noise
    return X, Y

def custom_mean_squared_error(Y_true, Y_pred):
    return np.mean((Y_true - Y_pred) ** 2)

def calculate_r_squared(Y_true, Y_pred):
    ss_res = np.sum((Y_true - Y_pred) ** 2)
    ss_tot = np.sum((Y_true - np.mean(Y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def moore_penrose_pseudoinverse(A):
    At = A.T
    AtA = np.dot(At, A)

    try:
        AtA_inv = np.linalg.inv(AtA)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix is singular and cannot be inverted.")
    
    return np.dot(AtA_inv, At)

def fit_linear_regression(X, Y):
    A = np.hstack([np.ones((X.shape[0], 1)), X])  
    A_plus = moore_penrose_pseudoinverse(A)
    coefficients = np.dot(A_plus, Y)
    Y_pred = np.dot(A, coefficients)
    
    error = custom_mean_squared_error(Y, Y_pred)
    r_squared = calculate_r_squared(Y, Y_pred)

    return coefficients, error, r_squared

def plot_data_in_new_window(X, Y, coefficients):
    new_window = tk.Toplevel()
    new_window.title("Fitted Model Plot")
    figure = plt.Figure(figsize=(6, 4), dpi=100)
    ax = figure.add_subplot(111)
    
    ax.scatter(X, Y, color='blue', label="Data Points")

    X_line = np.linspace(min(X), max(X), 100)
    Y_line = np.polyval(np.flip(coefficients.flatten()), X_line)
    
    ax.plot(X_line, Y_line, color='red', label="Fitted Model")
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()

    canvas = FigureCanvasTkAgg(figure, master=new_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
def plot_data(X, Y, coefficients):
    new_window = tk.Toplevel()
    new_window.title("Generated Data")
    figure = plt.Figure(figsize=(6, 4), dpi=100)
    ax = figure.add_subplot(111)
    
    ax.scatter(X, Y, color='blue', label="Data Points")

    X_line = np.linspace(min(X), max(X), 100)
    Y_line = np.polyval(np.flip(coefficients.flatten()), X_line)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    canvas = FigureCanvasTkAgg(figure, master=new_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Linear Regression Model Fitting")
        self.root.geometry('800x700')

        self.X, self.Y, self.coefficients = None, None, None

        self.tab_frame = ttk.Frame(self.root)
        self.tab_frame.pack(fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(self.tab_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="Data Generation")
        self.create_data_generation_widgets()

        self.model_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.model_tab, text="Model Fitting")
        self.create_model_fitting_widgets()

        self.stats_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_tab, text="Statistics")
        self.create_statistics_widgets()

        self.create_results_area()

    def create_results_area(self):
        self.output_frame = ttk.LabelFrame(self.root, text="Results", padding=(10, 10))
        self.output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.results_text = tk.Text(self.output_frame, wrap=tk.WORD, height=8, font=('Helvetica', 12))
        self.results_text.grid(row=0, column=0, sticky='nsew')

        self.scrollbar = ttk.Scrollbar(self.output_frame, command=self.results_text.yview)
        self.scrollbar.grid(row=0, column=1, sticky='ns')
        self.results_text['yscrollcommand'] = self.scrollbar.set

        self.clear_results_button = ttk.Button(self.output_frame, text="Clear Results", command=self.clear_results)
        self.clear_results_button.grid(row=1, column=0, sticky='ne', pady=(5, 0))

    def clear_results(self):
        self.results_text.delete('1.0', tk.END)

    def create_data_generation_widgets(self):
        self.data_frame = ttk.LabelFrame(self.data_tab, text="Generate Data", padding=(10, 10))
        self.data_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(self.data_frame, text="Number of Data Points:").grid(row=0, column=0, sticky='w')
        self.num_points_entry = ttk.Entry(self.data_frame, width=10)
        self.num_points_entry.grid(row=0, column=1, sticky='w')
        
        ttk.Button(self.data_frame, text="Generate Data", command=self.generate_data).grid(row=1, column=0, columnspan=2, pady=(10, 0))

    def create_model_fitting_widgets(self):
        self.model_frame = ttk.LabelFrame(self.model_tab, text="Model Fitting", padding=(10, 10))
        self.model_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Button(self.model_frame, text="Fit Linear Regression", command=self.fit_linear_model).grid(row=0, column=0, columnspan=2, pady=(10, 0))
        ttk.Button(self.model_frame, text="Visualize Plot", command=self.plot_data).grid(row=2, column=0, columnspan=2, pady=(10, 0))
        ttk.Button(self.model_frame, text="Save Model", command=self.save_model).grid(row=3, column=0, pady=(10, 0))
        ttk.Button(self.model_frame, text="Load Model", command=self.load_model).grid(row=3, column=1, pady=(10, 0))

        ttk.Button(self.model_frame, text="Reset Data", command=self.reset_data).grid(row=4, column=0, pady=(10, 0))
        ttk.Button(self.model_frame, text="Reset Plot", command=self.reset_plot).grid(row=4, column=1, pady=(10, 0))

    def create_statistics_widgets(self):
        self.stats_frame = ttk.LabelFrame(self.stats_tab, text="Statistics", padding=(10, 10))
        self.stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(self.stats_frame, text="Statistics for Model").grid(row=0, column=0, sticky='w')
        self.stats_text = tk.Text(self.stats_frame, wrap=tk.WORD, height=10, font=('Helvetica', 12))
        self.stats_text.grid(row=1, column=0, sticky='nsew')
        self.scrollbar_stats = ttk.Scrollbar(self.stats_frame, command=self.stats_text.yview)
        self.scrollbar_stats.grid(row=1, column=1, sticky='ns')
        self.stats_text['yscrollcommand'] = self.scrollbar_stats.set

    def generate_data(self):
        try:
            num_points = int(self.num_points_entry.get())
            if num_points <= 0:
                raise ValueError("Number of data points must be positive.")
            
            self.X, self.Y = generate_random_data(num_points)
            self.results_text.insert(tk.END, f"Generated {num_points} data points.\n")
            self.results_text.insert(tk.END, f"X: {self.X.flatten()}\nY: {self.Y.flatten()}\n")
            self.plot_generated_data()
        except ValueError as e:
            self.results_text.insert(tk.END, f"Error: {str(e)}\n")
        except Exception as e:
            self.results_text.insert(tk.END, f"Unexpected Error: {str(e)}\n")

    def plot_generated_data(self):
        try:
            plot_data(self.X, self.Y, np.array([[0], [0]])) 
            
        except Exception as e:
            self.results_text.insert(tk.END, f"Error plotting generated data: {str(e)}\n")       

    def fit_linear_model(self):
        if self.X is None or self.Y is None:
            self.results_text.insert(tk.END, "Please generate data first.\n")
            return
        
        try:
            self.coefficients, error, r_squared = fit_linear_regression(self.X, self.Y)
            self.results_text.insert(tk.END, f"Linear Model Coefficients: {self.coefficients.flatten()}\n")

            self.update_statistics_tab()

            self.show_statistics_window(error, r_squared)
        except Exception as e:
            self.results_text.insert(tk.END, f"Error fitting model: {str(e)}\n")

    def update_statistics_tab(self):
        beta_0 = self.coefficients[0, 0]
        beta_1 = self.coefficients[1, 0]
        model_equation = f"Predicted Model: Y = {beta_0:.2f} + {beta_1:.2f}X"
        
        self.stats_text.delete('1.0', tk.END)  
        self.stats_text.insert(tk.END, model_equation + "\n")
        self.stats_text.insert(tk.END, f"Beta 0: {beta_0:.2f}\n")
        self.stats_text.insert(tk.END, f"Beta 1: {beta_1:.2f}\n")
        self.stats_text.insert(tk.END, f"Number of Data Points: {len(self.X)}\n")

        if self.X is not None and self.Y is not None:
            try:
                r_squared = calculate_r_squared(self.Y, np.dot(np.hstack([np.ones((self.X.shape[0], 1)), self.X]), self.coefficients))
                self.stats_text.insert(tk.END, f"R-squared: {r_squared:.4f}\n")
                if r_squared >= 0.5:
                    self.stats_text.insert(tk.END, "Fit Quality: Good fit\n")
                else:
                    self.stats_text.insert(tk.END, "Fit Quality: Poor fit\n")
            except Exception as e:
                self.stats_text.insert(tk.END, f"Error calculating R-squared: {str(e)}\n")

    def show_statistics_window(self, error, r_squared):
        try:
            stats_window = tk.Toplevel(self.root)
            stats_window.title("Model Statistics")
            stats_window.geometry('400x300')

            ttk.Label(stats_window, text=f"Mean Squared Error: {error:.2f}").pack(pady=(10, 0))
            ttk.Label(stats_window, text=f"R-squared: {r_squared:.4f}").pack(pady=(10, 0))

            if r_squared >= 0.5:
                ttk.Label(stats_window, text="Fit Quality: Good fit").pack(pady=(10, 0))
            else:
                ttk.Label(stats_window, text="Fit Quality: Poor fit").pack(pady=(10, 0))

            stats_window.protocol("WM_DELETE_WINDOW", lambda: self.on_close_statistics(stats_window))
        except Exception as e:
            self.results_text.insert(tk.END, f"Error showing statistics window: {str(e)}\n")

    def on_close_statistics(self, window):
        window.destroy()

    def plot_data(self):
        if self.X is not None and self.Y is not None:
            try:
                plot_data_in_new_window(self.X, self.Y, self.coefficients)
            except Exception as e:
                self.results_text.insert(tk.END, f"Error plotting data: {str(e)}\n")
        else:
            self.results_text.insert(tk.END, "Please generate data and fit the model first.\n")

    def save_model(self):
        if self.coefficients is not None:
            try:
                model_filename = "linear_model.json"
                with open(model_filename, 'w') as f:
                    json.dump(self.coefficients.flatten().tolist(), f)
                self.results_text.insert(tk.END, f"Model saved as {model_filename}.\n")
            except Exception as e:
                self.results_text.insert(tk.END, f"Error saving model: {str(e)}\n")
        else:
            self.results_text.insert(tk.END, "No model to save. Please fit a model first.\n")

    def load_model(self):
        try:
            model_filename = "linear_model.json"
            if not os.path.exists(model_filename):
                raise FileNotFoundError("Model file does not exist.")
            
            with open(model_filename, 'r') as f:
                self.coefficients = np.array(json.load(f)).reshape(-1, 1)
            self.results_text.insert(tk.END, f"Model loaded from {model_filename}.\n")
        except Exception as e:
            self.results_text.insert(tk.END, f"Error loading model: {str(e)}\n")

    def reset_data(self):
        self.X, self.Y, self.coefficients = None, None, None
        self.results_text.insert(tk.END, "Data has been reset.\n")

    def reset_plot(self):
        self.results_text.insert(tk.END, "Plot has been reset.\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

