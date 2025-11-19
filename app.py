import os
import sympy as sp
from flask import Flask, render_template, request, jsonify
from typing import List, Dict, Tuple, Optional

# --- Initialization ---
app = Flask(__name__)

# Define the symbolic variable for SymPy operations
x_sym = sp.Symbol('x')

# --- Core Solver Functions (Same as ZOF_CLI, adapted for Web) ---

def run_solver(
    method: str,
    f_str: str,
    initial_guesses: List[float],
    tol: float = 1e-6,
    max_iter: int = 50,
    delta: Optional[float] = None # Used only for Modified Secant
) -> Dict:
    """
    Generalized function to run a specific root-finding method for the web app.
    Returns a dictionary of results and iteration history.
    """
    results = {'success': False, 'root': 'N/A', 'error': 'N/A', 'iterations': 0, 'history': [], 'message': ''}
    
    try:
        f_sym = sp.sympify(f_str)
        f = sp.lambdify(x_sym, f_sym, 'numpy')
        
        # Automatic Differentiation for Newton and Modified Secant
        df_sym = sp.diff(f_sym, x_sym)
        df = sp.lambdify(x_sym, df_sym, 'numpy')

    except (sp.SympifyError, TypeError) as e:
        results['message'] = f"Invalid function expression: {e}"
        return results

    # --- Bisection & Regula Falsi Setup ---
    if method in ["Bisection", "Regula Falsi"]:
        if len(initial_guesses) < 2:
            results['message'] = "Bisection/Regula Falsi require two initial guesses (a and b)."
            return results
        
        a, b = initial_guesses[0], initial_guesses[1]
        if f(a) * f(b) >= 0:
            results['message'] = "Bisection/Regula Falsi require f(a) and f(b) to have opposite signs."
            return results

        if method == "Bisection":
            for i in range(1, max_iter + 1):
                c = (a + b) / 2
                f_c = f(c)
                error = abs(b - a) / 2

                results['history'].append({'i': i, 'x_new': c, 'f_x': float(f_c), 'error': float(error), 'a': a, 'b': b})

                if error < tol:
                    results.update({'success': True, 'root': c, 'error': error, 'iterations': i, 'message': 'Converged successfully.'})
                    return results

                if f(a) * f_c < 0:
                    b = c
                else:
                    a = c
            
            # Max iterations reached
            results.update({'success': True, 'root': c, 'error': error, 'iterations': max_iter, 'message': 'Reached max iterations.'})
            return results

        elif method == "Regula Falsi":
            x_new = 0.0
            error = float('inf')
            for i in range(1, max_iter + 1):
                if f(a) == f(b):
                    results['message'] = "Regula Falsi: Division by zero (f(a) == f(b))."
                    return results

                x_new = b - f(b) * (a - b) / (f(a) - f(b))
                f_x_new = f(x_new)
                error = abs(b - a) # Error using interval width

                results['history'].append({'i': i, 'x_new': x_new, 'f_x': float(f_x_new), 'error': float(error), 'a': a, 'b': b})

                if error < tol and i > 1:
                    results.update({'success': True, 'root': x_new, 'error': error, 'iterations': i, 'message': 'Converged successfully.'})
                    return results

                if f(a) * f_x_new < 0:
                    b = x_new
                else:
                    a = x_new
            
            results.update({'success': True, 'root': x_new, 'error': error, 'iterations': max_iter, 'message': 'Reached max iterations.'})
            return results


    # --- Iterative Methods Setup (Newton, Secant, Fixed Point, Modified Secant) ---
    else:
        if len(initial_guesses) < 1:
            results['message'] = "Iterative methods require at least one initial guess (x0)."
            return results

        x_current = initial_guesses[0]
        x_prev = 0.0 # Placeholder for Secant/Fixed Point
        
        if method == "Secant":
            if len(initial_guesses) < 2:
                 results['message'] = "Secant method requires two initial guesses (x0 and x1)."
                 return results
            x_prev = initial_guesses[0]
            x_current = initial_guesses[1]
        
        elif method == "Fixed Point":
            # For fixed-point, we solve x = g(x). We treat f(x) as g(x).
            g = f # g(x) is the input function string
            x_prev = x_current


        for i in range(1, max_iter + 1):
            if method == "Fixed Point":
                x_new = g(x_prev)
                error = abs(x_new - x_prev)
                f_x = f(x_new) # Evaluate f(x) at the new root estimate for display

            elif method == "Newton-Raphson":
                f_x_current = f(x_current)
                df_x = df(x_current)
                if abs(df_x) < 1e-12: # Check for near-zero derivative
                    results['message'] = f"Newton-Raphson: Derivative is near zero at x={x_current:.6f}"
                    return results
                x_new = x_current - f_x_current / df_x
                error = abs(x_new - x_current)
                f_x = f(x_new)

            elif method == "Secant":
                f_current = f(x_current)
                f_prev = f(x_prev)
                if abs(f_current - f_prev) < 1e-12:
                    results['message'] = "Secant: Denominator (f(x_current) - f(x_prev)) is near zero."
                    return results
                x_new = x_current - f_current * (x_current - x_prev) / (f_current - f_prev)
                error = abs(x_new - x_current)
                f_x = f(x_new)
            
            elif method == "Modified Secant":
                if delta is None or delta == 0:
                    results['message'] = "Modified Secant requires a non-zero delta perturbation value."
                    return results
                    
                f_x_current = f(x_current)
                # Approximate derivative: (f(x + delta*x) - f(x)) / (delta*x)
                df_approx_denom = (delta * x_current)
                if abs(df_approx_denom) < 1e-12:
                    results['message'] = f"Modified Secant: Denominator (delta*x) is near zero at x={x_current:.6f}"
                    return results
                    
                df_approx = (f(x_current + df_approx_denom) - f_x_current) / df_approx_denom
                
                if abs(df_approx) < 1e-12:
                    results['message'] = "Modified Secant: Approximate derivative is near zero."
                    return results
                    
                x_new = x_current - f_x_current / df_approx
                error = abs(x_new - x_current)
                f_x = f(x_new)

            else:
                results['message'] = f"Unknown method: {method}"
                return results

            # Store iteration data
            results['history'].append({'i': i, 'x_new': x_new, 'f_x': float(f_x), 'error': float(error)})

            # Check convergence
            if error < tol:
                results.update({'success': True, 'root': x_new, 'error': error, 'iterations': i, 'message': 'Converged successfully.'})
                return results

            # Update for next iteration
            if method == "Secant":
                x_prev = x_current
                x_current = x_new
            elif method in ["Newton-Raphson", "Modified Secant", "Fixed Point"]:
                x_current = x_new
                if method == "Fixed Point":
                    x_prev = x_new
        
        # Max iterations reached
        results.update({'success': True, 'root': x_new, 'error': error, 'iterations': max_iter, 'message': 'Reached max iterations.'})
        return results


# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main solver page."""
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    """Handles the form submission and runs the solver."""
    try:
        data = request.json
        
        # Parse inputs
        method = data.get('method')
        f_str = data.get('function_expr').replace('^', '**') # Allow common power notation
        tol = float(data.get('tolerance', 1e-6))
        max_iter = int(data.get('max_iterations', 50))
        delta = float(data.get('delta', 0.01))

        # Initial Guesses
        g1_str = data.get('guess1')
        g2_str = data.get('guess2')
        
        initial_guesses = []
        if g1_str:
            initial_guesses.append(float(g1_str))
        if g2_str:
            initial_guesses.append(float(g2_str))
            
        # Run the solver
        result = run_solver(method, f_str, initial_guesses, tol, max_iter, delta)
        
        # Clean up history for JSON serialization (SymPy objects might be present)
        for item in result['history']:
            for key in ['x_new', 'f_x', 'error', 'a', 'b']:
                if key in item and not isinstance(item[key], (float, int)):
                     # Convert SymPy float to standard float if needed
                     item[key] = float(item[key])
        
        return jsonify(result)

    except ValueError:
        return jsonify({'success': False, 'message': 'Invalid numerical input (guesses, tolerance, iterations, or delta).'}), 400
    except Exception as e:
        return jsonify({'success': False, 'message': f'An unexpected error occurred: {str(e)}'}), 500

