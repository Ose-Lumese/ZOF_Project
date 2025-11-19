# Zero of Functions (ZOF) Solver

## Project Overview

This repository hosts a Python application designed to find the **zeros (roots)** of nonlinear equations using six common numerical methods. The project is split into a standalone **Command-Line Interface (CLI)** tool and a **responsive Web Graphical User Interface (GUI)** built with Flask.

The core of the application uses **SymPy** for robust handling of user-defined functions and automated symbolic differentiation, ensuring the required derivatives for methods like Newton-Raphson are calculated accurately.

## ⚙️ Implemented Numerical Methods

The application allows users to select and analyze root-finding for the following six methods:

1.  **Bisection Method**
2.  **Regula Falsi (False Position) Method**
3.  **Secant Method**
4.  **Newton–Raphson Method**
5.  **Fixed Point Iteration Method**
6.  **Modified Secant Method**

## 💻 Application Components

### 1\. Web GUI (`app.py` and `index.html`)

The web application provides an intuitive interface for inputting functions and viewing detailed iteration results.

  * **Technology Stack:** Flask (Python), SymPy, HTML/Tailwind CSS, and JavaScript.
  * **Features:**
      * Dynamic input fields that change based on the selected numerical method (e.g., showing two guesses for Bisection, and delta for Modified Secant).
      * Real-time display of the final estimated root, error, and iterations.
      * Detailed, responsive table output showing iteration number, new estimate ($\mathbf{x_{new}}$), function evaluation ($\mathbf{f(x_{new})}$), and estimated error ($\mathbf{E_a}$).

### 2\. Command-Line Interface (CLI) (`ZOF_CLI.py`)

A fully functional, standalone Python script that can run any of the six solvers directly from the terminal.

  * **Features:**
      * Interactive prompts for function input, parameters (tolerance, max iterations), and initial guesses.
      * Console output of iteration tables and final results, suitable for quick testing and detailed analysis.

## 📂 Repository Structure

The project adheres to the following structure for deployment and organization:

/ZOF_Project/
  |- ZOF_CLI.py              # Standalone Command Line Application
  |- app.py                  # Flask Web Application Backend
  |- requirements.txt        # Python dependencies (Flask, SymPy, numpy)
  |- ZOF_hosted_webGUI_link.txt # Required submission file with live URL
  |- /templates/             # HTML templates folder
  |    |- index.html         # Web GUI Front-end (HTML/Tailwind CSS/JS)


