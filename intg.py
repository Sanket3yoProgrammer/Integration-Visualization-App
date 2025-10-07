import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
import time

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Integration Visualization App", layout="wide")
st.title("Integration Visualization")
st.write("Explore the construction of integrals using iterative Riemann sums.")

# ---------- INPUT PANEL ----------
with st.sidebar:
    st.header("Input Parameters")
    func_str = st.text_input("Enter the function f(x):", "x**2")
    a = st.number_input("Lower limit (a):", value=0.0)
    b = st.number_input("Upper limit (b):", value=5.0)
    n = st.slider("Number of iterations/subdivisions:", min_value=1, max_value=500, value=10)
    method = st.selectbox("Integration Method:", ["Left", "Right", "Midpoint", "Trapezoid"])
    show_table = st.checkbox("Display Iteration Table", value=True)

# ---------- FUNCTION PARSING AND NORMALIZATION ----------
x = sp.symbols('x')
try:
    func_sympy = sp.sympify(func_str)
    f_lambdified = sp.lambdify(x, func_sympy, modules=["numpy"])
    st.subheader("Normalized Function")
    st.latex(sp.latex(func_sympy))
except (sp.SympifyError, TypeError):
    st.error("Invalid function format. Please enter a valid mathematical expression.")
    f_lambdified = lambda x: np.zeros_like(x)

# ---------- CALCULATIONS ----------
x_vals = np.linspace(a, b, n+1)
dx = (b - a) / n

if method == "Left":
    heights = f_lambdified(x_vals[:-1])
elif method == "Right":
    heights = f_lambdified(x_vals[1:])
elif method == "Midpoint":
    heights = f_lambdified((x_vals[:-1] + x_vals[1:]) / 2)
elif method == "Trapezoid":
    heights = (f_lambdified(x_vals[:-1]) + f_lambdified(x_vals[1:])) / 2

areas = heights * dx
cumulative = np.cumsum(areas)
total_integral = cumulative[-1]

# ---------- GRAPH AREA WITH LOADER ----------
st.subheader("Graph")
placeholder = st.empty()  # Placeholder for skeleton/loader

with placeholder.container():
    st.write("Rendering graph...")
    skeleton = st.empty()
    for i in range(5):  # Simple animation for loader effect
        skeleton.progress((i+1)*20)
        time.sleep(0.1)

    # Once 'loaded', display the actual plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 4))
    x_plot = np.linspace(a, b, 500)
    ax.plot(x_plot, f_lambdified(x_plot), label="f(x)", color="cyan", linewidth=2)

    for i in range(n):
        if method == "Left":
            xi = x_vals[i]
        elif method == "Right":
            xi = x_vals[i+1] - dx
        elif method == "Midpoint":
            xi = (x_vals[i] + x_vals[i+1])/2 - dx/2
        elif method == "Trapezoid":
            xi = x_vals[i]
        ax.bar(xi, heights[i], width=dx, alpha=0.3, color="orange", align='edge', edgecolor='white')

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(f"Integration Visualization ({method} Riemann Sum)\nApproximate Integral = {total_integral:.4f}")
    ax.legend()
    placeholder.pyplot(fig)

# ---------- ITERATION TABLE ----------
if show_table:
    st.subheader("Iteration Table: Step-by-Step Contribution to Integral")
    table_data = pd.DataFrame({
        "Iteration": np.arange(1, n+1),
        "x_i (Position)": np.round(x_vals[:-1], 4),
        "f(x_i) (Height)": np.round(heights, 4),
        "f(x_i)·Δx (Rectangle Area)": np.round(areas, 4),
        "Cumulative Sum (Approx Integral)": np.round(cumulative, 4)
    })
    st.dataframe(table_data.style.background_gradient(cmap="Oranges", subset=["Cumulative Sum (Approx Integral)"], axis=0), height=300)

# ---------- DETAILED EXPLANATION ----------
st.subheader("Detailed Explanation")
st.markdown("""
Consider a curve defined by the function f(x). The integral represents the total area beneath the curve between the limits a and b.

1. **Subdivision of the interval**: The interval [a, b] is divided into n equal parts.  
   For example, for f(x) = x² over [0, 5], the width of each subinterval is:
   $$\\Delta x = \\frac{b-a}{n} = \\frac{5-0}{n} = \\frac{5}{n}$$

2. **Height of rectangles**: For each subinterval, the value of the function at the right endpoint determines the height of a rectangle:
   $$x_i = i \\cdot \\Delta x, \\quad f(x_i) = (x_i)^2$$

3. **Area of rectangles**: Multiply the height by the width of the subinterval to compute each rectangle's area:
   $$\\text{Area}_i = f(x_i) \\cdot \\Delta x = \\left(i \\frac{5}{n}\\right)^2 \\cdot \\frac{5}{n} = \\frac{125 i^2}{n^3}$$

4. **Cumulative sum (Riemann sum)**: Summing the areas of all rectangles approximates the integral:
   $$R_n = \\sum_{i=1}^{n} \\frac{125 i^2}{n^3} = \\frac{125}{n^3} \\sum_{i=1}^{n} i^2 = \\frac{125 (n+1)(2n+1)}{6 n^2}$$

5. **Definite integral**: As the number of rectangles increases ($n \\to \\infty$), the Riemann sum converges to the exact area:
   $$\\int_0^5 x^2 dx = \\lim_{n \\to \\infty} R_n = \\frac{125}{3} \\approx 41.6667$$

6. **Iteration table & visualization**: You can create a table showing each rectangle's position, height, area, and cumulative contribution. Using more rectangles improves accuracy and illustrates the concept of integration clearly.

This combines a graphical representation with numerical data for professional and clear understanding.
""", unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("""
---

<div style="text-align: center; font-size: 14px; color: gray;">
Made with fun by <a href="https://github.com/sanket3yoprogrammer" target="_blank" style="text-decoration: none; color: gray;">sanketyoprogrammer</a>
</div>
""", unsafe_allow_html=True)
