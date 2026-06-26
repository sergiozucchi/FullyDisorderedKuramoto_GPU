# About the Program
GPU-based integration of a Fully-Coupled Kuramoto model and its tangent space with random interaction or a user-specified adjacency matrix. The numerical solver is a general Runge-Kutta 4th order method.

This program enables an analysis on the leading Lyapunov vector and the largest Lyapunov exponent, specifically in the weak coupling regime.

# Usage of Repository

This repository contains the program for the numerical simulation which runs on a GPU.

## Structure

- `rk4_64.*`: Generic Runge-Kutta 4th order ODE solver.
- `kuramoto_rk4_64_v3.cu`: CUDA C program for the simulation of the Kuramoto model


## Compilation

Specify the appropriate GPU architecture in the flag `-arch`

```bash
# Clone the repository
git clone <repo-url>
cd FullyDisorderedKuramoto_GPU

nvcc -O3 -lcurand -lcublas -lcuda rk4_64.cu kuramoto_rk4_64_v3.cu -arch=sm_70 -o kuramotoGPU
```

## Running Kuramoto
### Help message
Run `kuramotoGPU -h` to get the help message:

```
Usage: ./kuramotoGPU [OPTIONS] filebase

Required:
  -N, --num N               Number of oscillators
  -J, --coupling J          Coupling strength coefficient
  filebase                  Base filename for output

Flags:
  -h, --help                Show this help message and exits
  -n, --normal              Normal distributed frequencies          (default: Uniform)
  -A, --adj                 Stores the full adjacency matrix        (uses more memory)
  -i, --reload-theta        Reload only thetas as intial conditions from file

Reload Options:
      --reload-ic           Reload full initial conditions from file
      --reload-freq         Reload frequencies from file
      --reload-adj          Reload adjacency A from file

Simulation Parameters:
  -e, --eta eta             Matrix asymmetry parameter [-1,1]               (default: 0.0)
  -I, --amplitude I         Scale for random uniform initial conditions     (default: 1.0)
  -f, --freq f              Frequency scale                                 (default: 1.0)
  -t, --time t              Total integration time (excl. therm.)           (default: 1e2)
  -o, --output-step dt      Integration steps between outputs               (default: 10)
  -d, --timestep h          Integrator timestep                             (default: 1e-2)
  -l, --tau tau             Time between Lyapunov exponent evaluations      (default: 20.0)
  -w, --thterm t            Thermalization time                             (default: 100.0)
  -W, --thterm-lyap t       Tangent space thermalization time               (default: 500.0)
  -s, --seed s              Random seed                                     (default: 0)
Mode Parameter (Add other modes if needed):
  -m, --mode m              Mode selector       (default: 0)
                              0 = Mode for one run of MLE evaluation.
                              1 = Mode for multiple runs for the average in MLE evaluations.
                              2 = Mode for one run to save the full vector.
                              3 = Mode 1 with L0 norm instead of L2 norm.

Output Parameter:
  -D, --dense d             Output density      (default: 0)
                              0 = MLE and final state
                              1 = 0 + thetas
                              2 = 1 + adjacency

Example:
  ./kuramotoGPU -N 500 -J 0.5 -e 1 filebase

```

### Adding modes
If you need add a new mode for something specific in the simulation, write the step evaluation function (`step_eval_*`) and the general mode function (`mode_*`) in the Kuramoto CUDA file. Then, add the mode in the node selection section and the flag `-m`.

See examples in the code.

### Examples

This execution runs a 6400 oscilator simulation with assymetry parameter $\eta=1$ and coupling strength $J=0.1$.
```bash
./kuramoto -nA -I 1.0 -N 6400 -J 0.1 -e 1 -t 5000 -d 0.1 -l 100 -w 100 -m 1 -s 0  out
```

### Output and Input files

The required argument `filebase` gives the base name for the output and input files. In general:

- `filebase.out` is a text file containing command lines, runtime and some general information.
- `filebase_fs.dat` is the final state in binary format. The final time and step size h are appended at the end.
- `filebase_freq.dat` is the oscillators natural frequency $\omega$ in binary format.
- `filebase_adj.dat` is the adjacency matrix in columns and binary format.

## Data files

The data folder contains:

- `figData/`: The necessary data to replicate the figures in the notebook `kuramotoLyap_Figures.ipynb`.
- `fits*`: A numpy binary file with the fitted coefficients for the linear regressions.

    - `fitsLE_inf.npz`: Power law scaling coefficients $\alpha$ and $k$ for the regression of $\lambda_N$. First dimension is the coupling strength index in $J=\[0.14,0.2,0.28,0.4\]$, the second dimension is the asymmetry parameter index in $\eta=\[-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1\]$ and the third dimension is the value \[0\] and the error \[1\].
    - `fitsWeak.npz`: Coefficients $a$ and $b$ for the regression of $\lambda_{\infty}$ in the weak coupling regime. The first dimension is the asymmetry parameter index in $\eta=\[-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1\]$ and the second contains the value \[0\] and the error \[1\].
    
- `rawData/`: Raw data used in the analysis.
    
    - `linfopt_bootstrap_J*`: The bootstrap iterations for the estimation of $\lambda_{\infty}$. For values of the asymmetry parameter $\eta=\[-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1\]$, the first dimension corresponds to the asymmetry parameter index and the second column to the bootstrap index.
    - `data_MLE_J*/`: Raw data from the simulations with the simulation seed (1st column), asymmetry parameter (2nd column), Lyapunov exponent $\lambda_N$ (3rd column) and standard deviation from the finite time Lyapunov exponent (4th column). Mean and standard deviation for each label can be found in the file `avLE_sim.csv`
    - `*.csv`: Labeled raw data for the $\[\lambda_N\]$, $\lambda_{\infty}$ and diffusion $\[D_N\]$.

# Acknowledgments

Sergio Zucchi aacknowledges financial support by CSIC under the JAE Intro ICU Programme Ref. JAEICU_25_03514

# References

The code, data and jupyter notebook were used to generate the figures in Sergio Zucchi, Iván León and Diego Pazó "The Lyapunov exponent is insensitive to coupling asymmetry in the fully-disordered Kuramoto model at weak coupling" (Date) [In progress]

<p align="right">(<a href="#readme-top">back to top</a>)</p>
