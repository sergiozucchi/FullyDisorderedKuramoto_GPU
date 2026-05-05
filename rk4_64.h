/**
RK4 solver on the GPU.

This specific header file contians the CPU functions that will be called by the model.
*/

#ifndef RK4_H // Header guard 
#define RK4_H

#include <stdlib.h>         // For memory management in CPU
#include <cuda_runtime.h>   // For memory management in GPU

/**
Initialization for the RK4 program.

Sets memory allocation in GPU and CPU, and initializes variables for algorithm.

@param  {int}        n       Number of equations to integrate.
@param  {double}     hstep   Fixed step size for the algorithm.
@param  {double*}    yloc   Pointer to the values of the equation's variables.
@param  {function*}  func   ODEs to solve.
*/
void rk4_init(unsigned int n, double hstep, double *yloc, void (*func)(double,double*,double*,void*)); 

/**
Runge-Kutta 4 step from t to t+h. 

The use of an integer for the time is to avoid excessive floating point error.

@param  {int*}   ct      Time counter for the integrator
@param  {void*}  pars    Pointer to parameter structure for the ODEs
*/
void rk4_step (unsigned long int* ct, void* pars);

/**
Runs thermalization steps.

@param  {int*}      ct     Time counter for the integrator
@param  {double}    tterm   Minimum thermalization time
@param  {void*}     pars   Pointer to parameter structure    

@return {double*}   y      Evolution of variables after thermalization
*/
double* rk4_run_term(unsigned long int* ct, double tterm, void* pars);

/**
Runs algorithm until time tf.

@param  {int*}      ct             Time counter for the integrator
@param  {double}    tf              Final time for the simulation
@param  {void*}     file           Pointer to files' structure for the ODEs
@param  {void*}     pars           Pointer to parameter structure for the ODEs
@param  {void}      (*step_eval)    Reference to the pointer evaluation for the system

@return {double*}   y              Evolution of variables at tf
*/
double* rk4_run(unsigned long int* ct, double tf, void* pars, void* files, void (*step_eval)(double, double*, void*,void*)); 

/**
Resets the RK4 program allocations 

@param  {int}       n           New number of ODEs in the system
@param  {double*}   yloc       Pointer to new array with equation variables
@param  {void}      (*func)     New ODEs
*/
void rk4_reset(unsigned int n, double *yloc, void (*func)(double, double*,double*,void*));

/**
Function to free up the memory allocations in the GPU and the CPU for the RK4 program
*/
void rk4_free();

#endif
