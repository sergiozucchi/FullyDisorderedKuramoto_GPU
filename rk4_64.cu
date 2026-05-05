#include "rk4_64.h"         // Includes the RK4 header

/***        File parameters     ***/
static double *hdiv;
static double *y, *ytemp, *k1, *k2, *k3, *k4;

static unsigned int N;

// ODEs pointer (the parameters are time t, variables y, where to save f and parameters pars)
static void (*dydt)(double, double*, double*, void*); 

/***        GPU stepper functions       ***/
/*
Runge-Kutta steps evaluation
*/

__global__ void step2(double* y, double* k1, double* ytemp, const double h2, const unsigned int N){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N){
        ytemp[i]=y[i]+h2*k1[i];
    }
}

__global__ void step3(double* y, double* k2, double* ytemp, const double h2, const unsigned int N){
    int i =blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N){
        ytemp[i]=y[i]+h2*k2[i];    
    }
}

__global__ void step4(double* y, double* k3, double* ytemp, const double h1, const unsigned int N){
    int i =blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N){
        ytemp[i]=y[i]+h1*k3[i];    
    }
}

__global__ void eval(double* y, double* k1, double* k2, double* k3, double* k4, const double h6, const unsigned int N){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N){
        y[i]+=h6*(k1[i]+2*k2[i]+2*k3[i]+k4[i]);
    }
}

/***        CPU Functions           ***/

void rk4_step(unsigned long int* ct, void* pars){

    double t = (*ct)*hdiv[0]; // The time as an int is to avoid as much as posible floating point errors

    (*dydt)(t,y,k1,pars);
    
    step2<<<(N+255)/256,256>>>(y,k1,ytemp,hdiv[1],N);
    (*dydt)(t+hdiv[1],ytemp,k2,pars);

    step3<<<(N+255)/256,256>>>(y,k2,ytemp,hdiv[1],N);
    (*dydt)(t+hdiv[1],ytemp,k3,pars);

    step4<<<(N+255)/256,256>>>(y,k3,ytemp,hdiv[0],N);
    (*dydt)(((*ct)+1)*hdiv[0],ytemp,k4,pars);

    eval<<<(N+255)/256,256>>>(y,k1,k2,k3,k4,hdiv[2],N);    

    (*ct)++; //Increments by 1 the time counter
}

double* rk4_run_term(unsigned long int* ct, double tterm, void* pars){

    double tval=tterm-hdiv[1];
    double t=(*ct)*hdiv[0];

    while(t<tval){
        rk4_step(ct,pars);
        t=(*ct)*hdiv[0];
    }
    return y;
}

double* rk4_run(unsigned long int* ct, double tf, void* pars, void* files, void (*step_eval)(double,double*,void*,void*)){
    double tval=tf-hdiv[1];
    double t=(*ct)*hdiv[0];

    while(t<tval){
        rk4_step(ct,pars);
        t=(*ct)*hdiv[0];
        (*step_eval)(t,y,pars,files);
    }
    return y;
}

void rk4_reset(unsigned int n, double *yloc, void (*func)(double, double*,double*,void*)){
    cudaFree(y);
    cudaFree(ytemp);

    cudaFree(k1);
    cudaFree(k2);
    cudaFree(k3);
    cudaFree(k4);    

    N=n;
    dydt=func;
    
    cudaMalloc((void**)&y, N*sizeof(double));
    cudaMalloc((void**)&ytemp, N*sizeof(double));

    cudaMalloc((void**)&k1, N*sizeof(double));
    cudaMalloc((void**)&k2, N*sizeof(double));
    cudaMalloc((void**)&k3, N*sizeof(double));
    cudaMalloc((void**)&k4, N*sizeof(double));    

    cudaMemcpy(y, yloc, N*sizeof(double), cudaMemcpyHostToDevice);

}

void rk4_init(unsigned int n, double hstep, double *yloc, void (*func)(double,double*,double*,void*)){
    N=n;
    dydt=func;
        
    double h2=hstep*0.5;
    hdiv = (double*)malloc(3*sizeof(double)); 

    hdiv[0]=hstep;
    hdiv[1]=h2;
    hdiv[2]=h2/3.0;
    
    cudaMalloc((void**)&y, N*sizeof(double));
    cudaMalloc((void**)&ytemp, N*sizeof(double));

    cudaMalloc((void**)&k1, N*sizeof(double));
    cudaMalloc((void**)&k2, N*sizeof(double));
    cudaMalloc((void**)&k3, N*sizeof(double));
    cudaMalloc((void**)&k4, N*sizeof(double));    

    cudaMemcpy(y, yloc, N*sizeof(double), cudaMemcpyHostToDevice);
}

void rk4_free(){
    free(hdiv);    
    
    cudaFree(y);
    cudaFree(ytemp);

    cudaFree(k1);
    cudaFree(k2);
    cudaFree(k3);
    cudaFree(k4);
}
