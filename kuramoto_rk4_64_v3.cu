#include <sys/time.h> // Timer library
#include <stdlib.h> // For CPU allocation
#include <math.h> // For math functionn
#include <random> // To initiate the random initial conditions
#include <stdio.h> // File, input and output library
#include <unistd.h> // To read bash calls to the program
#include <iostream> // To print
#include <getopt.h> // For the long options and better reading

#include "cublas_v2.h" // CUBLAS library 
#include <curand_kernel.h> // CURAND library for random numbers in cuda
#include <cuda_runtime.h> // For memory management in GPU

#include "rk4_64.h" // RK4 64 bits algorithm

/**
Parameters to pass onto the model
*/
typedef struct parameters {
    unsigned long int N; // Number of oscillators
    
    double *y2; // angle for the cos separation
    double *f2; // Coupling with j angle for the cos separation
    double *omegas; // Natural frequencies ot the system
    double *adj; // Stores the adjacency matrix for the system when needed
    
    double eta;
    double JsqrtN;
    double eS; // Symmetric part
    double eA; // Asymmetric part
    
    unsigned long int ct; // Initial time counter
    int dt; // Separation time in integration steps for outputs.
    int ntau; // Counter to save MLE
    double h; // Integration step
    
    double tf; // Final simulation time
    double tterm; // Termalization for the trajectories
    double tterm_lyap; // Termalization for the tangent space

    int A; // Bool for efficient adj or stored
    int reload_ic;

    double *yloc; // CPU pointer to save output
    double lyap;
    
    int count2save;
    int nsave;
    int dense;
    int seed;

    cublasHandle_t handle; // CUBLAS handle for operations
    curandStatePhilox4_32_10_t *state; // Counter-based PRNG state (stable for parallel programming)
    
    char* filebase;
    
} parameters;


/**
Structure files for the different modes (when creating a mode you can add whatever you want)
*/
typedef struct files_lyap {
    FILE *outtimes;
    FILE *outthetas;
    FILE *outlyap;
    FILE *outnorm;
} files_lyap;

typedef struct files_MLE {
    double lyap;
    double lyap2;
} files_MLE;

typedef struct files_cT {
    FILE *outthetas;
} files_cT;

/**
Dynamic equation for a Kuramoto oscillators and the tangent space

    d theta_i / dt = omega_i + (J/sqrt(N)) * [cos(theta_i)*f2[2*i+1]-sin(theta_i)*f2[2*i]]

where f2 is a 2N array which gets all contribution from the connections (using the adjacency matrix A_ij) to theta_i (i.e. f2[2*i+1]sum^N_{j=0} A_ij * sin(theta_j)). 2*i+1 is sin (odd) and 2*i is cos (even).
The tangent space starts from N to 2N.

@param  {double*}   y2         4N array for the sin or cos of theta_i and its multiplication with delta_i
@param  {double*}   f          2N array for the oscillator theta_i and the tangent space delta_i
@param  {double*}   f2         4N array for contributions to the connections for theta_i
@param  {double*}   omegas     N array for natural frequencies
@param  {double}    JsqrtN     Constant variable J/sqrt(N)
@param  {int}       N          Number of oscillators
*/
__global__ void kuramoto_traj_tan(double *y2, double* f, double *f2,  double *omegas ,const double JsqrtN, const unsigned long int N){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<N){
        int j = N+i; // Offset from theta to v
        
        const double2* __restrict__ y2_cs = reinterpret_cast<double2*>(y2);
        const double2* __restrict__ f2_cs = reinterpret_cast<double2*>(f2);
        
        // cs mean it is (cos, sin) accessed through cs.x and cs.y respectively.
        // Acs is the row sum with the adjacency matrix
        double2 cs_theta = y2_cs[i];
        double2 cs_v = y2_cs[j];
        double2 Acs_theta = f2_cs[i];
        double2 Acs_v = f2_cs[j];
        
        
        f[i]=omegas[i]+JsqrtN*(cs_theta.x*Acs_theta.y-cs_theta.y*Acs_theta.x); 
        f[j]=JsqrtN*(cs_theta.x*Acs_v.x+cs_theta.y*Acs_v.y-(cs_v.x*Acs_theta.x+cs_v.y*Acs_theta.y));
    } 
}

__global__ void kuramoto_traj(double *y2, double *f, double *f2, double *omegas, const double JsqrtN, const unsigned long int N){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<N){
    
        const double2* __restrict__ y2_cs = reinterpret_cast<double2*>(y2);
        const double2* __restrict__ f2_cs = reinterpret_cast<double2*>(f2);
        
        // cs mean it is (cos, sin) accessed through cs.x and cs.y respectively.
        // Acs is the row sum with the adjacency matrix
        double2 cs_theta = y2_cs[i];
        double2 Acs_theta = f2_cs[i];
        
        f[i]=omegas[i]+JsqrtN*(cs_theta.x*Acs_theta.y-cs_theta.y*Acs_theta.x); 
    }
}


/**
Given the separation of sin(theta_j-theta_i) into products of sin and cos, this will be stored in a 2N array with sin in odds and cos in evens. Following that the cos and sin will be multiplied by the tangent space to get the other contributions for the tangent space equations

@param  {double*}   y       2N array for the angle theta_i
@param  {double*}   y2      4N array for the sin and cos of theta_i
@param  {int}       N       Number of oscillator
*/
__global__ void makey2_tan(double *y, double *y2, const unsigned long int N){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<N){
    
        double2* __restrict__ y2v = reinterpret_cast<double2*>(y2);

        double s, c;
        sincos(y[i], &s, &c);
        
        double v = y[N+i];
        
        y2v[i] = make_double2(c, s);
        y2v[N+i] = make_double2(c*v, s*v);
    }
}

__global__ void makey2(double *y, double *y2, const unsigned long int N){
    int i = blockIdx.x*blockDim.x+threadIdx.x; // Number thread for the computation.
    if (i<N){
    
        double2* __restrict__ y2v = reinterpret_cast<double2*>(y2);

        double s, c;
        sincos(y[i], &s, &c);
        
        y2v[i] = make_double2(c, s);
    }
}

/**
Returns the sum of the contributions from other oscillators to i.

Only used when A is true (We don't save the adjacency matrix).

@param  {double*}           y2              4N array for the trig-angles
@param  {double*}           f               4N array for the sum output
@param  {int}               N               Number of oscillators
@param  {double}            eS              Symmetric constant sqrt((1+eta)/2)
@param  {double}            eA              Asymmetric constant sqrt((1-eta)/2)
@param  {PhiloxState*}      globalState     Global state for the Philox RNG
*/
__global__ void makef2_tan(double* y2, double* f, const unsigned long int N, const double eS, const double eA, curandStatePhilox4_32_10_t *globalState){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<N){
        curandStatePhilox4_32_10_t state = *globalState;
        skipahead(2*i*(i-1),&state);
        
        const double2* __restrict__ y2v = reinterpret_cast<const double2*>(y2);
        
        double2 tmpSum = make_double2(0.0,0.0);
        double2 tmpPert = make_double2(0.0,0.0);
        double elem;
        double2 rnd;
        
        
        // Lower triangular loop
        for(int j=0;j<i;j++){

            rnd = curand_normal2_double(&state);
            elem=eS*rnd.x-eA*rnd.y;

            tmpSum.x+=elem*y2v[j].x;
            tmpSum.y+=elem*y2v[j].y;
            
            tmpPert.x+=elem*y2v[j+N].x;
            tmpPert.y+=elem*y2v[j+N].y;
            
        }
        
        skipahead(2*2*i,&state);
        
        // Upper triangular loop
        for(int j=i+1;j<N;j++){
            
            rnd = curand_normal2_double(&state);
            elem=eS*rnd.x+eA*rnd.y;

            tmpSum.x+=elem*y2v[j].x;
            tmpSum.y+=elem*y2v[j].y;
            
            tmpPert.x+=elem*y2v[j+N].x;
            tmpPert.y+=elem*y2v[j+N].y;
            
            skipahead(2*2*(j-1),&state);
        }
        
        // Saving sums
        
        reinterpret_cast<double2*>(f)[i] = tmpSum;
        reinterpret_cast<double2*>(f)[N+i] = tmpPert;

    }
}

__global__ void makef2(double* y2, double* f, const unsigned long int N, const double eS, const double eA, curandStatePhilox4_32_10_t *globalState){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<N){
        curandStatePhilox4_32_10_t state = *globalState;
        skipahead(2*i*(i-1),&state);
        
        const double2* __restrict__ y2v = reinterpret_cast<const double2*>(y2);
        
        double2 tmpSum = make_double2(0.0,0.0);
        double elem;
        double2 rnd;
        
        
        // Lower triangular loop
        for(int j=0;j<i;j++){

            rnd = curand_normal2_double(&state);
            elem=eS*rnd.x-eA*rnd.y;

            tmpSum.x+=elem*y2v[j].x;
            tmpSum.y+=elem*y2v[j].y;
                       
        }
        
        skipahead(2*2*i,&state);
        
        // Upper triangular loop
        for(int j=i+1;j<N;j++){
            
            rnd = curand_normal2_double(&state);
            elem=eS*rnd.x+eA*rnd.y;

            tmpSum.x+=elem*y2v[j].x;
            tmpSum.y+=elem*y2v[j].y;
            
            skipahead(2*2*(j-1),&state);
        }
        
        // Saving sums
        
        reinterpret_cast<double2*>(f)[i] = tmpSum;

    }
}

/**
Generates the adjacency matrix

@param  {double*}           adj             N*N Adjacency matrix
@param  {int}               N               Number of oscillators
@param  {double}            eS              Symmetric constant sqrt((1+eta)/2)
@param  {double}            eA              Asymmetric constant sqrt((1-eta)/2)
@param  {PhiloxState*}      globalState     Global state for the Philox RNG
*/
__global__ void makeadj(double* adj, const unsigned long int N, const double eS, const double eA, curandStatePhilox4_32_10_t *globalState){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<N){
        curandStatePhilox4_32_10_t state = *globalState;
        
        // The extra 2 in this function is due to bug in the function..
        // Check symmetry in matrix to make sure if in doubt
        skipahead(2*i*(i-1),&state); 
        
        double* __restrict__ row = adj + i * N;
        double2 rnd;

        // Lower triangular loop
        for(int j=0;j<i;j++){

            rnd = curand_normal2_double(&state);
            row[j]=eS*rnd.x-eA*rnd.y;
        }
        
        // Diagonal element
        row[i]=0.0;
        skipahead(2*2*i,&state);
        
        // Upper triangular loop
        for(int j=i+1;j<N;j++){
            
            rnd = curand_normal2_double(&state);
            row[j]=eS*rnd.x+eA*rnd.y;

            skipahead(2*2*(j-1),&state);
        }

    }
}

/**
Initializes the global state for the PhiloxRNG with a seed.
You can't call __device__ curand_init function from __host__ main. That's why this is here.

@param  {PhiloxState*}      globalState     Global state for the Philox RNG
@param  {int}               seed            Seed for the PRNG
*/
__global__ void init_global_state(curandStatePhilox4_32_10_t *globalState, int seed){
    curand_init(seed, 0, 0, globalState); //Global state without update
}

/**
Reduces the final state so the trajectories do not explode if reused in another simulation.
We don't do anything with the tangent space because we keep normalizing when calculating the FTLE. 
*/
__global__ void reduceFinalState(double* y, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<N){
        double s, c;
        sincos(y[i], &s, &c);
        y[i]=atan2(s,c);
    }
}

/**
Generates the f2 array for a certain time of a simulation

@param  {double*}   y2      2N array for the trig-angle of theta_i (2i+1 for sin and 2i for cos) 
@param  {double*}   f2      2N array for trig-angle sum contributions       
@param  {void*}     pars    Pointer to parameter structure
*/
void makecouplings_tan(double *y2, double *f2, void *pars){

    parameters *p = (parameters *) pars;

    if (p->A){
        double alpha=1.0;
        double beta=0.0;
        // Gets the multiplication as for the trajectories
        cublasDgemv(p->handle, CUBLAS_OP_T, p->N, p->N, &alpha, p->adj,p->N,y2,2,&beta,f2,2);
        cublasDgemv(p->handle, CUBLAS_OP_T, p->N, p->N, &alpha, p->adj,p->N,y2+1,2,&beta,f2+1,2);
        // Gets the multiplication for the tangent space
        cublasDgemv(p->handle, CUBLAS_OP_T, p->N, p->N, &alpha, p->adj,p->N,y2+2*p->N,2,&beta,f2+2*p->N,2);
        cublasDgemv(p->handle, CUBLAS_OP_T, p->N, p->N, &alpha, p->adj,p->N,y2+2*p->N+1,2,&beta,f2+2*p->N+1,2);
    } else  {
        makef2_tan<<<(p->N+255)/256,256>>>(y2,f2,p->N,p->eS, p->eA, p->state);
    }
}

void makecouplings(double *y2, double *f2, void *pars){

    parameters *p = (parameters *) pars;

    if (p->A){
        double alpha=1.0;
        double beta=0.0;
        // IMPORTANT: The CUBLAS function assumes adj is stored by columns..
        cublasDgemv(p->handle, CUBLAS_OP_T, p->N, p->N, &alpha, p->adj, p->N,y2, 2,&beta,f2,2);
        cublasDgemv(p->handle, CUBLAS_OP_T, p->N, p->N, &alpha, p->adj, p->N,y2+1, 2,&beta,f2+1,2);
    } else  {
        makef2<<<(p->N+255)/256,256>>>(y2, f2, p->N, p->eS, p->eA, p->state);
    }
}

/**
Set of steps for the numerical integration of the differential equations.

@param  {double}    t       Time variable
@param  {double*}   y       2N array for the angles theta_i and dtheta_i
@param  {double*}   f       2N array to save the new theta_i after integration.
@param  {void*}     pars    Pointer to parameter structure
*/
void dydt_tan(double t, double* y, double* f, void* pars){
    parameters *p = (parameters *) pars;

    makey2_tan<<<(p->N+255)/256,256>>>(y,p->y2,p->N);
    makecouplings_tan(p->y2,p->f2,pars);

    kuramoto_traj_tan<<<(p->N+255)/256,256>>>(p->y2, f, p->f2, p->omegas, p->JsqrtN, p->N);
}

void dydt(double t, double* y, double* f, void* pars){
    parameters *p = (parameters *) pars;

    makey2<<<(p->N+255)/256,256>>>(y,p->y2,p->N);
    makecouplings(p->y2,p->f2,pars);

    kuramoto_traj<<<(p->N+255)/256,256>>>(p->y2, f, p->f2, p->omegas, p->JsqrtN, p->N);
}

/**
Function to normalize N-dimensional vector space and return the normalization factor.

@param      {double*}       y       Vector pointer to normalize 
@param      {int}           N       Dimension of the vector to normalize 
@param      {CuBLAShandle}  handle  Handle for the library operations

@returns    {double}        norm    Normalization factor
*/
double normVec(double* y, int N, cublasHandle_t handle){

    double norm;
    cublasDnrm2(handle,N,y, 1,&norm); // the y pointer starts at the tangent space
    
    double alpha=1/norm;
    cublasDscal(handle,N,&alpha,y,1);
    
    return norm;
}

/**
Step evaluation function for Maximum Lyapunov Exponent

@param  {double}    t       Time variable
@param  {double*}   y       2N array for angles and tangent space
@param  {void*}     pars    Pointer to parameter structure
@param  {files*}    files   Pointer to file structure
*/
void step_eval_lyap(double t,double* y, void* pars, void* files){
    parameters *p = (parameters *) pars;
    files_lyap *f = (files_lyap *) files;
    
    p->nsave++;
    
    if (p->nsave>=p->ntau){
        
        double norm=normVec(y+p->N,p->N,p->handle);
        
        if (p->dense>=1){
            fwrite(&norm,sizeof(double),1,f->outnorm);
        }
        
        p->lyap=std::log(norm)/(p->nsave*p->h);
        fwrite(&(p->lyap),sizeof(double),1,f->outlyap);
        
        p->nsave=0;
    }
    
    if (p->dense>=1){
        
        p->count2save++;
        
        if (p->count2save%p->dt==0){
            
            fwrite(&t,sizeof(double),1,f->outtimes);

            cublasGetVector(2*p->N, sizeof(double), y, 1, p->yloc, 1);
            fwrite(p->yloc,sizeof(double),2*p->N,f->outthetas);
            
            p->count2save=0;       
        }
    }
}

void step_eval_MLE(double t,double* y, void* pars, void* files){
    parameters *p = (parameters *) pars;
    files_MLE *f = (files_MLE *) files;
    
    p->nsave++;
    
    if (p->nsave>=p->ntau){
        
        double norm=normVec(y+p->N,p->N,p->handle);
        
        p->lyap=std::log(norm)/(p->nsave*p->h);
        f->lyap+=p->lyap;
        f->lyap2+=(p->lyap*p->lyap);
        
        p->count2save++;
        p->nsave=0;
    }
}

void step_eval_checkTrans(double t,double* y, void* pars, void* files){
    parameters *p = (parameters *) pars;
    files_cT *f = (files_cT *) files;
    
    p->nsave++;
    
    if (p->nsave>=p->ntau){
        
        double norm=normVec(y+p->N,p->N,p->handle);
        double tf=p->ct*p->h;
        double lyap=std::log(norm)/(p->nsave*p->h);
        
        cublasGetVector(2*p->N, sizeof(double), y, 1, p->yloc, 1);
        fwrite(p->yloc,sizeof(double),2*p->N,f->outthetas);
        fwrite(&tf,sizeof(double),1,f->outthetas);
        fwrite(&lyap,sizeof(double),1,f->outthetas);
        
        p->count2save++;
        p->nsave=0;
    }
}

/**
Mode for MLE simulation

@param  {void*}     pars    Pointer to parameter structure
*/
void mode_lyapunov(void* pars){
    parameters *p = (parameters *) pars;
    
    double *y;
    char file[256];
    
    FILE *outtimes, *outthetas, *outlyap, *outnorm;
    
    // Termalization trajectories, reset and termalization Lyapunov
    rk4_init(p->N, p->h, p->yloc, &dydt);
    
    y=rk4_run_term(&(p->ct),p->tterm,pars);
    
    cublasGetVector(p->N, sizeof(double), y, 1, p->yloc, 1);
    
    rk4_reset(2*p->N,p->yloc,&dydt_tan);
    
    y=rk4_run_term(&(p->ct),p->tterm+p->tterm_lyap,pars);
    
    static_cast<void>(normVec(y+p->N,p->N,p->handle)); // Just normalizing without return
    
    cublasGetVector(2*p->N, sizeof(double), y, 1, p->yloc, 1);
    
    strcpy(file,p->filebase);
    strcat(file, "_MLE.dat");
    if(p->reload_ic){
      outlyap = fopen(file,"ab");
    } else {
      outlyap = fopen(file,"wb");
    }    
    
    if (p->dense>=1){
        strcpy(file,p->filebase);
        strcat(file, "_norm.dat");
        if(p->reload_ic){
          outnorm = fopen(file,"ab");
        } else {
          outnorm = fopen(file,"wb");
        }
    }
    // Run simulation
    if (p->dt==0){
        y=rk4_run_term(&(p->ct), p->tf, pars);
    } else {    
        if (p->dense>=1){
            strcpy(file,p->filebase);
            strcat(file, "_times_mL.dat");
            if(p->reload_ic){
              outtimes = fopen(file,"ab");
            } else {
              outtimes = fopen(file,"wb");
            }

            strcpy(file,p->filebase);
            strcat(file, "_thetas_mL.dat");
            if(p->reload_ic){
              outthetas = fopen(file,"ab");
            } else {
              outthetas = fopen(file,"wb");
            }
            
            double t=p->ct*p->h;
            fwrite(&t,sizeof(double),1,outtimes);
            fwrite(p->yloc,sizeof(double),2*p->N,outthetas);
        }
        
        files_lyap fPointers = {
            .outtimes=outtimes,
            .outthetas=outthetas,
            .outlyap=outlyap,
            .outnorm=outnorm
        };
        
        rk4_run(&(p->ct), p->tf, pars, &fPointers, &step_eval_lyap);
        
        fclose(outlyap);
        if (p->dense>=1){
            fclose(outnorm);
            fclose(outtimes);
            fclose(outthetas);
        }        
        
    }
    
    cublasGetVector(2*p->N, sizeof(double), y, 1, p->yloc, 1);
    
    double tf=p->ct*p->h;
    
    strcpy(file,p->filebase);
    strcat(file,"_fs.dat");
    FILE *outlast=fopen(file,"wb");

    fwrite(p->yloc,sizeof(double),2*p->N,outlast);
    fwrite(&tf,sizeof(double),1,outlast);
    fwrite(&(p->h),sizeof(double),1,outlast);
    fclose(outlast);
}

/**
Mode for the average of lyapunov exponents simulation

@param  {void*}     pars    Pointer to parameter structure
*/
void mode_MLE(void* pars){
    parameters *p = (parameters *) pars;
    
    double *y;
    char file[256];
    
    FILE *outMLE;
    
    // Termalization trajectories, reset and termalization Lyapunov
    
    rk4_init(p->N, p->h, p->yloc, &dydt);
    
    y=rk4_run_term(&(p->ct),p->tterm,pars);
    
    cublasGetVector(p->N, sizeof(double), y, 1, p->yloc, 1);
    
    rk4_reset(2*p->N,p->yloc,&dydt_tan);
    
    y=rk4_run_term(&(p->ct),p->tterm+p->tterm_lyap,pars);
    
    static_cast<void>(normVec(y+p->N,p->N,p->handle)); // Just normalizing without return
    
    cublasGetVector(2*p->N, sizeof(double), y, 1, p->yloc, 1);
    
    strcpy(file,p->filebase);
    strcat(file, "_MLE.dat");
    outMLE = fopen(file,"a");
    
    files_MLE fPointers = {
        .lyap=0,
        .lyap2=0
    };
    
    rk4_run(&(p->ct), p->tf, pars, &fPointers, &step_eval_MLE);
    
    fPointers.lyap/=p->count2save;
    
    double varLyap=fPointers.lyap2/p->count2save-fPointers.lyap*fPointers.lyap;
    
    fprintf(outMLE,"%d %.2f %.12e %.7e\n",p->seed, p->eta,fPointers.lyap,varLyap);
    
    fclose(outMLE);   
    
    // Final state save
    reduceFinalState<<<(p->N+255)/256,256>>>(y,p->N);
    cublasGetVector(2*p->N, sizeof(double), y, 1, p->yloc, 1);
    
    double tf=p->ct*p->h;
    
    strcpy(file,p->filebase);
    strcat(file,"_fs.dat");
    FILE *outlast=fopen(file,"wb");

    fwrite(p->yloc,sizeof(double),2*p->N,outlast);
    fwrite(&tf,sizeof(double),1,outlast);
    fwrite(&(p->h),sizeof(double),1,outlast);
    fclose(outlast);
    
}

/**
Mode for to check for the transition time of the tangent space

@param  {void*}     pars    Pointer to parameter structure
*/
void mode_checkTrans(void* pars){
    parameters *p = (parameters *) pars;
    
    double *y;
    char file[256];
    
    FILE *outthetas;
    
    // Termalization trajectories, reset and termalization Lyapunov
    rk4_init(p->N, p->h, p->yloc, &dydt);
    
    y=rk4_run_term(&(p->ct),p->tterm,pars);
    
    cublasGetVector(p->N, sizeof(double), y, 1, p->yloc, 1);
    
    rk4_reset(2*p->N,p->yloc,&dydt_tan);
    
    y=rk4_run_term(&(p->ct),p->tterm+p->tterm_lyap,pars);
    
    static_cast<void>(normVec(y+p->N,p->N,p->handle));
    
    strcpy(file,p->filebase);
    strcat(file, "_Tout.dat");
    outthetas = fopen(file,"w");
    
    files_cT fPointers = {
        .outthetas=outthetas
    };
    
    rk4_run(&(p->ct), p->tf, pars, &fPointers, &step_eval_checkTrans);
    
    fclose(outthetas);
    
    // Final state save
    cublasGetVector(2*p->N, sizeof(double), y, 1, p->yloc, 1);
    
    double tf=p->ct*p->h;
    
    strcpy(file,p->filebase);
    strcat(file,"_fs.dat");
    FILE *outlast=fopen(file,"wb");

    fwrite(p->yloc,sizeof(double),2*p->N,outlast);
    fwrite(&tf,sizeof(double),1,outlast);
    fwrite(&(p->h),sizeof(double),1,outlast);
    fclose(outlast);
    
}

static void print_help(const char* prog_name){
    printf("Usage: %s [OPTIONS] filebase\n\n", prog_name);
    
    printf("Required:\n");
    printf("  -N, --num N               Number of oscillators\n");
    printf("  -J, --coupling J          Coupling strength coefficient\n");
    printf("  filebase                  Base filename for output\n");

    printf("\nFlags:\n");
    printf("  -h, --help                Show this help message and exits\n");
    printf("  -n, --normal              Normal distributed frequencies          (default: Uniform)\n");
    printf("  -A, --adj                 Stores the full adjacency matrix        (uses more memory)\n");
    printf("  -i, --reload-theta        Reload only thetas as intial conditions from file\n");

    printf("\nReload Options:\n");
    printf("      --reload-ic           Reload full initial conditions from file\n");
    printf("      --reload-freq         Reload frequencies from file\n");
    printf("      --reload-adj          Reload adjacency A from file\n");

    printf("\nSimulation Parameters:\n");
    printf("  -e, --eta eta             Matrix asymmetry parameter [-1,1]               (default: 0.0)\n");
    
    printf("  -I, --amplitude I         Scale for random uniform initial conditions     (default: 1.0)\n");
    printf("  -f, --freq f              Frequency scale                                 (default: 1.0)\n");
    
    printf("  -t, --time t              Total integration time (excl. therm.)           (default: 1e2)\n");
    printf("  -o, --output-step dt      Integration steps between outputs               (default: 10)\n");
    printf("  -d, --timestep h          Integrator timestep                             (default: 1e-2)\n");
    
    printf("  -l, --tau tau             Time between Lyapunov exponent evaluations      (default: 20.0)\n");
    
    printf("  -w, --thterm t            Thermalization time                             (default: 100.0)\n");
    printf("  -W, --thterm-lyap t       Tangent space thermalization time               (default: 500.0)\n");
    
    printf("  -s, --seed s              Random seed                                     (default: 0)\n");

    printf("Mode Parameter (Add other modes if needed):\n");
    printf("  -m, --mode m              Mode selector       (default: 0)\n");
    printf("                              0 = Mode for one run of MLE evaluation.\n");
    printf("                              1 = Mode for multiple runs for the average in MLE evaluations.\n");
    printf("                              2 = Mode for one run to save the full vector.\n");
    
    printf("\nOutput Parameter:\n");
    printf("  -D, --dense d             Output density      (default: 0)\n");
    printf("                              0 = MLE and final state\n");
    printf("                              1 = 0 + thetas\n");
    printf("                              2 = 1 + adjacency\n");
    
    printf("\nExample:\n");
    printf("  %s -N 500 -J 0.5 -e 1 filebase\n", prog_name);
}

/**
Main program for the simulation

@param  {int}   argc    Argument counter
@param  {char*} argv[]  Argument character variables
*/
int main(int argc, char*argv[]){
    // Starts program timer
    struct timeval start, end;
    gettimeofday(&start,NULL);
    
    /***    Program variables (and default declarations)    ***/
    unsigned long int N=1; // Number of oscillators
    double J=0.1; // Coupling constant J
    
    double eta=0.0; // Asymmetry parameter. Defaults set to completely symmetric.
    double tf=1e2; // Final default time
    double h=1e-2; // Integration step
    double tau=20.0;
    double tterm=0.0;  // Final time t1 and integration step
    double tterm_lyap=10.0; // Termalization time for the tangent space...
    int dt=10; // Spacing between outputs (in itegration steps)

    double freq=1.0; // Frequency scale for initial conditions
    double I=1.0; // Uniform scale for initial conditions
    int normal=0; // Uniform frequency distribution
    
    int seed=0; // seed for random  

    int A=0; // Set to False: Does not generate adj matrix

    char* filebase = NULL; // Base name for the simulation files

    int reload_ic=0; // Reload initial conditions, False
    int reload_freq=0; // Reload frequencies, False
    int reload_adj=0; // Reaload adjacency matrix, False
    int ic_start=0; // Reload ONLY thetas, False
    
    int m=0;
    int dense=0;
    
    int opt, option_index = 0; //For the options is better an int here
    int has_N = 0, has_J = 0; // Tracks required arguments 
    
    static struct option long_options[] = {
        {"help",         no_argument,       NULL, 'h'},
        {"normal",       no_argument,       NULL, 'n'},
        {"adj",          no_argument,       NULL, 'A'},
        {"reload-theta", no_argument,       NULL, 'i'},
        {"num",          required_argument, NULL, 'N'},
        {"coupling",     required_argument, NULL, 'J'},
        {"eta",          required_argument, NULL, 'e'},
        {"amplitude",    required_argument, NULL, 'I'},
        {"mode",         required_argument, NULL, 'm'},
        {"freq",         required_argument, NULL, 'f'},
        {"time",         required_argument, NULL, 't'},
        {"output-step",  required_argument, NULL, 'o'},
        {"timestep",     required_argument, NULL, 'd'},
        {"tau",          required_argument, NULL, 'l'},
        {"tterm",        required_argument, NULL, 'w'},
        {"tterm-lyap",   required_argument, NULL, 'W'},
        {"seed",         required_argument, NULL, 's'},
        {"dense",        required_argument, NULL, 'D'},
        {"reload-ic",    no_argument,       NULL,  1 },
        {"reload-freq",  no_argument,       NULL,  2 },
        {"reload-adj",   no_argument,       NULL,  3 },
        {NULL,           0,                 NULL,  0 }
    };

    

    while (optind < argc) {
        opt = getopt_long(argc, argv, "hnAiN:J:e:I:m:f:t:o:d:l:w:W:s:D:R:h",
                          long_options, &option_index);

        if (opt == -1) {
            // The non-option argument (last one) is the filebase
            filebase = argv[optind++];
            continue;
        }

        switch (opt) {
            case 'N': N          = (unsigned long int)atoi(optarg); has_N = 1; break;
            case 'J': J          = atof(optarg);                    has_J = 1; break;
            case 'e': eta        = atof(optarg);                    break;
            case 'I': I          = atof(optarg);                    break;
            case 'm': m          = atoi(optarg);                    break;
            case 'f': freq       = atof(optarg);                    break;
            case 't': tf         = atof(optarg);                    break;
            case 'o': dt         = atoi(optarg);                    break;
            case 'd': h          = atof(optarg);                    break;
            case 'l': tau        = atof(optarg);                    break;
            case 'w': tterm      = atof(optarg);                    break;
            case 'W': tterm_lyap = atof(optarg);                    break;
            case 's': seed       = atoi(optarg);                    break;
            case 'D': dense      = atoi(optarg);                    break;
            case 'A': A          = 1;                               break;
            case 'i': ic_start   = 1;                               break;
            case 'n': normal     = 1;                               break;
            case 1  : reload_ic   = 1;                              break;
            case 2  : reload_freq = 1;                              break;
            case 3  : reload_adj  = 1;                              break;
            case 'h': print_help(argv[0]); return 0;
            case '?':
                fprintf(stderr, "Error: Unknown Option.\nRun '%s --help' for Usage.\n", argv[0]);
                return 1;
        }
    }

    // Check if required options are missing
    int missing = 0;
    if (!has_N)    { fprintf(stderr, "ERROR: -N Number of oscillators is required\n"); missing = 1; }
    if (!has_J)    { fprintf(stderr, "ERROR: -J Coupling strength is required\n");     missing = 1; }
    if (!filebase) { fprintf(stderr, "ERROR: filebase is required\n");                 missing = 1; }

    if (missing) {
        fprintf(stderr, "Run '%s --help' for Usage.\n", argv[0]);
        return 1;
    }
    
    double ti=0; //Starting integration time
    FILE *in;
    char file[256];

    if (dt*h>=tf){
      fprintf(stderr,"ERROR: Output saving bigger than total simulation time. Nothing will be saved!\n ");
      fprintf(stderr,"       If this is desired set -o 0 (dt=0) to simulate without saving.\n ");
      return 1;
    }
    
    tf+=tterm; // Just to set the final time with the thermalization
    
    double *omegasloc, *yloc, *adjloc;
    double *omegas, *adj, *y2, *f2;
    double lyap=0;

    cublasStatus_t stat;
    cublasHandle_t handle;
    cudaSetDevice(0);
    stat=cublasCreate(&handle);
    if (stat!=CUBLAS_STATUS_SUCCESS){
        printf("CUBLAS initialization failed!\n\n");
        return EXIT_FAILURE;
    }
    
    unsigned long int Nmode=2*N; // N oscilators + N tangent space
    tf+=tterm_lyap;
    
    printf("%lu %f %f %f %f %i %f %i\n", N, J, eta, tf, h, dt, tterm+tterm_lyap, seed);
    for (int  i=0; i<argc; i++){
      printf("%s ", argv[i]);
    }
    printf("\n");

    size_t fr, total, req;
    cudaMemGetInfo(&fr,&total);
    if(A){
        req=(15*Nmode+N*N)*sizeof(double); // An overestimation just to be sure by 3*Nmode*sizeof(double)
    } else {
        req=15*Nmode*sizeof(double);
    }
    if(fr < req) {
      fprintf(stderr, "ERROR: Low GPU Memory for specified simulation\n");
      return 1;
    }

    std::mt19937 gen(seed); // Initialize the initial condition random generator
    
    // Allocates memory
    yloc = (double*)malloc(Nmode*sizeof(double));
    cudaMalloc((void**)&y2,2*Nmode*sizeof(double));

    cudaMalloc((void**)&f2,2*Nmode*sizeof(double));

    omegasloc = (double*)malloc(N*sizeof(double));
    cudaMalloc((void**)&omegas,N*sizeof(double));    

    if (A){
        adjloc = (double*)malloc(N*N*sizeof(double));
        cudaMalloc((void**)&adj,N*N*sizeof(double));
    }    
    
    /***    Initializes GPU random seed     ***/
    curandStatePhilox4_32_10_t *state;
    cudaMalloc((void**)&state, sizeof(curandStatePhilox4_32_10_t));
    init_global_state<<<1,1>>>(state, seed);
    
    
    /***    Initial conditions  ***/
    strcpy(file,filebase);
    strcat(file,"_fs.dat");
    
    in = fopen(file,"r");

    if (reload_ic && in != NULL){

        printf("Reloading initial condition from file.\n");
        size_t read_y=fread(yloc,sizeof(double),Nmode,in);

        if (read_y!=Nmode){
            printf("Initial conditions not compatible with N!\n\n");
            return 1;
        } else {
            size_t read=fread(&ti,sizeof(double),1,in);
            read = fread(&h,sizeof(double),1,in);
            
            if (read!=1){
                printf("Couldn't read start time and h\n\n");
                return 1;
            }
        }
        fclose(in);
        printf("Restarting at t=%f with h=%f\n",ti,h);

    } else if (ic_start && in != NULL){

        printf("Reloading initial condition from file.\n");
        size_t read_y=fread(yloc,sizeof(double), N, in);

        if (read_y!=N){
            printf("Initial conditions not compatible with N!\n\n");
            return 1;
        }
        
        fclose(in);
        
        std::normal_distribution<double> distr_tany(0.0,1.0);
        for(int j=0; j<N; j++) {
            yloc[j+N] = distr_tany(gen);
        }

    } else {
        printf("Using random initial conditions.\n");
        std::uniform_real_distribution<double> distr_y(-1.0,1.0);
        std::normal_distribution<double> distr_tany(0.0,1.0);
        for(int j=0; j<N; j++) {
            yloc[j] = 0.5*I*distr_y(gen);
            yloc[j+N] = distr_tany(gen);
        }
    }
    
    strcpy(file,filebase);
    strcat(file,"_freq.dat");
    
    in = fopen(file,"r");

    if (reload_freq && in != NULL){

        printf("Reloading frequencies from file.\n");
        size_t read_freq=fread(omegasloc,sizeof(double),N,in);
        fclose(in);

        if (read_freq!=N){
            printf("Frequencies not compatible with N!\n\n");
            return 1;
        }
        
    } else {
        printf("Using random frequencies.\n");
        if (normal){
            std::normal_distribution<double> distr_freq(0.0,freq);
            for(int j=0; j<N; j++) {
                omegasloc[j] = distr_freq(gen);
            }
        }else{
            std::uniform_real_distribution<double> distr_freq(-freq,freq);
            for (int j=0; j<N; j++){
                omegasloc[j] = distr_freq(gen);
            }
        }
        in=fopen(file,"wb");
        fwrite(omegasloc,sizeof(double),N,in);
        fclose(in);
    }
    cublasSetVector (N, sizeof(double), omegasloc, 1, omegas, 1); // Copies omegas from host into omegas from GPU

    // Start numerical evaluation
    double eS=std::sqrt(0.5*(1.0+eta));
    double eA=std::sqrt(0.5*(1.0-eta));
    double JsqrtN=J/std::sqrt(N);
    
    strcpy(file,filebase);
    strcat(file,"_adj.dat");    
    
    if (A && (in = fopen(file,"r")) && reload_adj==1){
    
        printf("Using adjacency from file\n");
        size_t read_adj=fread(adjloc,sizeof(double),N*N,in);
        fclose(in);
        
        if (read_adj!=N){
            printf("Adjacency matrix not compatible with N!\n\n");
            return 1;
        }
        cublasSetVector(N*N, sizeof(double), adjloc, 1, adj, 1);
    
    } else {
        printf("Using random adjacency matrix\n");
        if(A){
            printf("Generating adjacency matrix from seed.\n");
            
            makeadj<<<(N+255)/256,256>>>(adj, N, eS, eA, state);
            
            if (dense>=2) {
                printf("Saving adjacency matrix.\n");
                cublasGetVector(N*N, sizeof(double), adj, 1, adjloc, 1);
                in=fopen(file,"wb");
                fwrite(adjloc,sizeof(double),N*N,in);
                fclose(in);
            }
        }
    }
    
    fflush(stdout);
    
    parameters pars ={ 
        .N=N, 

        .y2=y2, 
        .f2=f2, 
        .omegas=omegas,  
        .adj=adj, 

        .eta=eta,
        .JsqrtN=JsqrtN,
        .eS=eS,
        .eA=eA,        
                
        .ct=static_cast<unsigned long int>(std::round(ti/h)),
        .dt=dt,
        .ntau=static_cast<int>(std::round(tau/h)),
        .h=h,
        
        .tf=tf,
        .tterm=tterm,
        .tterm_lyap=tterm_lyap,        
        
        .A=A,
        .reload_ic=reload_ic,

        .yloc=yloc,
        .lyap=lyap,
        
        .count2save=0,
        .nsave=0,
        .dense=dense,
        .seed=seed,

        .handle=handle,
        .state=state,
        
        .filebase=filebase
    };
    
    /** Mode Selector (Add any if needed) **/
    if (m==0){
        mode_lyapunov(&pars);
    } else if (m==1) {
        mode_MLE(&pars);
    } else if (m==2) {
        mode_checkTrans(&pars);
    }
    
    /** Print Simulation Timer **/
    gettimeofday(&end,NULL);
    
    float runtime=end.tv_sec-start.tv_sec;
    int runmin=0;
    int runh=0;

    if(runtime>60){
        runmin=runtime/60;
        if(runmin>60){
            runh=runmin/60;
            runmin=runmin%60;
        }
        runtime=std::fmod(runtime,60.0);
    }

    runtime+=1e-6*(end.tv_usec-start.tv_usec);

    printf("\nruntime: %d h %d min %f s \n\n",runh, runmin, runtime);
    fflush(stdout);
    
    /*** Free CPU and GPU memory allocations ***/
    free(yloc);
    cudaFree(y2);
    cudaFree(f2);
    free(omegasloc);
    cudaFree(omegas);

    if (A){
        free(adjloc);
        cudaFree(adj);    
    }
    
    cudaFree(state);
    cublasDestroy(handle);     

    rk4_free();

    return 0;
} 
