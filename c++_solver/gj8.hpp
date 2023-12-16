#ifndef GAUSS_JACKSON_HPP
#define GAUSS_JACKSON_HPP

/*
    Gauss- Jackson integrator method for the second order differenttial equation

                                y''= f(t,y,y')

    Method and coefficients taken from: Berry and Healy, Implementation of Gauss-Jackson 
    Integration for Orbit Propagation, J. Astronaut. Sci. 52, 331-357, 2004.
    https://drum.lib.umd.edu/handle/1903/2202

    Method has three parts
      i) A start-up procedure (Runge-Kutta-Fehlberg 78) 
     ii) A predictor (Adams-Bashforth)
    iii) A corrector (Adams-Moulton)

    The start-up procedure populates 4 steps either side of the base epoch using
    an RKF78 method from the ODEINT library. Based on the previous 8 points the
    next point is estimated, if the step is accepted then the state is updated, if
    not it is iterated again until it is within tolerence. 9 values of the y,
    y' and y'' are stored internally and shifted to make room for a new value while
    keeping the array at a constant size. All steps refer to procedure in above reference


     To integrate call GJ8::propagate( state, ode, timeStart, timeEnd, h, output, outputTime)

     Where:

     State: Initial state of y and y'

     ode: Function to be integrated

     timeStart: Initial time point

     timeEnd: Final time point

     h: Step size

     output: Boolean as to whether full output is saved, default is  to 'output.csv'

     outputTime: Request larger steps than h in output if desired. 

*/

#include <iostream>
#include <string>
#include <math.h>
#include <fstream>
#include <boost/numeric/odeint.hpp>
#include <boost/array.hpp>

namespace GJ8{

    const double tolerance = 1e-9; // Tolerance which must be met to allow next value to be calculated
    const int maxIterations = 400; // Maximum number of iterations in start-up and in predictor/corrector
    const std::string outputFileName = "output.csv"; 
    const int outputPrecision=16; 
    const int sizeOfMultiStepArray=9;

    typedef boost::array<boost::array<double, sizeOfMultiStepArray>, sizeOfMultiStepArray+1> Coeff_array; 
    typedef boost::array<double, sizeOfMultiStepArray> GJ8_array;

    // Constants for integration
    // Table 6
    Coeff_array GJ8_a_coeff  =
    {{{
      3250433.0 / 53222400.0,    572741.0 /  5702400.0,    -8701681.0 /  39916800.0,
      4026311.0 / 13305600.0,   -917039.0 /  3193344.0,     7370669.0 /  39916800.0,
      -1025779.0 / 13305600.0,    754331.0 / 39916800.0,     -330157.0 / 159667200.0
    },{
      -330157.0 / 159667200.0,   530113.0 /  6652800.0,      518887.0 /  19958400.0,
      -27631.0 /    623700.0,    44773.0 /  1064448.0,     -531521.0 /  19958400.0,
      109343.0 /   9979200.0,    -1261.0 /   475200.0,       45911.0 / 159667200.0
    },{
        45911.0 / 159667200.0,   -185839.0 / 39916800.0,     171137.0 /   1900800.0,
        73643.0 /  39916800.0,    -25775.0 /  3193344.0,      77597.0 /  13305600.0,
        -98911.0 /  39916800.0,     24173.0 / 39916800.0,      -3499.0 /  53222400.0
    },{
        -3499.0 /  53222400.0,      4387.0 /  4989600.0,     -35039.0 /   4989600.0,
        90817.0 /    950400.0,    -20561.0 /  3193344.0,       2117.0 /   9979200.0,
        2059.0 /   6652800.0,      -317.0 /  2851200.0,        317.0 /  22809600.0
    },{
      317.0 /  22809600.0,     -2539.0 / 13305600.0,      55067.0 /  39916800.0,
      -326911.0 /  39916800.0,     14797.0 /   152064.0,    -326911.0 /  39916800.0,
      55067.0 /  39916800.0,     -2539.0 / 13305600.0,        317.0 /  22809600.0
    },{
      317.0 /  22809600.0,      -317.0 /  2851200.0,       2059.0 /   6652800.0,
      2117.0 /   9979200.0,    -20561.0 /  3193344.0,      90817.0 /    950400.0,
      -35039.0 /   4989600.0,      4387.0 /  4989600.0,      -3499.0 /  53222400.0
    },{
        -3499.0 /  53222400.0,     24173.0 / 39916800.0,     -98911.0 /  39916800.0,
        77597.0 /  13305600.0,    -25775.0 /  3193344.0,      73643.0 /  39916800.0,
        171137.0 /   1900800.0,   -185839.0 / 39916800.0,      45911.0 / 159667200.0
    },{
        45911.0 / 159667200.0,     -1261.0 /   475200.0,     109343.0 /   9979200.0,
        -531521.0 /  19958400.0,     44773.0 /  1064448.0,     -27631.0 /    623700.0,
        518887.0 /  19958400.0,    530113.0 /  6652800.0,    -330157.0 / 159667200.0
    },{
      -330157.0 / 159667200.0,    754331.0 / 39916800.0,   -1025779.0 /  13305600.0,
      7370669.0 /  39916800.0,   -917039.0 /  3193344.0,    4026311.0 /  13305600.0,
      -8701681.0 /  39916800.0,    572741.0 /  5702400.0,    3250433.0 /  53222400.0
    },{
      3250433.0 /  53222400.0, -11011481.0 / 19958400.0,    6322573.0 /   2851200.0,
      -8660609.0 /   1663200.0,  25162927.0 /  3193344.0, -159314453.0 /  19958400.0,
      18071351.0 /   3326400.0, -24115843.0 /  9979200.0,  103798439.0 / 159667200.0
    }}};

    // Table 5
    Coeff_array GJ8_b_coeff  =
    {{{
        19087.0 /     89600.0,   -427487.0 /   725760.0,    3498217.0 /   3628800.0,
        -500327.0 /    403200.0,      6467.0 /     5670.0,   -2616161.0 /   3628800.0,
        24019.0 /     80640.0,   -263077.0 /  3628800.0,       8183.0 /   1036800.0
    },{
       8183.0 /   1036800.0,     57251.0 /   403200.0,   -1106377.0 /   3628800.0,
       218483.0 /    725760.0,       -69.0 /      280.0,     530177.0 /   3628800.0,
       -210359.0 /   3628800.0,      5533.0 /   403200.0,       -425.0 /    290304.0
    },{
       -425.0 /    290304.0,     76453.0 /  3628800.0,       5143.0 /     57600.0,
       -660127.0 /   3628800.0,       661.0 /     5670.0,      -4997.0 /     80640.0,
       83927.0 /   3628800.0,    -19109.0 /  3628800.0,          7.0 /     12800.0
    },{
        7.0 /     12800.0,    -23173.0 /  3628800.0,      29579.0 /    725760.0,
        2497.0 /     57600.0,     -2563.0 /    22680.0,     172993.0 /   3628800.0,
        -6463.0 /    403200.0,      2497.0 /   725760.0,      -2497.0 /   7257600.0
    },{
        -2497.0 /   7257600.0,      1469.0 /   403200.0,     -68119.0 /   3628800.0,
        252769.0 /   3628800.0,                      0.0,    -252769.0 /   3628800.0,
        68119.0 /   3628800.0,     -1469.0 /   403200.0,       2497.0 /   7257600.0
    },{
       2497.0 /   7257600.0,     -2497.0 /   725760.0,       6463.0 /    403200.0,
       -172993.0 /   3628800.0,      2563.0 /    22680.0,      -2497.0 /     57600.0,
       -29579.0 /    725760.0,     23173.0 /  3628800.0,         -7.0 /     12800.0
    },{
        -7.0 /     12800.0,     19109.0 /  3628800.0,     -83927.0 /   3628800.0,
        4997.0 /     80640.0,      -661.0 /     5670.0,     660127.0 /   3628800.0,
        -5143.0 /     57600.0,    -76453.0 /  3628800.0,        425.0 /    290304.0
    },{
         425.0 /    290304.0,     -5533.0 /   403200.0,     210359.0 /   3628800.0,
        -530177.0 /   3628800.0,        69.0 /      280.0,    -218483.0 /    725760.0,
        1106377.0 /   3628800.0,    -57251.0 /   403200.0,      -8183.0 /   1036800.0
    },{
        -8183.0 /   1036800.0,    263077.0 /  3628800.0,     -24019.0 /     80640.0,
        2616161.0 /   3628800.0,     -6467.0 /     5670.0,     500327.0 /    403200.0,
        -3498217.0 /   3628800.0,    427487.0 /   725760.0,     -19087.0 /     89600.0
    },{
        25713.0 /     89600.0,  -9401029.0 /  3628800.0,    5393233.0 /    518400.0,
        -9839609.0 /    403200.0,    167287.0 /     4536.0, -135352319.0 /   3628800.0,
        10219841.0 /    403200.0, -40987771.0 /  3628800.0,    3288521.0 /   1036800.0
    }}};


    template <size_t dimension>
    void calculate_sn_backwards(boost::array<GJ8_array, dimension> &sn, boost::array<GJ8_array, dimension> &f,
                                int startStep, int endStep)
    {
        for(int i = startStep; i > endStep-1;i--){
            for(int j=0;j<dimension;j++){
                sn[j][i]=sn[j][i+1]-(f[j][i+1]+f[j][i])/2;
            }
        }
    }


    template <size_t dimension>
    void calculate_sn_forwards(boost::array<GJ8_array, dimension> &sn, boost::array<GJ8_array, dimension> &f, 
                                int startStep, int endStep)
    {
        for(int i = startStep; i < endStep;i++){
            for(int j=0;j<dimension;j++){
                sn[j][i+1]=sn[j][i]+(f[j][i]+f[j][i+1])/2;
            }
        }  
    }

    template <size_t dimension>
    void calculate_Sn_backwards(boost::array<GJ8_array, dimension> &Sn, boost::array<GJ8_array, dimension> &sn, 
                                boost::array<GJ8_array, dimension> &f, int startStep, int endStep)
    {
        for(int i = startStep; i > endStep-1;i--){
            for(int j=0;j<dimension;j++){
                Sn[j][i] = Sn[j][i+1] - sn[j][i+1] + f[j][i+1] / 2;
            }
        }
    }


    template <size_t dimension>
    void calculate_Sn_forwards(boost::array<GJ8_array, dimension> &Sn, boost::array<GJ8_array, dimension> &sn, 
                                boost::array<GJ8_array, dimension> &f, int startStep, int endStep)
    {
        for(int i = startStep; i < endStep;i++){
            for(int j=0;j<dimension;j++){
                Sn[j][i+1] = Sn[j][i] + sn[j][i] + f[j][i] / 2;
            }
        }
    }

    // Calculates terms of the form: /sum^{4}_{-4} coeff*y''*y''
    template <size_t dimension>
    void calculateCoefficientSum(boost::array<double, dimension> &coefficientSum, boost::array<GJ8_array, dimension> &f,
                                Coeff_array &coeff, int row)
    {
        double sum;
        for (int j = 0; j < dimension; j++){
            sum = 0;
            for (int k = 0; k < row; k++) {
                sum += (coeff[row][k] * f[j][k]);
            }
            coefficientSum[j] = sum;
        }
    }

    // Combination of steps 3bii, 3biii and 3biv to calculate predicted start-up value
    template <size_t dimension, size_t arraySize>
    void updateStartupPrediction(boost::array<boost::array<double, arraySize>, dimension> &prediction, boost::array<GJ8_array,dimension> &f,
                                boost::array<GJ8_array, dimension> &sArray, Coeff_array &coeff, int arrayStart, double timeParameter)
    {
        double sum;
        for (int i = arrayStart; i < sizeOfMultiStepArray; i++) {
            for (int j = 0; j < dimension; j++) {
                sum=0;
                // Step 3bii and 3biii
                for (int k = arrayStart; k < sizeOfMultiStepArray; k++) {
                    sum += (coeff[i][k] * f[j][k]);
                }
                // Step 3biv
                prediction[j][i] = timeParameter * (sArray[j][i] + sum);
            }
        }
    }
    
    // Calculate central S and s terms in start-up step 3a)
    template <size_t dimension>
    void calculateCentralSvalue(boost::array<GJ8_array, dimension> &sArray,  boost::array<GJ8_array,dimension> &y, 
                                boost::array<GJ8_array,dimension> &f, Coeff_array &coeff, double timeParameter)
    {
        for (int i = 0; i < dimension; i++) {
            double sum = 0;
            for (int j = 0; j < sizeOfMultiStepArray; j++){
                sum += coeff[4][j] * f[i][j];
            }
            sArray[i][4] = y[i][4] / timeParameter - sum;
        }    
    }

    // Initiates start-up procedure using odeint stepper do_step method
    template <size_t stateDimension, size_t dimension>
    void populateStartupValues(void (*ODE)(const boost::array<double, stateDimension> &, boost::array<double, stateDimension > &, double), 
                                boost::array<GJ8_array, dimension> &y, boost::array<GJ8_array,dimension> &dydt, boost::array<GJ8_array, dimension> &f,
                                boost::array<double, stateDimension> &state, double time, double h, int step)
    {
        boost::numeric::odeint::runge_kutta_fehlberg78< boost::array< double , stateDimension > > stepper;
        boost::array<double, stateDimension> derivatives;
        boost::array<double,stateDimension> startupState=state;

        int i = 4;
        do{
            stepper.do_step(ODE, startupState, time, h);

            time+=h;
            ODE(startupState, derivatives, time);

            for (int j = 0; j < dimension; j++) {
                y[j][i + step] = startupState[j];
                dydt[j][i + step] = startupState[j + dimension];            
                f[j][i + step] = derivatives[j+ dimension];
            }
            i += step;

        } while (i > 0 && i < 8); 
    }

    // Steps 3bv and 3c) Calculates updated accelerations and tests whether convergence has been met.
    template <size_t stateDimension, size_t dimension>
    bool checkStartupConvergence(void (*ODE)(const boost::array<double, stateDimension> &, boost::array<double, stateDimension > &, double),  
                                boost::array<double, stateDimension> &state, boost::array<GJ8_array, dimension> &y, boost::array<GJ8_array, dimension> &dydt, 
                                boost::array<GJ8_array, dimension> &f, double h)
    {
                // Step 3bv) Evaluate the updated acceleration
                double time = -4*h;
                boost::array<double, stateDimension> derivatives;

                for (int i = 0; i < sizeOfMultiStepArray; i++){
                    for(int j=0; j < dimension; j++){
                        state[j]=y[j][i];
                        state[j+dimension]=dydt[j][i];
                    }

                    ODE(state,derivatives,time);

                    // Step 3c) Test convergances            
                    for(int j=0; j < dimension; j++){
                        if (std::fabs(derivatives[j+dimension] - f[j][i]) > tolerance) {
                                return false;
                        }
                    }
                    time+=h;
                }
                return true;
    }

    template <size_t dimension>
    bool checkCorrectorConvergence(boost::array<GJ8_array, dimension> &y, boost::array<GJ8_array, dimension> &dydt, boost::array<double, dimension> &predictedY, 
                                    boost::array<double, dimension> &predictedDydt)
    {
        // calculate difference between previous value and predicted value
        boost::array<boost::array<double, 2>, dimension> RelativeError;
        for (int j = 0; j < dimension; j++) {
            RelativeError[j][0] = std::fabs(predictedY[j] - y[j][8]);
            RelativeError[j][1] = std::fabs(predictedDydt[j] - dydt[j][8]);
        }

        // return false if any difference is greater than tolerance
        for(int j=0;j<dimension;j++){
            for(int k=0;k<2;k++){
                if(RelativeError[j][k] > tolerance){
                     return false;
                }
            }
        }

        // Update last state with predicted value
        for(int i = 0; i < dimension; i++){
            y[i][8]=predictedY[i];
            dydt[i][8]=predictedDydt[i] ;
        }

        return true;
    }

    // Shifts all elements in passed array to the left. 
    template <size_t dimension>
    void shiftArray(boost::array<GJ8_array, dimension> &state){
        for (int i = 1; i < sizeOfMultiStepArray; i++) {
            for (int j = 0; j < dimension; j++) {
                state[j][i - 1] = state[j][i];
            }
        }
    }

    // Update state vector with values of y & y'
    template<size_t stateDimension, size_t dimension>
    void updateState(boost::array<double, stateDimension> &state, boost::array<GJ8_array, dimension> &y, boost::array<GJ8_array, dimension> &dydt)
    {
        for(int i=0; i < dimension; i++){
            state[i]=y[i][sizeOfMultiStepArray-1];
            state[i+dimension]=dydt[i][sizeOfMultiStepArray-1];
        }
    }

    // Update the output file
    void updateOutput(std::ofstream& outputFile, double* y, double* dydt, double time, int dimension) 
    {
        if(outputFile.is_open()){
            outputFile << time<< ",";
            for(int i=0;i<dimension;i++){
                outputFile << std::setprecision(outputPrecision)<<*(y+sizeOfMultiStepArray*i) << ",";
            }
            for(int i=0;i<dimension;i++){
                outputFile << std::setprecision(outputPrecision)<<*(dydt+sizeOfMultiStepArray*i) << ",";
            }
            outputFile << "\n";
        }
    
        else{
             outputFile.open(outputFileName);
             outputFile << "time"<< ",";
            for(int i=0; i<dimension;i++){
                outputFile<< "y" << (i+1) << ",";
            }
            for(int i=0; i<dimension;i++){
                outputFile<< "y'" << (i+1) << ",";
            }
            outputFile << "\n";
            updateOutput(outputFile, y, dydt, time, dimension);
        }

    }

    // Before the predictor corrector is run it is necessary to generate 4 points either side of epoch
    // If start-up is unable to converge it will return false and programme will end.
    template <size_t stateDimension, size_t dimension>
    bool startupProcedure(boost::array<double,stateDimension> &state, double T0, void (*ODE)(const boost::array< double, stateDimension> &, boost::array< double , stateDimension> &, double),
                            boost::array<GJ8_array, dimension> &y, boost::array<GJ8_array, dimension> &dydt, boost::array<GJ8_array, dimension> &f, boost::array<GJ8_array, dimension> &sn,
                            boost::array<GJ8_array, dimension> &Sn, double h)
    {
        boost::array<GJ8_array, dimension> predictedY;
        boost::array<GJ8_array, dimension> predictedDydt;          
        boost::array<double, stateDimension> derivatives;
        int iteration = 0;
        double h2 = h * h;
        
        // Calculate the initial derivatives at epoch 
        ODE(state,derivatives,T0);

        // Get initial y, y' and y''
        for(int i = 0; i < dimension; i++){
            y[i][4]=state[i];
            dydt[i][4]=state[i+dimension];
            f[i][4]=derivatives[i+dimension]; 
        }

        // Step 1 and 2: Populate y, y' and y'' at 8 points around the initial time
        populateStartupValues(ODE, y, dydt, f, state, T0, -h, -1); // 4 steps backwards
        populateStartupValues(ODE, y, dydt, f, state, T0, h, 1); // 4 steps forwards

        while (iteration <= maxIterations) { // While y'' not converged
            iteration++;

            // Return false if too many itterations taken
            if (iteration == maxIterations) {
                std::cout << "Error in Start-up: Maximum number of iterations reached, tolerance may be too strict, exiting"<< std::endl;
                return false;
            }

            // Step 3a) Calculate s0 and S0
            calculateCentralSvalue(sn, dydt, f, GJ8_b_coeff, h); // s0
            calculateCentralSvalue(Sn, y, f, GJ8_a_coeff, h2); // S0

            // Step 3bi) Calculate sn and Sn for n= +- 1,2,3,4
            calculate_sn_backwards(sn,f,3,0); // sn=-1:-4
            calculate_sn_forwards(sn,f,4,8); // sn=1:4
            calculate_Sn_backwards(Sn, sn, f, 3, 0); // Sn=-1:-4
            calculate_Sn_forwards(Sn, sn, f, 4, 8); // Sn=1:4
            
            // Step 3bii, 3biii, 3biv) Calculate prediction of y and y'
            updateStartupPrediction(predictedDydt, f, sn, GJ8_b_coeff,0, h);
            updateStartupPrediction(predictedY, f, Sn, GJ8_a_coeff, 0, h2);

            // Step 3bv and 3c) Calculate y'' and run convergence test. Return true if
            // converged, else use new predictions and recalculate
            if(checkStartupConvergence(ODE, state, predictedY, predictedDydt, f, h)){
                return true;
            } 
            else{
                y = predictedY;
                dydt = predictedDydt;
            }
        
        } // close while loop
        return true;
    }

    // Main part of the method
    template <size_t stateDimension, size_t dimension>
    void predictorCorrector(void (*ODE)(const boost::array< double, stateDimension> &, boost::array< double, stateDimension> &, double),
                            boost::array<GJ8_array, dimension> &y, boost::array<GJ8_array, dimension> &dydt, boost::array<GJ8_array, dimension> &f, 
                            boost::array<GJ8_array, dimension> &sn, boost::array<GJ8_array, dimension> &Sn, double &time, double h, int numberOfSteps, 
                            bool output, double outputTime, std::ofstream& outputFile)
        {
        boost::array<double,stateDimension> state;
        boost::array<double, stateDimension> derivatives;
        int iteration;
        double h2 = h * h;
        int outputStep = round(outputTime/h);

        // update output from start-up is necessary and get time to the correct value
        for(int i = 4; i<sizeOfMultiStepArray; i++){            
            if(output && (i-4)%outputStep==0 ){
                updateOutput(outputFile, &y[0][i], &dydt[0][i], time, dimension);
            }      
            time+=h; // increment time to the 5th h step
        }

        // Main loop
        for (int i = 9; i < numberOfSteps+4; i++) { //main predictor corrector for loop start
            // Steps 5 and 6) Caclculate sum term of previous y'' values with last row of coefficients for predictor
            boost::array<double, dimension> aCoeffSumTerm5;
            boost::array<double, dimension> bCoeffSumTerm5;
            calculateCoefficientSum(aCoeffSumTerm5, f, GJ8_a_coeff, sizeOfMultiStepArray);
            calculateCoefficientSum(bCoeffSumTerm5, f, GJ8_b_coeff, sizeOfMultiStepArray);

            //------------------- Predictor -------------------

            shiftArray(Sn);
            shiftArray(sn);
            shiftArray(y);
            shiftArray(dydt);

            // Steps 4 and 7
            for (int j = 0; j < dimension; j++) {
                Sn[j][8] += sn[j][8] + f[j][8] / 2; // Predict Sn+1
                y[j][8] = h2 * (Sn[j][8] + aCoeffSumTerm5[j]); // calculate r+1
                dydt[j][8] = h * (sn[j][8] +f[j][8] / 2 + bCoeffSumTerm5[j]); // calculate rdot+1
            }

            //------------------- Corrector -------------------

            shiftArray(f);

            updateState(state, y, dydt);

            ODE(state,derivatives,time);

            // Step 8) Calculate new y''          
            for(int j=0; j < dimension; j++){
                f[j][8] = derivatives[j+dimension];
            }
        
            // Step 10bi and 10bii) Caclculate sum term of previous y'' values with penultimate row of GJ coefficients for corrector
            boost::array<double, dimension> aCoeffSumTerm4;
            boost::array<double, dimension> bCoeffSumTerm4;
            calculateCoefficientSum(aCoeffSumTerm4, f, GJ8_a_coeff, sizeOfMultiStepArray-1);
            calculateCoefficientSum(bCoeffSumTerm4, f, GJ8_b_coeff, sizeOfMultiStepArray-1);
            
            iteration = 0;

            while (iteration < maxIterations) {
                iteration++;

                // return if corrector is ran too much for an iteration
                if (iteration == maxIterations) {
                    std::cout << "Error: Maximum number of iterations reached, tolerance may be too strict, state not updated, exiting"<< std::endl;
                    if(output){
                        outputFile.close();
                    }
                    return;
                }
                
                // Step 10a) calculate sn
                calculate_sn_forwards(sn,f,7,8);

                boost::array<double, dimension> predictedY;
                boost::array<double, dimension> predictedDydt; 

                // Step 10d) Calculated predicted y and y''
                for (int j = 0; j < dimension; j++) {
                    predictedY[j] = h2 * (Sn[j][8] + aCoeffSumTerm4[j] +  GJ8_a_coeff[8][8] * f[j][8]);
                    predictedDydt[j] = h * (sn[j][8] + bCoeffSumTerm4[j] +  GJ8_b_coeff[8][8] * f[j][8]);
                }

                // Steo 10e) Test convergence of corrector 
                if (checkCorrectorConvergence(y, dydt, predictedY,predictedDydt)) {      
                    
                    if(output && (i-4)%outputStep==0 ){
                        updateOutput( outputFile, &y[0][8], &dydt[0][8], time, dimension);
                    }
                    time += h;
                    break; // Increment time and go to next step.
                } 

                // If convergence not passed, recalulate acceleration with new predicted positions
                for (int j = 0; j < dimension; j++) {
                    y[j][8] = predictedY[j];
                    dydt[j][8] = predictedDydt[j];
                }

                updateState(state, y, dydt);

                ODE(state,derivatives,time);

                for(int j=0; j < dimension; j++){
                    f[j][8] = derivatives[j+dimension];
                }

            } //while loop closed
                
        } // Number of steps For loop closed

        return;
    }

    // Main function call propagate( state, ode, timeStart, timeEnd, h, output, outputTime)
    template <size_t stateDimension>
    void propagate(boost::array<double,stateDimension> &state, void (*ODE)(const boost::array< double, stateDimension> &, boost::array< double,
        stateDimension> &, double), double timeStart, double timeEnd, double h, bool output = 0, double outputTime=1.0)
    {	
        std::ofstream outputFile;
        
        double time = timeStart;
        const int numberOfSteps = (int)((timeEnd-timeStart) / h)+1;

        // Define arrays common to both start-up and predicictor corrector
        boost::array<GJ8_array, stateDimension/2> y;    // y
        boost::array<GJ8_array, stateDimension/2> dydt; // y'
        boost::array<GJ8_array, stateDimension/2> f;    // y''
        boost::array<GJ8_array, stateDimension/2> sn;
        boost::array<GJ8_array, stateDimension/2> Sn;

        // Main calculation, if start-up succesfull then proceed with predictor-corrector. 
        if(startupProcedure(state, timeStart, ODE, y,dydt, f, sn, Sn, h)){
            
            predictorCorrector(ODE, y, dydt, f, sn, Sn, time, h, numberOfSteps, output, outputTime, outputFile);
            
            // Update state with last calculated values for y and y'
            updateState(state, y, dydt);

        } 

        return;
    
    } // end of Propagate
} // End namespace

#endif
