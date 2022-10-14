#include <stdio.h>
#include <math.h>

#define TAYLOR_COEFFS 20

double sine_taylor(double in) {
    double result;           // final result
    double term;             // untermediate term for each iter
    double power = in;       // base case
    double factorial = 1;    // base case
    
    double power_iter = in * in;
    double factorial_iter;
    
    for (unsigned int i = 0; i < TAYLOR_COEFFS; i++) {
        
        term = power / factorial;
        
        power = power * power_iter;
        factorial_iter = factorial * (i + 2) * (i + 3);
        factorial = factorial * factorial_iter;
        printf ("power:     %f\n", power);
        printf ("factorial: %f\n", factorial);
        printf ("term:      %f\n", term * 1e32);
        
        if (i & 0x01)   // is odd
            result -= term;
        else            // is even
            result += term;
    }
    
    return result;
}

int main () {
    for (double i = 0.0000001; i < 100; i *= 10) {
        double my_answer = sine_taylor(i);
        double reference = sin(i);

        printf ("ref: %f\n", reference);
        printf ("my : %f\n", my_answer);
        printf ("-----------------\n");
    }
}