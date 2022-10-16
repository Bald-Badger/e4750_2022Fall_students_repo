#include <stdio.h>
#include <math.h>

#define TAYLOR_COEFFS 15

float sine_taylor(float in) {
    float result;           // final result
    float term;             // untermediate term for each iter
    float power = in;       // base case
    float factorial = 1;    // base case
    
    float power_iter = in * in;
    float factorial_iter;
    
    for (unsigned int i = 0; i < TAYLOR_COEFFS; i++) {
        
        term = power / factorial;
        
        power = power * power_iter;
        factorial_iter = (i + 2) * (i + 3);
        factorial = factorial * factorial_iter;
        printf ("power:     %f\n", power);
        printf ("factorial: %f\n", factorial);
        printf ("term:      %f\n", term);
        printf ("\n");
        
        if (i & 0x01)   // is odd
            result -= term;
        else            // is even
            result += term;
    }
    return result;
}

int main () {
    for (float i = 0.000000001; i < 1; i *= 10) {
        float my_answer = sine_taylor(i);
        float reference = sin(i);
        printf ("calculating i = %f\n", i);
        printf ("ref: %f\n", reference);
        printf ("my : %f\n", my_answer);
        printf ("-----------------\n");
    }
}