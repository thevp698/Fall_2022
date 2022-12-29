This is the file for MECH-559 Engineering systems optimization final project code

There are multiple files inside this folder, you can find individual files for each sub-systems and three other files for MDO
Our MDO, is designed only for Machine Element analysis and thermal analysis.

1. Machine element Analyis
There is a file called structural_analysis.ipynb where we've done optimization for the this sub-system. There are two different ways we solved the problem
First if using SLSQP algorithm, for this method you define X0 = [ri,ro]
Second algorithm is not giving us results
As a result you'll get the values of the optimized values of x

2. Thermal Analysis
There is a file name thermal_analysis.ipynb where we've done optimization for this sub-system. There are two different ways that we've solved this problem 
First is SLSQP algorithm , for this you define starting x0 = [del_T, r, A]

3. Vibrattion Analysis
For this file you run the code by each cell and you'll get resuls

4. MDO
For this part we've multiple files that are uploaded,
4.1 Black box: in black_box.py file you'll find the code for running the black box optimization function, in this we've calculated the function values that are need to pass throught to the optimized function
4.2 Optimized functions: optimized_functions.py file containes the constraint check of the given functions
4.3 Run: This file is for running the MDO problem
\\\ Note: MDO doen't run.
If you have any questions then feel free to reach at : vraj.patel@mail.mcgill.ca