This python script uses DEAP (Distributed Evolutionary Algorithms) library to perform a genetic algorithm-based feature selection.  


        
The evaluate function serves as the fitness function. It takes an individual "Column" (a binary string representing the feature selection) as input and evaluates its fitness by performing a random mix of mathematical operations on the selected columns (It is interesting to add columns values and shuffle columns to increase performance). The fitness value is calculated as the mean squared error (MSE) between the resulting values and a selected column. Various operators such as addition, multiply, division are registered.  

The script keeps track of the best individual (with the lowest fitness) and its corresponding column selected and operators throughout the evolutionary process. After the evolutionary process is complete, the script prints the results. It displays the individuals in the Hall of Fame (selected columns), the best individual (selected columns), the selected operators used by the best individual, and the fitness obtained by the best individual.  


        
<p align="center">    
Example for a row: The script find the operators between columns        
</p> 
<p align="center">
  <img src="https://github.com/user-attachments/assets/e57bfec2-0330-4f10-b1a8-5a37cede92f2" width="375" height="256" alt="imagen" />
</p>  
<p align="center">
So for row (X): column 2 (4), column 1 (7), column 4 (2) and column 3 (8) = Column Result (5)
</p> 
