# Vector space modeling of MovieLens & IMDB movie data - Phase 3

#### Software requirements:
1. Python 3.6.2 :: Anaconda, Inc.
2. Install gensim for LDA. Run the below command in anaconda command prompt
conda install -c anaconda gensim
Run anaconda prompt as administrator if you encounter permission issues
3. Install tensorly package for CP decomposition.

#### Directory Structure:
The project directory structure has the following directories:
1. "resources" - contains the csv files that constitute the data set. 
2. "scripts" - contains command line interface along with the other supporting scripts needed for the successful execution of the project.

#### Execution Steps:
```
Help: This describes how to use the command line interface
Usage: python <command-line-interface> --help
Example: python phase_3_task_4.py --help

Task 1:
Command line interface - phase_3_task_1.py
Usage: python phase_3_task_1.py user_id model
Example: python phase_3_task_1.py 3 SVD

Task 2:
Command line interface - phase_3_task_2.py
Usage: python phase_3_task_2.py
Example: python phase_3_task_2.py

Task 3:
Command line interface - phase_3_task_3.py
Usage: python phase_3_task_3.py <num_layers> <num_hashs_per_layer>
Example: python phase_3_task_3.py 2 3

Task 4:
Command line interface - phase_3_task_4.py
Usage: phase_3_task_4.py
Example: python phase_3_task_4.py

Task 5:
Command line interface - phase_3_task_5.py
Usage: phase_3_task_5.py <model>
Example: python phase_3_task_5.py RNN
```

#### Troubleshooting:
1. The tag weight calculations are performed dynamically whenever the input is passed to the command line interface. Also, this project uses in-memory data frames (via python pandas library) for storage and retrieval. You may observe delay in the output of the command line interface based on the input. Please be patient.
2. Please ensure the data set (csv files) have the same names and column descriptors as the sample data set for correct execution.
3. For task 5, please ensure that the movie labels are provided through "label_movies.json" as per the format specified in the sample file.
4. Ensure you are running the correct python interpreter. The correct interpreter will give the following output on the command line:
	 python --version
	 Python 3.6.2 :: Anaconda, Inc.
