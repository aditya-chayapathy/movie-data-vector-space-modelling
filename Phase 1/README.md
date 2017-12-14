# Vector space modeling of MovieLens & IMDB movie data - Phase 1

#### Introduction:
The tasks involve representing relationships between different entities of the data set. The data set contains information taken from MovieLens and IMDB, pertaining to movies, genres, actors, users and the tags associated by the user to the movies, provided through csv files.

#### Software requirements:
Python 2.7.13 :: Anaconda 4.4.0 (64-bit)

#### Directory Structure:
The project directory structure has the following directories:
	1. "resources" - contains the csv files that constitute the data set. 
	2. "scripts" - contains command line interface along with the other supporting scripts needed for the successful execution of the project.

#### Execution Steps:
```
Help: This describes how to use the command line interface
Usage: python <command-line-interface> --help
Example: python print_genre_vector.py --help

Task 1:
Command line interface - print_actor_vector.py
Usage: python print_actor_vector.py <actor_id> <model>
Example: python print_actor_vector.py 579260 tf

Task 2:
Command line interface - print_genre_vector.py
Usage: python print_genre_vector.py <genre> <model>
Example: python print_genre_vector.py Animation tf

Task 3:
Command line interface - print_user_vector.py
Usage: python print_user_vector.py <user_id> <model>
Example: python print_user_vector.py 109 tf

Task 4:
Command line interface - differentiate_genre.py
Usage: differentiate_genre.py <genre1> <genre2> <model>
Example: python differentiate_genre.py Children Thriller tf
```

#### Troubleshooting:
1. The tag weight calculations are performed dynamically whenever the input is passed to the command line interface. Also, this project uses in-memory data frames (via python pandas library) for storage and retrieval. You may observe delay in the output of the command line interface based on the input. Please be patient.
2. Please ensure the data set (csv files) have the same names and column descriptors as the sample data set for correct execution.
3. Ensure you are running the correct python interpreter. The correct interpreter will give the following output on the command line:
	python --version
	Python 2.7.13 :: Anaconda 4.4.0 (64-bit)
