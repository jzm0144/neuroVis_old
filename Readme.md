
This project uses Python=3.6

Please install all the required Libariers for the project using the following command.

    $ pip install -r requirements.txt 

Please run the file main.py with the four arguments: the dataset, the brain disorder, number of paths, the class id

<h2>Here is an example run:</h2>

    $ python main.py ageMatchUnmatch disorder dataExample topPaths classID
	$ python main.py ageMatched         PTSD      13         10       0



<h3>Part1 Results:</h3>
    Part1 results are in Results/Part1/PTSD/. Generates specific explanation for each heatmap method and each example.


<h3>Part2 Results:</h3>
    Part2 results are in Results/Part2/PTSD/. In Part2 experiment we calculate an avg of each Heatmap Method over all data examples. It generates a 1 final image (avg of heatmaps of all examples) for each heatmap method.

<h3>Part3 Results:</h3>
    Part3 results are in Results/Part3/PTSD/. It generates an avg of All Heatmaps for a given example. So this kind of explanation takes the opinion of each heatmap method but in a very naive way.

<h3>Part4 Results:</h3>
    Part4 results are in Results/Part4/PTSD/. This experiment calculates an explanation based on the opinion of all heatmaps.

    --------- step1: Calc all Heatmaps for the Same Example
    --------- step2: Sparsify: Let only top-X paths pass thruough
    --------- step3: Calc Binary Intersection of path occurences
    --------- step4: Calc Mean and Element-wise multiply with Binary Intersetion