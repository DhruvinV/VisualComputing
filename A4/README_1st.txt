Assignment #4
CSC320 Fall 2019
Kyros Kutulakos

Notes on the starter code for the PatchMatch algorithm

---------------
GENERAL REMARKS
---------------

A. STARTER CODE

  The top-level python executable is

     code/viscomp.py

  You can run it by calling 

     cd code
	 python viscomp.py --help
	
  to see its input arguments.

  Check the directory test_images/jaguar2/README.txt for an example of how to
  run the code for one of the supplied datasets.

B. GETTING FAMILIAR WITH THE ALGORITHM'S RESULTS

  I am supplying the full result of running the reference solution on CDF,
  including all the intermediate results, for the dataset test_images/jaguar2.
  Get the algorithm's output after each iteration to see what NNF it computes,
  how it improves after each iteration and how the reconstructed source image
  improves as well. Since these results take up a lot of space, I am 
  putting them in a separate file called 
  
      a4_results.tar.gz

C. GETTING FAMILIAR WITH THE STARTER CODE

  1. Unlike A3, the code cannot be executed out of the box: you need to complete
  the helper functions in algorithms.py before you can run the code.

  2. Once you've completed those functions, it is possible to begin working on 
  extending your A3 implementation of propagate_and_random_search()

  3. Try testing your code on the jaguar2/ dataset. The best-NNF (ie. self._f_k[0])
  should give results very similar to what you got for A3.


D. COMPLETING THE STARTER CODE

  1. You should start by implementing the helper functions. Then work on
  extending propagate_and_random_search() to make it work with heaps.

  2. Once you're convinced that function is working correctly, implement the
  nlm algorithm. This algorihm takes up less than 15 lines of code in the 
  reference implementation and relies heavily on other functions in the code.

---------------------
STRUCTURE OF THE CODE
---------------------

1. GENERAL NOTES

  * code/viscomp.py
       top-level routine 

2. IMPORTANT: 

  We will be running scripts to test your code automatically. To 
  ensure proper handling and marking, observe the following:

  * All your code should go in the following files:
  		code/algorithm.py 
        code/patchMatch.py
  * The only modification you will do to patchMatch.py
    is to transfer the read/write functionality you implemented in A1 as these
    are not provided in the A4 starting code either.
  * Do not modify any other python files
  * Do not modify any parts of the above files except where specified
  * Do not add any extra files or directories


3. GENERAL STRUCTURE

  The implementation centers on a single class called
  PatchMatch, defined in patchMatch.py. An instance of this
  class is created when the program is first run. It 
  contains private variables that hold all the input and
  output images, methods for reading/writing those
  variables from/to files, etc.


4. FILES IN THE DIRECTORY code/

   nnf.py
   			Start from here, after reading the paper and running the
   			starter implementation. This file describes how
			nearest-neighbor fields are represented. 
			The file is heavily commented so study it carefully. 
			Functions to look at in this file are:
    			* init_NNF(): the most important function, 
				  which creates the NNF.
    			* create_NNF_image(): look at this file to understand 
				  how NNFs are visualized as a color image. You will need 
				  to understand this function very well because this is 
				  the main mechanism you have to debug your algorithm, 
				  and to know whether the field you compute is what 
				  you expect

   algorithm.py	
			This is where your implementation goes. The functions 
			you must implement are fully specified here. In addition
			to those functions, you must read carefully and understand
			the following two utility functions:
			   * make_patch_matrix(): This function takes an image
			     and generates a matrix of patches. You will need to
			     understand the structure of the patch matrix since
			     you will be processing its contents in your 
			     implementation. 
			   * make_coordinate_matrix(): another extremely important
			     function that you should use as much as possible in
			     your code. Be sure to understand how loops are
			     avoided with the help of functions that operate on
			     matrices.
			
   patchMatch.py
            Implementation of the PatchMatch class. This provides the
 			data structure that contains private variables used by the
			method. Look at the method's initializer to see the default
			values of the various parameters. This is where you should
			place your A1 code for reading/writing images.
    
   viscomp.py
            Code for processing arguments and creating an instance of the
			PatchMatch class. You do not need to look at this file.

      
