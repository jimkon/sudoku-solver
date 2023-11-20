# Visual Sudoku puzzle solver

![full solution.gif](res%2Ffull%20solution.gif)

This is a Proof-of-Concept attempt to detect and extract a Sudoku Puzzle from an image by using standard Computer
Vision methods. No Supervised Machine Learning methods are used and that's intentional. Goal was to experiment with
Open-CV and the standard CV techniques for de-noising and processing and image, recognize patters and provide 
visualisation.

There are 5 stages:
* #### Clear the image from the noise so only the sudoku puzzle is left in the image
* #### Find the pattern of the grid and extract the sudoku with the digits
* #### Locate and identify the digits 
* #### Find the solution of the sudoku (out of scope for this project)
* #### Present the solution back to the original image

For more details about the steps and the code see the "poc draft" notebook.