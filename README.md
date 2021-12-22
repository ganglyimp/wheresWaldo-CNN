# Where's Waldo CNN
This CNN is a binary classifier that is trained to identify whether or not an image contains an image of Waldo in it. Ideally, you would be able to take a full sized Waldo puzzle, break it down into tiles, and have the neural network be able to identify which tile contains Waldo. 
___
### Some Notes
This CNN uses the [Hey Waldo](https://github.com/vc1492a/Hey-Waldo) dataset.

This program was made as a semester project for a Deep Learning class. Due to lack of time, this network still has some issues. In the Hey-Waldo dataset, there was an imbalance between the "Waldo" and "Not-Waldo" samples (about 80:6000). To rectify this issue, fake "Waldo" samples were created by overlaying a PNG of Waldo over some of the existing "Not-Waldo" samples. This created an overfitting issue where the CNN was great at identifying these fake samples, but struggled to identify actual, unaltered images of Waldo. The network can identify fake Waldo samples with about 90% accuracy, but when given real Waldo samples, the accuracy drops to around 30%. To see the types of images this network guesses correctly and incorrectly, refer to the `guesses_correct` and `guesses_incorrect` folders (TP = "true positive", FP = "false positive", etc). 

___
### Instructions for Running
**waldo.py**
- To run: `python3 waldo.py`
- Outputs average loss and list of correct and incorrect guesses

**waldoGPU.py**
- To run: `python3 waldoGPU.py`
- Uses CUDA calculates for faster output
- Functions exactly the same as waldo.py

**waldoSolver.py**
- To run: `python3 waldoSolver.py`
- Requires user input. Follow instructions from text prompts.
- Runs network on full-sized waldo puzzles. Takes from original-images folder.
- Outputs the tile that the CNN is most confident contains Waldo.
