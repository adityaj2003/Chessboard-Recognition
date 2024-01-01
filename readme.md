# Chess Board Analysis and Piece Classification Script

## Overview
This script is designed for advanced image processing and chess piece classification. It uses OpenCV, NumPy, SciPy, and TensorFlow with the Keras API to process images of chess boards, identify individual squares, and classify the chess pieces on each square. The output is a Forsyth-Edwards Notation (FEN) string representing the board's current state.

## Features
- **Canny Edge Detection**: Identifies edges in the chess board images.
- **Hough Line Transformation**: Detects lines to determine the grid of the chess board.
- **Intersection Points Calculation**: Finds intersection points of lines to locate individual squares.
- **Perspective Transformation**: Adjusts image perspective for accurate piece identification.
- **Piece Classification**: Uses pretrained models (`InceptionV3` and `ResNet`) to classify pieces on the board.
- **FEN String Generation**: Converts the classification results into a FEN string.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- SciPy
- TensorFlow 2.x
- Keras

## Usage
1. Place the chess board images in the same directory as the script.
2. Ensure that the required Python libraries are installed.
3. Run the script. It will process the images and output the FEN strings along with classification accuracies.

I gratefully acknowledge the creators of these datasets for making them publicly available, which has been instrumental in the development and performance of our piece classification models.

@article{mallasen2020LiveChess2FEN,
  title = {LiveChess2FEN: A Framework for Classifying Chess Pieces Based on CNNs},
  author = {Mallas{\'e}n Quintana, David and Del Barrio Garc{\'i}a, Alberto Antonio and Prieto Mat{\'i}as, Manuel},
  year = {2020},
  month = dec,
  journal = {arXiv:2012.06858 [cs]},
  eprint = {2012.06858},
  eprinttype = {arxiv},
  url = {http://arxiv.org/abs/2012.06858},
  archiveprefix = {arXiv}
}
