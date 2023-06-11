# CS 449 Final Project

This re-implementation of "Attention Is All You Need" was a final group project for the Deep Learning (CS449) class in Spring 2023 at Northwestern University. The three-member group consisted of students from the McCormick School of Engineering and the Kellogg School of Management.

While the goal was to re-implement as much of the paper from scratch as possible, time and resource constraints forced us to make tradeoffs with our focus. We implemented all of the multiheaded attention mechanism (the core of the paper) as well as encoder and decoder components. However, we used off-the-shelf tokenizers rather than implementing them on our own, which would have been an interesting exploration in itself but would have taken our attention (no pun intended) away from the core aspects of the paper.

Our findings and analyses can be found in `report.pdf`.

# Set up

- Create a virtual environment and activate it
- Install all required Python libraries: `pip install -r requirements.txt`

# Train model

We trained the model using Google Cloud's Deep Learning VM; however, you can also train locally. Run the following command to train:
`python -m src.transformer_runner <num_examples> <batch_size> <max_length> <n_epochs> <train (1 | 0)> <language_pair (fr-en | de-en)> <loss (mse | cross_entropy)>`

For example, to train with 1000 examples with batch size of 100, sequence length of 50, and for 25 epochs on the French-English translation task using Cross Entropy Loss, run the following command:

`python -m src.transformer_runner.py 1000 100 50 25 1 fr-en cross_entropy`
