# GPT From Scratch
Implementing a Generatively Pretrained Transformer (GPT) based on the 'Attention is All You Need' paper and insights from Andrej Karpathy's lecture


Here is the working :
Step 1: Download the Dataset (a tiny version of Shakespeare's writings)

Step 2: Read the Text File

Step 3: Understand the Characters in the Text
    set(text) finds all unique characters (like letters, punctuation, etc.).
    Then we sort them and count how many there are
    For example, the vocab might include letters like a, b, c... and symbols like , . ! ?.

Step 4: Character Encoding
    Converts characters to numbers and back because neural networks can only work with numbers, not raw text.

Step 5: Convert Text to Tensors
    Turns the entire book into a long list of numbers (characters encoded).

Step 6: Split into Training and Validation Sets (90-10)

Step 7: Prepare Data for Training

Step 8: Batch Preparation
    Instead of training on one piece of text at a time, we train on multiple sequences (a batch).
    Helps the model learn faster.

Step 9: Define the Model
    This class defines a very simple language model:

    It predicts the next character based only on the current one (bigram = two-character relationship).

    It uses nn.Embedding to map characters to predictions.

    It also has a generate method that can generate new text!

