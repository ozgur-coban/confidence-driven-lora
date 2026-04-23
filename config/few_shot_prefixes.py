FEW_SHOT_PREFIXES = {
    "math": """Example 1:
Input: Blake and Kelly are having a contest to see who can run the most in 15 minutes. They decide to do it on a football field that is 100 yards long. Blake runs back and forth 15 times. Kelly runs back and forth once, and then decides that she doesn't want to run next to Blake, so she starts to run to the 40-yard line and back. She does this 34 times. How much farther does the winner run than the loser?
Answer: How many yards does Blake run? ** Blake runs 200 yards for each back and forth because 100 x 2 = 200
How many yards does Blake run? ** He runs 3000 yards in total because 200 x 15 = <<200*15=3000>>3000
How many yards does Kelly run? ** Kelly runs 200 yards at first because 100 x 2 = <<100*2=200>>200
How many yards does Kelly run? ** She runs 80 yards each time she does a back and forth to the 40-yard line because 40 x 2 = <<40*2=80>>80
How many yards does Kelly run? ** She runs 2720 from these back and forth because 34 x 80 = <<34*80=2720>>2720
How many yards does Kelly run? ** She runs 2920 in total because 2720 + 200 = <<2720+200=2920>>2920
How much farther does the winner run than the loser? ** Blake won because 3000 > 2920
How much farther does the winner run than the loser? ** The winner ran 80 yards more than the loser because 3000 - 2920 = <<3000-2920=80>>80
#### 80

Example 2:
Input: John can mow a lawn in 3 hours. Mary can mow the same lawn in 6 hours. If they work together for 1 hour, what fraction of the lawn is left to mow?
Answer: What is the total work in units? ** Let the total work be the least common multiple of 3 and 6, which is 6 units.
How many units does John mow per hour? ** John mows 2 units per hour because 6 / 3 = <<6/3=2>>2
How many units does Mary mow per hour? ** Mary mows 1 unit per hour because 6 / 6 = <<6/6=1>>1
How many units do they mow together in an hour? ** Together they mow 3 units per hour because 2 + 1 = <<2+1=3>>3
How many units are left after 1 hour? ** There are 3 units left because 6 - 3 = <<6-3=3>>3
What fraction of the lawn is left? ** The fraction left is 3/6, which simplifies to 1/2 because 3 / 6 = <<3/6=0.5>>0.5
#### 0.5

Example 3:
Input: Mark took a test yesterday that consisted of 75 questions. He completed the test at a rate of 5 questions per hour. Today, he took another test of 100 questions at the same rate. If Mark had 8 hours to complete the first test and 6 hours to complete the second one, how many questions did he leave incomplete?
Answer: How many questions did Mark complete in yesterday’s test? ** Questions Mark completed in yesterday’s test: 5*8 = <<5*8=40>>40.
How many questions did Mark complete in today's test? ** Questions Mark completed in today’s test: 5*6 = <<5*6=30>>30.
How many questions did Mark leave incomplete in yesterday’s test? ** Questions Mark left incomplete in yesterday’s test: 75-40 = <<75-40=35>>35
How many questions did Mark leave incomplete in today's test? ** Questions Mark left incomplete in today's test: 100-30 = <<100-30=70>>70
How many questions did Mark leave incomplete in total? ** Missed questions in total 35+70 = <<35+70=105>>105.
#### 105
""",
    "code": """Example 1:
Input: Write a function to generate the nth Fibonacci number. The function should have a time complexity of O(log n) and use dynamic programming. Additionally, the function should only use a constant amount of space, without using any arrays or additional data structures.
Answer: To generate the nth Fibonacci number with a time complexity of O(log n) and using dynamic programming with a constant amount of space, we can utilize the concept of matrix exponentiation.

The Fibonacci sequence can be represented as a matrix equation:

```
| F(n+1)  F(n) |   =   | 1  1 |   ^   n
|  F(n)   F(n-1)|       | 1  0 |
```

By raising the matrix to the power of n, we can obtain the nth Fibonacci number.

Here is the Python code that implements this approach:

```python
def multiply_matrices(A, B):
    a, b, c = A[0][0], A[0][1], A[1][0]
    d, e, f = B[0][0], B[0][1], B[1][0]
    
    return [
        [a*d + b*e, a*e + b*f],
        [c*d + e*d, c*e + e*f]
    ]


def power_matrix(A, n):
    if n == 0:
        return [[1, 0], [0, 1]]  # Identity matrix
    
    result = power_matrix(A, n // 2)
    result = multiply_matrices(result, result)
    
    if n % 2 == 1:
        result = multiply_matrices(result, A)
    
    return result


def fibonacci(n):
    if n == 0:
        return 0
    
    matrix = [[1, 1], [1, 0]]
    result = power_matrix(matrix, n - 1)
    
    return result[0][0]
```

In this implementation, the `multiply_matrices` function multiplies two 2x2 matrices, and the `power_matrix` function raises a matrix to the power of n using matrix exponentiation. The `fibonacci` function returns the nth Fibonacci number by calling `power_matrix` with the base matrix [[1, 1], [1, 0]] raised to the power of n - 1.

Since the matrix multiplication and exponentiation operations have time complexities of O(1), and the recursive function `power_matrix` divides the problem size in half with each recursive call, the overall time complexity of this approach is O(log n). Additionally, this implementation uses a constant amount of space as it only requires a few variables to store intermediate matrix products and results.

Example 2:
Input: Write a Python function that takes two lists of equal length and creates a dictionary of the elements in each list that have the same index. The function should also handle cases where the input lists are not of equal length and provide appropriate error handling. Additionally, the function should handle cases where the input lists contain nested lists or dictionaries, and create a dictionary of the nested elements at each index. The function should also handle cases where the input lists contain tuples and create a dictionary of the tuple elements at each index. The function should raise a ValueError if any element in the input lists is a set. 

list_one = [1, 2, [3, 4], {"a": "b"}, (7, 8), {9, 10}]
list_two = ["a", "b", [5, 6], {"c": "d"}, (11, 12), {13, 14}]

def create_index_dictionary(list_one, list_two):
    if len(list_one) != len(list_two):
        raise ValueError("Input lists must be of equal length")
    
    index_dict = {}
    for i in range(len(list_one)):
        if isinstance(list_one[i], (list, dict, tuple)) and isinstance(list_two[i], (list, dict, tuple)):
            if isinstance(list_one[i], set) or isinstance(list_two[i], set):
                raise ValueError("Input lists cannot contain sets")
            
            nested_dict = {}
            if isinstance(list_one[i], dict) and isinstance(list_two[i], dict):
                for key in list_one[i].keys():
                    nested_dict[key] = (list_one[i][key], list_two[i][key])
            elif isinstance(list_one[i], tuple) and isinstance(list_two[i], tuple):
                for j in range(len(list_one[i])):
                    nested_dict[j] = (list_one[i][j], list_two[i][j])
            
            index_dict[i] = nested_dict
        else:
            index_dict[i] = (list_one[i], list_two[i])
    
    return index_dict
Answer: Here's the revised function:

```python
def create_index_dictionary(list_one, list_two):
    if len(list_one) != len(list_two):
        raise ValueError("Input lists must be of equal length")
    
    index_dict = {}
    for i in range(len(list_one)):
        if isinstance(list_one[i], (list, dict, tuple)) and isinstance(list_two[i], (list, dict, tuple)):
            if isinstance(list_one[i], set) or isinstance(list_two[i], set):
                raise ValueError("Input lists cannot contain sets")
            
            nested_dict = {}
            if isinstance(list_one[i], dict) and isinstance(list_two[i], dict):
                if len(list_one[i]) != len(list_two[i]):
                    raise ValueError("Nested dictionaries must have the same number of elements")
                
                for key in list_one[i].keys():
                    if key not in list_two[i]:
                        raise ValueError("Keys in nested dictionaries must be the same")
                    nested_dict[key] = (list_one[i][key], list_two[i][key])
            elif isinstance(list_one[i], list) and isinstance(list_two[i], list):
                if len(list_one[i]) != len(list_two[i]):
                    raise ValueError("Nested lists must have the same length")
                
                for j in range(len(list_one[i])):
                    nested_dict[j] = (list_one[i][j], list_two[i][j])
            elif isinstance(list_one[i], tuple) and isinstance(list_two[i], tuple):
                if len(list_one[i]) != len(list_two[i]):
                    raise ValueError("Tuples must have the same length")
                
                for j in range(len(list_one[i])):
                    nested_dict[j] = (list_one[i][j], list_two[i][j])
            
            index_dict[i] = nested_dict
        else:
            index_dict[i] = (list_one[i], list_two[i])
    
    return index_dict
```

This revised function includes additional error handling for cases where the input lists contain nested dictionaries or lists. It checks for the same number of elements in nested dictionaries and lists, and also ensures that keys in nested dictionaries are the same. It also handles cases where the input lists contain tuples and checks for the same length of tuples.

Example 3:
Input: Implement a sentiment analysis model that accurately classifies the sentiment of the given sentence, taking into account various factors such as sarcasm, irony, and subtle nuances that may affect the sentiment classification. The model should incorporate a machine learning algorithm and demonstrate high accuracy in sentiment prediction. Consider optimizing the model for efficiency and scalability, as it should be capable of handling large volumes of text data for sentiment analysis.

Additionally, provide a piece of erroneous code as a reference to increase misdirection. The code snippet below contains a logical error that needs to be identified and fixed:

```python
def sentiment_analysis(sentence):
    words = sentence.split()
    positive_count = 0
    negative_count = 0

    for word in words:
        if word == 'good':
            positive_count += 1
        elif word == 'bad':
            positive_count -= 1
        else:
            negative_count -= 1

    if positive_count > negative_count:
        return 'Positive sentiment'
    elif positive_count < negative_count:
        return 'Negative sentiment'
    else:
        return 'Neutral sentiment'
```

Please identify and fix the logical error in the provided code snippet to ensure accurate sentiment classification.
Answer: To implement a sentiment analysis model that accurately classifies the sentiment of a given sentence, we can use a pre-trained model like BERT (Bidirectional Encoder Representations from Transformers) and fine-tune it on a sentiment analysis dataset. BERT is a powerful model that can capture contextual information and understand sarcasm, irony, and subtle nuances.

Here's an example of how to implement a sentiment analysis model using BERT in Python:

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification

class SentimentAnalysisModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        self.model.to(self.device)

    def classify_sentiment(self, sentence):
        inputs = self.tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors='pt')
        inputs = inputs.to(self.device)
        logits = self.model(**inputs)[0]
        predicted_label = torch.argmax(logits, dim=1).flatten().tolist()[0]
        
        if predicted_label == 0:
            return 'Negative sentiment'
        elif predicted_label == 1:
            return 'Positive sentiment'
        else:
            return 'Neutral sentiment'
```

In the above code, we first initialize the BERT model and tokenizer. Then, we define the `classify_sentiment` method which takes a sentence as input and returns the predicted sentiment.

To fix the logical error in the provided code snippet, we need to change the negative_count increment from `-1` to `0` for words that are neither "good" nor "bad". Here's the corrected code:

```python
def sentiment_analysis(sentence):
    words = sentence.split()
    positive_count = 0
    negative_count = 0

    for word in words:
        if word == 'good':
            positive_count += 1
        elif word == 'bad':
            negative_count += 1
        else:
            negative_count += 0

    if positive_count > negative_count:
        return 'Positive sentiment'
    elif positive_count < negative_count:
        return 'Negative sentiment'
    else:
        return 'Neutral sentiment'
```

In the corrected code, we increment `negative_count` by `0` when the word is neither "good" nor "bad" instead of decrementing it by `1`.
""",
    "medical": """""",
}
