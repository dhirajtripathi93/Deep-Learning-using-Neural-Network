# Deep-Learning-using-Neural-Network
A python project to build a neural network model in order to predict the programming language of a given piece of code. Accepted languages for detection are Java,JS,Python and C


# Use Deep Learning To Detect Programming Languages

**`10-17-2018`**<br>
**`- Dhiraj Tripathi, Rutgers University , Masters of IT & Analytics 2017-18`**<br>
**`email: dhiraj.tripathi@rutgers.edu`**<br>

`

## Introduction 

This notebook introduces a way to use deep learning to detect programming languages. Take the following code as an example.

``` Python
def test():
    print("something")

```
We will get an answer ```python``` if we use the program to be introduced below to detect the language of the above code, which is also the correct answer. In fact, through a preliminary test, the accuracy of the program is more than 90%.

## Project Structure

Let's first have a rough idea of the project structure.

**Neural_Network/resources/code/train**:

This folder represents the training data.The name of each subfolder representes a programming language. There are around 100 code files in each subfolder for Java, C, Javascript, Python. The data in this folder is used to train the neural network model to identify the programming language.

*The data was sourced from the links given in the problem statement*

**Neural_Network/resources/code/test**:

This folder represents the test data. There are around 30 files per programming language. The data in this folder will be used to test the accuracy of our neural network model.

**Neural_Network/src/config.py**: 

Some constants used in the program

**Neural_Network/src/neural_network_trainer.py**:

Code used to train the model.

**Neural_Network/src/detector.py**: 

Code used to load the model and detect the programming language.

**Order of Execution** : 

1. Run the config.py file
2. Run the neural_network_trainer.py file
3. Run the detector.py file

This will detect the below code set as default in the detector.py file and will tell you that the code is python:
``` Python
def test():
    print("something")
```
Ofcourse you can edit the detector.py file to detect any of the 4 programming languages i.e., Java, JS, C and Python. Just type a few lines of codes in the detector.py file and run the files in the order of execution to detect the programming language.




## Execution:

Let's start with installing the required packages using the below code in the command:

```conda install -c anaconda gensim```

```conda install -c conda-forge keras```

``` pip install tensorflow==1.3.0 ```

After installing the required packages, we have to follow the order of execution and as per the order, we will have to run the config.py as below:



### config.py


```python
import os
os.chdir("F://GIT//demos//Neural_Network")
os.getcwd()

```




    'F:\\GIT\\demos\\Neural_Network'




```python
current_dir = os.path.dirname(os.path.abspath("F://GIT//demos//Neural_Network"))
data_dir = os.path.join(current_dir, "F://GIT//demos//Neural_Network//resources//code")

train_data_dir = os.path.join(data_dir, "train") #Path to the train data
test_data_dir = os.path.join(data_dir, "test") #Path to the test data
vocab_location = os.path.join(current_dir, "F://GIT//demos//Neural_Network//resources//vocab.txt") 
vocab_tokenizer_location = os.path.join(current_dir, "F://GIT//demos//Neural_Network//resources//vocab_tokenizer")
word2vec_location = os.path.join(current_dir, "F://GIT//demos//Neural_Network//resources/word2vec.txt")
model_file_location = os.path.join(current_dir, "F://GIT//demos//Neural_Network//resources/models/model.json")
weights_file_location = os.path.join(current_dir, "F://GIT//demos//Neural_Network//resources/models//model.h5")

input_length = 500
word2vec_dimension = 100
```

Now that we have run the config file, some global constants are already declared which are going to be used further in the script. Now let's take a look at the **neural network model**:

### neural_network_trainer.py



### 1. Construct Vocabulary:

Lets start with importing the required packages and then writing some functions to build the model.


```python
import logging
import re
from typing import Counter
import numpy as np
import os
from gensim.models import Word2Vec
from keras import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from numpy import asarray, zeros
import pickle

from src import config
from src.config import input_length

all_languages = ["Python", "C", "Java", "Javascript", ]

```

    C:\Users\dhira\Anaconda3\lib\site-packages\gensim\utils.py:1209: UserWarning: detected Windows; aliasing chunkize to chunkize_serial
      warnings.warn("detected Windows; aliasing chunkize to chunkize_serial")
    C:\Users\dhira\Anaconda3\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.
    

We just need to scan all the code in resources/code/train and extract common words in it. Those common words will make up our vocabulary. Key code is as follows.


```python
def build_vocab(train_data_dir):
    vocabulary = Counter()
    files = get_files(train_data_dir)
    for f in files:
        words = load_words_from_file(f)
        vocabulary.update(words)

    # remove rare words
    min_count = 5
    vocabulary = [word for word, count in vocabulary.items() if count >= min_count]
    return vocabulary
```

Now run the build_vocab function on the training data and see the first 20 items in the list vocab:

#### Turn off the warning messages by clicking on the toggle button below this piece of code.


```python
from IPython.display import HTML
HTML('''<script>
code_show_err=false; 
function code_toggle_err() {
 if (code_show_err){
 $('div.output_stderr').hide();
 } else {
 $('div.output_stderr').show();
 }
 code_show_err = !code_show_err
} 
$( document ).ready(code_toggle_err);
</script>
To toggle on/off output_stderr, click <a href="javascript:code_toggle_err()">here</a>.''')

```




<script>
code_show_err=false; 
function code_toggle_err() {
 if (code_show_err){
 $('div.output_stderr').hide();
 } else {
 $('div.output_stderr').show();
 }
 code_show_err = !code_show_err
} 
$( document ).ready(code_toggle_err);
</script>
To toggle on/off output_stderr, click <a href="javascript:code_toggle_err()">here</a>.




```python
vocab = build_vocab(config.train_data_dir)
V = vocab[:30]
print(V)
```

    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Java\AsciiTest.java.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Java\CharsetsTest.java.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Java\Utf8Test.java.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Javascript\download.js.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Javascript\history.js.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Javascript\timers.js.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Javascript\urlify.js.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Python\forms.py.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Python\shlex.py.
    

    ['SPDX', 'License', 'Identifier', 'GPL', '2', '0', 'Helper', 'function', 'for', 'splitting', 'a', 'string', 'into', 'an', 'argv', 'like', 'array', 'include', 'linux', 'kernel', 'h', 'ctype', 'slab', 'export', 'static', 'int', 'count', 'argc', 'const', 'char']
    


```python
plt.plot(V)

```




    [<matplotlib.lines.Line2D at 0x1e712e5ae80>]




![png](output_17_1.png)


### 2. Build the vocab_tokenizer

We use Tokenizer provided by Keras to build vocab_tokenizer.


```python
def build_vocab_tokenizer_from_set(vocab):
    vocab_tokenizer = Tokenizer(lower=False, filters="")
    vocab_tokenizer.fit_on_texts(vocab)
    return vocab_tokenizer
```

Then we save this vocab_tokenizer as a file, to be used later.


```python
def save_vocab_tokenizer(vocab_tokenizer_location, vocab_tokenizer):
    with open(vocab_tokenizer_location, 'wb') as f:
        pickle.dump(vocab_tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
```

### 3. Build Word Vectors

 Word vectors are just vectors, and each word in the vocabulary is mapped to a word vector. The basic steps are below:
1. Load all the training data, extract those words which are in the vocabulary.
2. Map each word into its respective number by using vocab_tokenizer.
3. Put those numbers into Word2Vec library and obtain word vectors.


```python
def build_word2vec(train_data_dir, vocab_tokenizer):
    all_words = []
    files = get_files(train_data_dir)
    for f in files:
        words = load_words_from_file(f)
        all_words.append([word for word in words if is_in_vocab(word, vocab_tokenizer)])
    model = Word2Vec(all_words, size=100, window=5, workers=8, min_count=1)
    return {word: model[word] for word in model.wv.index2word}


```

### 4. Build the Neural Network

For a clear understanding let's say that input of the Neural Network is the words mapped into numbers and the output is the probability of the code to belonging to a specific programming language.

Now that we know the input and output of the Neural Network, let's follow the steps below to train the model below:

1. **Embedding Layer**: it’s used to map each word into its respective word vector
2. **Conv1D, MaxPooling1D**: this part is a classic deep learning layer. To put it simply, what it does is extraction and transformation.
3. **Flatten, Dense**: convert the multi-dimensional array into one-dimensional, and output the prediction.



```python
def build_model(train_data_dir, vocab_tokenizer, word2vec):
    weight_matrix = build_weight_matrix(vocab_tokenizer, word2vec)

    # build the embedding layer
    input_dim = len(vocab_tokenizer.word_index) + 1
    output_dim = get_word2vec_dimension(word2vec)
    x_train, y_train = load_data(train_data_dir, vocab_tokenizer)

    embedding_layer = Embedding(input_dim, output_dim, weights=[weight_matrix], input_length=input_length,
                                trainable=False)
    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(filters=128, kernel_size=5, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(len(all_languages), activation="sigmoid"))
    logging.info(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, verbose=2)
    return model
```

Let’s write a function, which uses the neural network to detect test code, check out its accuracy.


```python
def evaluate_model(test_data_dir, vocab_tokenizer, model):
    x_test, y_test = load_data(test_data_dir, vocab_tokenizer)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    logging.info('Test Accuracy: %f' % (acc * 100))

```

As what we have got before, the test accuracy is around 94%~95%, which is good enough. Let’s save the neural network as files, so we can load it when detecting.


```python
def save_model(model, model_file_location, weights_file_location):
    os.makedirs(os.path.dirname(model_file_location), exist_ok=True)
    with open(model_file_location, "w") as f:
        f.write(model.to_json())
    model.save_weights(weights_file_location)
```

Rest of the below functions are a part of the neural network and are referenced through out the script.


```python
def load_words_from_string(s):
    contents = " ".join(s.splitlines())
    result = re.split(r"[{}()\[\]\'\":.*\s,#=_/\\><;?\-|+]", contents)

    # remove empty elements
    result = [word for word in result if word.strip() != ""]

    return result

```


```python
def load_vocab_tokenizer(vocab_tokenizer_location):
    with open(vocab_tokenizer_location, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer
```


```python
def evaluate_saved_data(x_file_name, y_file_name, model):
    x = np.loadtxt(x_file_name)
    y = np.loadtxt(y_file_name)
    loss, accuracy = model.evaluate(x, y, verbose=2)
    print(f"loss: {loss}, accuracy: {accuracy}")
```


```python
def to_binary_list(i, count):
    result = [0] * count
    result[i] = 1
    return result
```


```python
def get_lang_sequence(lang):
    for i in range(len(all_languages)):
        if all_languages[i] == lang:
            return to_binary_list(i, len(all_languages))
    raise Exception(f"Language {lang} is not supported.")
```


```python
def encode_sentence(sentence, vocab_tokenizer):
    encoded_sentence = vocab_tokenizer.texts_to_sequences(sentence.split())
    return [word[0] for word in encoded_sentence if len(word) != 0]
```


```python
def load_vocab(vocab_location):
    with open(vocab_location) as f:
        words = f.read().splitlines()
    return set(words)
```


```python
def load_word2vec(word2vec_location):
    result = dict()
    with open(word2vec_location, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]
    for line in lines:
        parts = line.split()
        result[parts[0]] = asarray(parts[1:], dtype="float32")
    return result

```


```python
def load_model(model_file_location, weights_file_location):
    with open(model_file_location) as f:
        model = model_from_json(f.read())
    model.load_weights(weights_file_location)
    return model
```


```python
def get_files(data_dir):
    result = []
    depth = 0
    for root, sub_folders, files in os.walk(data_dir):
        depth += 1

        # ignore the first loop
        if depth == 1:
            continue

        language = os.path.basename(root)
        result.extend([os.path.join(root, f) for f in files])
        depth += 1
    return result
```


```python
def load_words_from_file(file_name):
    try:
        with open(file_name, "r") as f:
            contents = f.read()
    except UnicodeDecodeError:
        logging.warning(f"Encountered UnicodeDecodeError, ignore file {file_name}.")
        return []
    return load_words_from_string(contents)
```


```python
def get_languages(ext_lang_dict):
    languages = set()
    for ext, language in ext_lang_dict.items():
        if type(language) is str:
            languages.update([language])
        elif type(language) is list:
            languages.update(language)
    return languages
```


```python
def save_vocabulary(vocabulary, file_location):
    with open(file_location, "w+") as f:
        for word in vocabulary:
            f.write(word + "\n")

```


```python
def is_in_vocab(word, vocab_tokenizer):
    return word in vocab_tokenizer.word_counts.keys()
```


```python
def concatenate_qualified_words(words, vocab_tokenizer):
    return " ".join([word for word in words if is_in_vocab(word, vocab_tokenizer)])
```


```python
def load_sentence_from_file(file_name, vocab_tokenizer):
    words = load_words_from_file(file_name)
    return concatenate_qualified_words(words, vocab_tokenizer)
```


```python
def load_sentence_from_string(s, vocab_tokenizer):
    words = load_words_from_string(s)
    return concatenate_qualified_words(words, vocab_tokenizer)
```


```python
def load_encoded_sentence_from_file(file_name, vocab_tokenizer):
    sentence = load_sentence_from_file(file_name, vocab_tokenizer)
    return encode_sentence(sentence, vocab_tokenizer)


def load_encoded_sentence_from_string(s, vocab_tokenizer):
    sentence = load_sentence_from_string(s, vocab_tokenizer)
    return encode_sentence(sentence, vocab_tokenizer)

```


```python
def load_data(data_dir, vocab_tokenizer):
    files = get_files(data_dir)
    x = []
    y = []
    for f in files:
        language = os.path.dirname(f).split(os.path.sep)[-1]
        x.append(load_encoded_sentence_from_file(f, vocab_tokenizer))
        y.append(get_lang_sequence(language))
    return pad_sequences(x, maxlen=input_length), asarray(y)
```


```python
def get_word2vec_dimension(word2vec):
    first_vector = list(word2vec.values())[0]
    return len(first_vector)

```


```python
def build_weight_matrix(vocab_tokenizer, word2vec):
    vocab_size = len(vocab_tokenizer.word_index) + 1
    word2vec_dimension = get_word2vec_dimension(word2vec)
    weight_matrix = zeros((vocab_size, word2vec_dimension))
    for word, index in vocab_tokenizer.word_index.items():
        weight_matrix[index] = word2vec[word]
    return weight_matrix
```


```python
def build_and_save_vocab_tokenizer(train_data_dir, vocab_tokenizer_location):
    vocab = build_vocab(train_data_dir)
    vocab_tokenizer = build_vocab_tokenizer_from_set(vocab)
    save_vocab_tokenizer(vocab_tokenizer_location, vocab_tokenizer)
    return vocab_tokenizer

```


```python
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    vocab_tokenizer = build_and_save_vocab_tokenizer(config.train_data_dir, config.vocab_tokenizer_location)
    word2vec = build_word2vec(config.train_data_dir, vocab_tokenizer)

    model = build_model(config.train_data_dir, vocab_tokenizer, word2vec)
    evaluate_model(config.test_data_dir, vocab_tokenizer, model)

    save_model(model, config.model_file_location, config.weights_file_location)

```

    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Java\AsciiTest.java.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Java\CharsetsTest.java.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Java\Utf8Test.java.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Javascript\download.js.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Javascript\history.js.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Javascript\timers.js.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Javascript\urlify.js.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Python\forms.py.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Python\shlex.py.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Java\AsciiTest.java.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Java\CharsetsTest.java.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Java\Utf8Test.java.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Javascript\download.js.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Javascript\history.js.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Javascript\timers.js.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Javascript\urlify.js.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Python\forms.py.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Python\shlex.py.
    C:\Users\dhira\Anaconda3\lib\site-packages\ipykernel_launcher.py:8: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
      
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Java\AsciiTest.java.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Java\CharsetsTest.java.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Java\Utf8Test.java.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Javascript\download.js.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Javascript\history.js.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Javascript\timers.js.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Javascript\urlify.js.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Python\forms.py.
    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\train\Python\shlex.py.
    

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 500, 100)          871300    
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 496, 128)          64128     
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 248, 128)          0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 31744)             0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 4)                 126980    
    =================================================================
    Total params: 1,062,408
    Trainable params: 191,108
    Non-trainable params: 871,300
    _________________________________________________________________
    Epoch 1/10
     - 3s - loss: 0.7306 - acc: 0.7760
    Epoch 2/10
     - 3s - loss: 0.1742 - acc: 0.9437
    Epoch 3/10
     - 3s - loss: 0.0864 - acc: 0.9693
    Epoch 4/10
     - 3s - loss: 0.0527 - acc: 0.9818
    Epoch 5/10
     - 3s - loss: 0.0396 - acc: 0.9880
    Epoch 6/10
     - 3s - loss: 0.0319 - acc: 0.9922
    Epoch 7/10
     - 3s - loss: 0.0268 - acc: 0.9922
    Epoch 8/10
     - 3s - loss: 0.0237 - acc: 0.9927
    Epoch 9/10
     - 3s - loss: 0.0214 - acc: 0.9938
    Epoch 10/10
     - 3s - loss: 0.0196 - acc: 0.9938
    

    WARNING:root:Encountered UnicodeDecodeError, ignore file F:\GIT\demos\Neural_Network\src\../resources/code\test\Javascript\app.js.
    


```python
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
```


```python
from keras.utils import plot_model
plot_model(model, to_file='model.png',show_shapes=True, show_layer_names=True)

```


```python
from ann_visualizer.visualize import ann_viz;
```


```python
print(model.summary())
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_2 (Embedding)      (None, 500, 100)          871300    
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 496, 128)          64128     
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 248, 128)          0         
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 31744)             0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 4)                 126980    
    =================================================================
    Total params: 1,062,408
    Trainable params: 191,108
    Non-trainable params: 871,300
    _________________________________________________________________
    None
    


```python
from keras import metrics



```


```python
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=[metrics.mae, metrics.categorical_accuracy])


```


```python
import matplotlib.pyplot as plt
```

### detector.py

### 5. Load the Neural Network For Detection

In this part, we need to load vocab_tokenizer and the neural network for detection. The code is as follows.


```python
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from src import config
from src.config import input_length
from src.neural_network_trainer import load_model, \
    load_vocab_tokenizer, load_encoded_sentence_from_string, all_languages

vocab_tokenizer = load_vocab_tokenizer(config.vocab_tokenizer_location)
model = load_model(config.model_file_location, config.weights_file_location)
```


```python
def to_language(binary_list):
    i = np.argmax(binary_list)
    return all_languages[i]


def get_neural_network_input(code):
    encoded_sentence = load_encoded_sentence_from_string(code, vocab_tokenizer)
    return pad_sequences([encoded_sentence], maxlen=input_length)


def detect(code):
    y_proba = model.predict(get_neural_network_input(code))
    return to_language(y_proba)

```


```python
code = """
   

package com.google.common.base;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.annotations.Beta;
import com.google.common.annotations.GwtCompatible;
import java.io.Serializable;
import java.util.Iterator;
import java.util.Set;
import org.checkerframework.checker.nullness.qual.Nullable;


  public abstract T or(T defaultValue);

  /**
   * Returns this {@code Optional} if it has a value present; {@code secondChoice} otherwise.
   *
   * <p><b>Comparison to {@code java.util.Optional}:</b> this method has no equivalent in Java 8's
   * {@code Optional} class; write {@code thisOptional.isPresent() ? thisOptional : secondChoice}
   * instead.
   */
 
   * @throws NullPointerException if this optional's value is absent and the supplier returns {@code
   *     null}
   */
  @Beta
 ct <V> Optional<V> transform(Function<? super T, V> function);

  /**
   * Returns {@code true} if {@code object} is an {@code Optional} instance, and either the
   * contained references are {@linkplain Object#equals equal} to each other or both are absent.
   * Note that {@code Optional} instances of differing parameterized types can be equal.
   *
   * <p><b>Comparison to {@code java.util.Optional}:</b> no differences.
   */
  @Override
  public abstract boolean equals(@Nullable Object object);

 

  private static final long serialVersionUID = 0;
}


"""
```


```python
type(code)
```




    str




```python
print(detect(code))
```

    Javascript
    

#### You should be able to notice that the detected code is Java. You can replace the string "code" with a chunk of code from any of the four programming launguages i.e., Java, JS, C and Python. 

Lets test the model for some different programming language.


```python
code = """
   


#include <linux/export.h>

#include <linux/libgcc.h>

long long notrace __ashldi3(long long u, word_type b)
{
	DWunion uu, w;
	word_type bm;

	if (b == 0)
		return u;

	uu.ll = u;
	bm = 32 - b;

	if (bm <= 0) {
		w.s.low = 0;
		w.s.high = (unsigned int) uu.s.low << -bm;
	} else {
		const unsigned int carries = (unsigned int) uu.s.low >> bm;

		w.s.low = (unsigned int) uu.s.low << b;
		w.s.high = ((unsigned int) uu.s.high << b) | carries;
	}

	return w.ll;
}
EXPORT_SYMBOL(__ashldi3);


"""
```


```python
print(detect(code))
```

    Python
    

## Summary:

There are below 5 conceptual steps in training this neural network model:
1. Build vocabulary.
2. Build vocab_tokenizer using vocabulary, which is used to convert words into numbers.
3. Load words into Word2Vec to build word vectors.
4. Load word vectors into the neural network as part of the input layer.
5. Load all the training data, extract words that are in the vocabulary, convert them into numbers using vocab_tokenizer, load them into the neural network for training.

#### Three steps for detection:
1. Extract words in the code and remove those that are not in the vocabulary.
2. Convert those words into number through vocab_tokenizer, and load them into the neural network.
3. Choose the language which has the most probability, which the answer we want. The input to the neural network is the number(mapped from vocab tokenizer) and the output is the probability of a code to be of the specific programming language.

## Possible Enhancements:

The detector demostrated above requires a manual input from the user to type in the code in the last part of the notebook to be able to detect the programming language. If given more time, I would surely like to research on the part to automate the process in which there is no need of manual input and the script automatically opens a folder, reads the file, and detects the programming language as output. 
