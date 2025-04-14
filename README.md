# Chinese Word2Vec Training and Testing

This project provides tools for training and testing Chinese Word2Vec models from JSON file using Gensim and Jieba. It includes scripts for training the model and testing semantic relationships in the trained model.

## Project Structure

```
.
├── README.md                # Project documentation
├── train_word2vec.py        # Script for training Word2Vec model
├── train2_word2vec.py       # Script for a better trained Word2Vec model
├── test_word2vec.py         # Script for testing semantic relationships
├── corpus_cut.txt           # Segmented Chinese corpus
├── corpus.txt               # Chinese corpus
├── segment.py               # Script for text segmentation
├── json_to_txt.py           # Script for converting JSON to text
└── jieba_dict/              # Directory containing Jieba dictionary files
```

## Requirements

- Python 3.x
- Gensim
- Jieba
- NumPy

Install the required packages:
```bash
pip install gensim jieba numpy
```

## Usage
### 1. Pre-treatment of corpus

The `json_to_txt.py` convert the json file into txt corpus. 

```bash
python json_to_txt.py
```

The `segment.py` segments original corpus into segmented corpus for further training using Jieba.

```bash
python segment.py
```

### 2. Training the Word2Vec Model

The `train_word2vec.py` or `train2_word2vec.py` script trains a Chinese Word2Vec model with several parameters:

```bash
python train_word2vec.py (train2_word2vec.py)
```

The script `train2_word2vec.py` uses the following optimized parameters:
- Vector size: 300 dimensions
- Window size: 8 (for capturing longer-range dependencies in Chinese)
- Minimum word count: 3
- Skip-gram algorithm (sg=1)
- Negative sampling with 10 negative samples
- 15 training epochs
- Automatic CPU core utilization

### 3. Testing Semantic Relationships

The `test_word2vec.py` script tests various semantic relationships in the trained model:

```bash
python test_word2vec.py
```

The script tests:
- Word analogies (e.g., 女王-女人+男人=国王)
- Word similarities
- Most similar words
- Various Chinese semantic relationships

## Model Parameters

The training script uses the following optimized parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| vector_size | 300 | Dimension of word vectors |
| window | 8 | Context window size |
| min_count | 3 | Minimum word frequency |
| sg | 1 | Skip-gram algorithm |
| negative | 10 | Number of negative samples |
| epochs | 15 | Training iterations |
| sample | 1e-4 | Downsampling threshold |
| alpha | 0.025 | Initial learning rate |
| min_alpha | 0.0001 | Final learning rate |

## Example Tests

The testing script includes several example tests:
1. Word analogies:
   - 公主 + 男人 - 女人 = 王子
   - 法国 + 北京 - 中国 = 巴黎
   - 人民币 + 美国 - 中国 = 美元
   - 笔 + 画家 - 作家 = 油画笔
   - 窗户 + 汽车 - 房子 = 车窗

2. Word similarities:
   - City similarities (北京-上海)

3. Most similar words:
   - Most similar to "中国"
   - Most similar to "上海"

## Notes

- The model is trained on a segmented Chinese corpus
- The training process uses all available CPU cores for faster processing
- The model includes loss computation for monitoring training progress
- Sample word frequencies are displayed for verification

## License

This project is open source and available under the MIT License. 
