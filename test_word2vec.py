from gensim.models import Word2Vec
import numpy as np

def load_model(model_path):
    print("Loading model...")
    model = Word2Vec.load(model_path)
    print(f"Model loaded. Vocabulary size: {len(model.wv)}")
    return model

def test_analogy(model, positive, negative, topn=5):
    """
    Test word analogies in the model
    
    Args:
        model: Loaded Word2Vec model
        positive: List of positive terms 
        negative: List of negative terms 
        topn: Number of similar words to return
    """
    try:
        results = model.wv.most_similar(positive=positive, negative=negative, topn=topn)
        print(f"\nTesting analogy: {' + '.join(positive)} - {' - '.join(negative)}")
        print("Most similar words:")
        for word, similarity in results:
            print(f"{word}: {similarity:.4f}")
    except KeyError as e:
        print(f"Error: {e} - One or more words not in vocabulary")

# Similarity between two words
def test_similarity(model, word1, word2):
    try:
        similarity = model.wv.similarity(word1, word2)
        print(f"\nSimilarity between '{word1}' and '{word2}': {similarity:.4f}")
    except KeyError as e:
        print(f"Error: {e} - One or more words not in vocabulary")

# Most similar words to a given word
def test_most_similar(model, word, topn=5):
    try:
        results = model.wv.most_similar(word, topn=topn)
        print(f"\nMost similar words to '{word}':")
        for similar_word, similarity in results:
            print(f"{similar_word}: {similarity:.4f}")
    except KeyError as e:
        print(f"Error: {e} - Word not in vocabulary")

if __name__ == "__main__":
    model = load_model("chinese_word2vec.model")
    
    # Test Chinese analogies
    test_analogy(model, positive=['公主', '男人'], negative=['女人'])
    test_analogy(model, positive=['法国', '北京'], negative=['中国']) 
    test_analogy(model, positive=['人民币', '美国'], negative=['中国']) 
    test_analogy(model, positive=['笔', '画家'], negative=['作家']) 
    test_analogy(model, positive=['窗户', '汽车'], negative=['房子']) 

    # Test word similarities
    test_similarity(model, '北京', '上海')  
    
    # Test most similar words
    test_most_similar(model, '中国') 
    test_most_similar(model, '上海')  