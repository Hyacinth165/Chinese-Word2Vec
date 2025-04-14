from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging
import time
import multiprocessing

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def train_word2vec(corpus_path, model_path, vector_size=300, window=8, min_count=3, workers=None, 
                  sg=1, hs=0, negative=10, ns_exponent=0.75, epochs=10, sample=1e-4, 
                  alpha=0.025, min_alpha=0.0001):
    """
    Train a Word2Vec model on a segmented Chinese corpus with optimized parameters
    
    Args:
        corpus_path: Path to the segmented corpus file
        model_path: Path to save the trained model
        vector_size: Dimension of word vectors 
        window: Maximum distance between the current and predicted word 
        min_count: Minimum word frequency threshold 
        workers: Number of worker threads to train the model 
        sg: Training algorithm: 1 for skip-gram, 0 for CBOW 
        hs: If 1, hierarchical softmax will be used 
        negative: Number of negative samples 
        ns_exponent: Negative sampling exponent 
        epochs: Number of iterations over the corpus 
        sample: Threshold for configuring which higher-frequency words are randomly downsampled 
        alpha: Initial learning rate 
        min_alpha: Final learning rate
    """
    print("Starting Word2Vec training...")
    start_time = time.time()
    
    if workers is None:
        workers = multiprocessing.cpu_count()
    
    sentences = LineSentence(corpus_path)
    
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        hs=hs,
        negative=negative,
        ns_exponent=ns_exponent,
        epochs=epochs,
        sample=sample,
        alpha=alpha,
        min_alpha=min_alpha,
        compute_loss=True 
    )
    
    model.save(model_path)
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Model saved to {model_path}")
    print(f"Vocabulary size: {len(model.wv)}")
    print(f"Training loss: {model.get_latest_training_loss():.2f}")
    

if __name__ == "__main__":
    
    corpus_path = "corpus_cut.txt"
    model_path = "chinese_word2vec.model"
    
    train_word2vec(
        corpus_path=corpus_path,
        model_path=model_path,
        vector_size=300,      
        window=8,            
        min_count=3,         
        workers=None,         
        sg=1,               
        hs=0,                
        negative=10,         
        ns_exponent=0.75,   
        epochs=10,           
        sample=1e-4,         
        alpha=0.025,         
        min_alpha=0.0001     
    ) 