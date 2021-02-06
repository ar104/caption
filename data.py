import tensorflow as tf
import numpy as np

def gen_ngrams(sentence, n):
    ngrams = []
    for i in range(len(sentence) - n + 1):
        ngrams.append(sentence[i:(i + n)])
    ngram_dict = {}
    for ng in ngrams:
        ng_key = " ".join([str(i) for i in ng]).lower()
        if ng_key not in ngram_dict:
            ngram_dict[ng_key] = 1
        else:
            ngram_dict[ng_key] += 1
    return ngram_dict

def bleu(candidate, references):
    bleu_numerator = 0
    bleu_denominator = 0
    for ngram, count in candidate.items():
        max_count = 0
        for r in references:
            if ngram in r:
                max_count = max(max_count, r[ngram])
        bleu_numerator += min(max_count, count)
        bleu_denominator += count
    return float(bleu_numerator)/bleu_denominator if bleu_denominator != 0 else 0.0

def build_annotations_vocab(fname):
    """ 
    Build an annotations vocabulary for the flickr8k dataset.
    Take a string filename as input.
    Returns a dictionary mapping words to token numbers
    and a dictionary mapping images to a list of tokens.
    """
    image_to_tokens = {}
    vocab = {'<START>' : 0}
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            key, desc = line.split('\t')
            try:
                jpeg, ann_num = key.split('#')
            except Exception as e:
                print('failed token key parsing on {}'.format(key))
                raise e
            if ann_num != '0':
                continue
            words = desc.strip('\n').split(' ')
            tokens = []
            for word in words:
                if len(word) < 2:
                    continue
                if word not in vocab:
                    vocab[word] = len(vocab)
                tokens.append(vocab[word])
            image_to_tokens[jpeg] = [vocab['<START>']] + tokens
    vocab['<STOP>'] = len(vocab)
    return vocab, image_to_tokens

def load_annotations_vocab(fname):
    """
    Load annotations vocabulary from a file.
    Takes filename as string argument.
    Returns a list of words corresponding to integer
    token indices.
    """
    token_to_word = {}
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            w, index = line.strip('\n').split(',')
            token_to_word[index] = w
    token_to_word_list = [0] * len(token_to_word)
    for k,v in token_to_word.items():
        token_to_word_list[int(k)] = v
    return token_to_word_list

def load_annotations_tokens(fname, stop_token):
    """
    Load annotations token from a file.
    Takes filename as string argument and a stop token number
    Returns a list of dictionary of images to tokens with
    an added stop token at the end.
    """
    image_to_tokens = {}
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            items = line.strip('\n').split(',')
            image_to_tokens[items[0]] = [int(k) for k in items[1:]]
            image_to_tokens[items[0]].append(stop_token)
    return image_to_tokens

def pad(s, max_caption_length, stop_symbol):
    """Pad out sequence of tokens s to max_caption_length using stop symbol."""
    l = s.copy()
    l.extend([stop_symbol] * (max_caption_length - len(l)))
    return np.asarray(l)

def load_image(fname, include_batch=True):
    """Load a single image and return the array with an added batch dimension."""
    image = tf.keras.preprocessing.image.load_img(fname, target_size = (224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    if include_batch:
        img_array = img_array.reshape(1, img_array.shape[0], img_array.shape[1], img_array.shape[2])
    return img_array

# Quick test
def quick_test():
    vocab, image_to_tokens = build_annotations_vocab('/datadrive/flickr8k/Flickr8k.token.txt')
    with open('/datadrive/flickr8k/Flickr8k.vocab.txt', 'w') as f:
        for w, index in vocab.items():
            f.write('{},{}\n'.format(w, index))
    with open('/datadrive/flickr8k/Flickr8k.image_to_tokens.txt', 'w') as f:
        for w, tokens in image_to_tokens.items():
            f.write('{}'.format(w))
            for t in tokens:
                f.write(',{}'.format(t))
            f.write('\n')
    token_to_word = load_annotations_vocab('/datadrive/flickr8k/Flickr8k.vocab.txt')
    image_to_tokens = load_annotations_tokens('/datadrive/flickr8k/Flickr8k.image_to_tokens.txt', len(token_to_word) - 1)
    for t in image_to_tokens['1001773457_577c3a7d70.jpg']:
        print(token_to_word[t])

if __name__ == '__main__':
    quick_test()
