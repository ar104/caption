import tensorflow as tf
import numpy as np
import data
import queue
import math

def make_vgg16_model():
    vgg_model = tf.keras.applications.vgg16.VGG16()
    base_model = tf.keras.Model(inputs=vgg_model.input, outputs=tf.keras.layers.Reshape(target_shape=(196, 512))(vgg_model.get_layer('block5_conv3').output))
    base_model.trainable = False
    return base_model

def make_model(lstm_units, embedding_size, stop_symbol, max_caption_length, vocab_size, dropout_rate=0.0, stochastic_loss_lambda=0.00005):
    conv_features = tf.keras.layers.Input(shape=(196, 512))
    h_init = tf.keras.layers.Dense(units=lstm_units, activation = 'relu')(tf.math.reduce_mean(conv_features, axis=-2))
    c_init = tf.keras.layers.Dense(units=lstm_units, activation = 'relu')(tf.math.reduce_mean(conv_features, axis=-2))
    lstm = tf.keras.layers.LSTM(units=lstm_units, return_state=True)
    attention = tf.keras.layers.Attention()
    token_inputs = tf.keras.Input(shape=(max_caption_length,))
    input_embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim = embedding_size)
    final_output_net = tf.keras.layers.Dense(units = vocab_size, activation = 'softmax')
    hidden_state_output_net = tf.keras.layers.Dense(units = embedding_size, activation = None)
    attention_output_net = tf.keras.layers.Dense(units = embedding_size, activation = None)
    attention_projection = tf.keras.layers.Dense(units=512)
    beta_computation = tf.keras.layers.Dense(units=1)
    output_symbols = []
    attention_scores_list = []
    for i in range(max_caption_length - 1):
        attention_query = attention_projection(h_init)
        attention_query = tf.keras.layers.Dropout(rate=dropout_rate)(attention_query)
        attention_input, attention_scores = attention([tf.keras.layers.Reshape(target_shape=(1, 512))(attention_query), conv_features], return_attention_scores=True)
        attention_scores = tf.keras.layers.Reshape(target_shape=(196,))(attention_scores)
        all_zeros = tf.zeros(shape=(1, 196), dtype=tf.float32)
        attention_scores = tf.where(tf.math.equal(token_inputs[:, i:i+1], tf.constant(stop_symbol, dtype=tf.float32)), all_zeros, attention_scores)
        beta = beta_computation(attention_scores)
        attention_scores = tf.math.multiply(beta, attention_scores)
        attention_scores_list.append(attention_scores)
        token_input = input_embedding(token_inputs[:, i:i+1])
        token_input  = tf.keras.layers.Reshape(target_shape=(1, embedding_size))(token_input)
        lstm_input = tf.keras.layers.concatenate([attention_input, token_input])
        _, h_init, c_init = lstm(inputs=lstm_input, initial_state=[h_init, c_init])
        output_driver = tf.keras.layers.Add()([token_input, hidden_state_output_net(h_init), attention_output_net(attention_input)])
        output_symbols.append(tf.keras.layers.Reshape(target_shape=(1, vocab_size))(final_output_net(output_driver)))
    
    attention_sum = tf.keras.layers.Add()(attention_scores_list)
    all_ones = tf.constant(1., shape=(196,), dtype=tf.float32)
    attention_diff = tf.math.subtract(all_ones, attention_sum)
    stochastic_loss_term = tf.math.reduce_sum(tf.math.square(attention_diff), axis=-1)
    stochastic_loss_multiplier = tf.constant(stochastic_loss_lambda, shape=(1,), dtype=tf.float32)
    stochastic_loss = tf.math.multiply(stochastic_loss_term, stochastic_loss_multiplier)
    output_array = tf.keras.layers.concatenate(output_symbols, axis = -2)
    stop_array_numpy = np.zeros((1, vocab_size))
    stop_array_numpy[0, stop_symbol] = 1.0
    stop_array = tf.convert_to_tensor(stop_array_numpy, dtype=tf.float32)
    mask_array = tf.reshape(tf.math.equal(token_inputs[:,:-1], tf.constant(stop_symbol, dtype=tf.float32)), [-1, max_caption_length - 1, 1])
    masked_output_array = tf.where(mask_array, stop_array, output_array)
    final_model = tf.keras.Model(inputs=[conv_features, token_inputs], outputs=masked_output_array)
    final_model.add_loss(tf.reduce_sum(stochastic_loss))
    final_model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy')
    return final_model

def beam_search(model, image, token_to_word, max_caption_length, start_symbol, stop_symbol, vocab_size, beam_width):
    epsilon = 1E-19
    def update_candidate(q, candidate, added_symbol, candidate_prob):
        if q.qsize() < beam_width:
            ccopy = candidate.copy()
            if added_symbol is not None:
                ccopy.append(added_symbol)
            q.put((candidate_prob, ccopy))
        else:
            alternate_prob, alternate = q.get()
            if alternate_prob < candidate_prob:
                ccopy = candidate.copy()
                if added_symbol is not None:
                 ccopy.append(added_symbol)
                q.put((candidate_prob, ccopy))
            else:
                q.put((alternate_prob, alternate))

    candidates = queue.PriorityQueue()
    candidates.put((math.log(1.0), [start_symbol]))
    length_increment = 1
    while length_increment > 0:
        new_candidates = queue.PriorityQueue()
        max_candidate_length = max([len(l[1]) for l in candidates.queue])
        while candidates.qsize() > 0:
            candidate_prob, candidate = candidates.get()
            candidate_len = len(candidate)
            if candidate_len == max_caption_length or candidate[-1] == stop_symbol:
                update_candidate(new_candidates, candidate, None, candidate_prob)
            else:
                padded_candidate = [np.asarray([[c]]) for c in candidate]
                padded_candidate.extend([np.asarray([[start_symbol]])] *(max_caption_length - len(candidate)))
                #print(padded_candidate)
                predict_out = model.predict([image, np.concatenate(padded_candidate, axis=-1)], batch_size=1)
                #print('OK')
                for i in range(vocab_size):
                    prob = predict_out[0][candidate_len - 1][i]
                    new_candidate_prob = math.log(prob + epsilon) + candidate_prob
                    update_candidate(new_candidates, candidate, i, new_candidate_prob)
        #print('PROCESSED')
        candidates = new_candidates
        length_increment = max([len(l[1]) for l in candidates.queue]) - max_candidate_length
        #print(candidates.queue)
    results = [e for e in candidates.queue]
    results.sort(reverse=True, key=lambda x:x[0])
    return results
    
# Quick Test
def quick_test():
    def display_results(input_tokens, output_tokens, index):
        print([token_to_word[int(input_tokens[index][i])] for i in range(input_tokens.shape[-1])])
        print([token_to_word[np.argmax(output_tokens[index][i])] for i in range(output_tokens.shape[1])])
    vocab, image_to_tokens = data.build_annotations_vocab('/datadrive/flickr8k/Flickr8k.token.txt')
    with open('/datadrive/flickr8k/Flickr8k.vocab.txt', 'w') as f:
        for w, index in vocab.items():
            f.write('{},{}\n'.format(w, index))
    with open('/datadrive/flickr8k/Flickr8k.image_to_tokens.txt', 'w') as f:
        for w, tokens in image_to_tokens.items():
            f.write('{}'.format(w))
            for t in tokens:
                f.write(',{}'.format(t))
            f.write('\n')
    token_to_word = data.load_annotations_vocab('/datadrive/flickr8k/Flickr8k.vocab.txt')
    vocab_size=len(token_to_word)
    stop_symbol=vocab_size - 1
    image_to_tokens = data.load_annotations_tokens('/datadrive/flickr8k/Flickr8k.image_to_tokens.txt', stop_symbol)
    max_caption_length=max([len(t) for t in image_to_tokens.values()])
    base_model = make_vgg16_model()
    img_array1 = data.load_image('/datadrive/flickr8k/Flicker8k_Dataset/1001773457_577c3a7d70.jpg')
    conv_features1 = base_model.predict(img_array1, batch_size=1)
    img_array2 = data.load_image('/datadrive/flickr8k/Flicker8k_Dataset/760180310_3c6bd4fd1f.jpg')
    conv_features2 = base_model.predict(img_array2, batch_size=1)
    conv_features = np.concatenate([conv_features1, conv_features2], axis=0)
    model = make_model(512, 8, stop_symbol, max_caption_length, vocab_size)
    token_array1 = np.asarray(data.pad(image_to_tokens['1001773457_577c3a7d70.jpg'], max_caption_length, stop_symbol))
    token_array2 = np.asarray(data.pad(image_to_tokens['760180310_3c6bd4fd1f.jpg'], max_caption_length, stop_symbol))
    token_array = np.stack([token_array1, token_array2], axis=0)
    test_input = [conv_features, token_array]
    test_output = token_array[:, 1:]
    model.fit(test_input, test_output, batch_size=2, epochs=500)
    model.save('test_model.h5')
    predict_output = model.predict(test_input, batch_size=2)
    display_results(token_array, predict_output, 0)
    print('--------')
    display_results(token_array, predict_output, 1)
    print('--------')
    r = beam_search(model, conv_features1, token_to_word, max_caption_length, 0, stop_symbol, vocab_size, beam_width=2)
    for entry in r[0:2]:
        print([token_to_word[t] for t in entry[1]])
        print('BLEU score = %f'.format(gen_ngrams(entry[1], 4), [gen_ngrams([int(input_tokens[0][i])])]))
    print('--------')
    r = beam_search(model, conv_features2, token_to_word, max_caption_length, 0, stop_symbol, vocab_size, beam_width=2)
    for entry in r[0:2]:
        print([token_to_word[t] for t in entry[1]])
        print('BLEU score = %f'.format(gen_ngrams(entry[1], 4), [gen_ngrams([int(input_tokens[1][i])])]))
    
if __name__ == '__main__':
    quick_test()
