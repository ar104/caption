import tensorflow as tf
import numpy as np
import data

def make_model(lstm_units, embedding_size, max_caption_length, vocab_size):
    vgg_model = tf.keras.applications.vgg16.VGG16()
    base_model = tf.keras.Model(inputs=vgg_model.input, outputs=tf.keras.layers.Reshape(target_shape=(196, 512))(vgg_model.get_layer('block5_conv3').output))
    base_model.trainable = False
    conv_features = base_model(vgg_model.input)
    h_init = tf.keras.layers.Dense(units=lstm_units, activation = 'relu')(tf.keras.layers.Flatten()(conv_features))
    c_init = tf.keras.layers.Dense(units=lstm_units, activation = 'relu')(tf.keras.layers.Flatten()(conv_features))
    lstm = tf.keras.layers.LSTM(units=lstm_units, return_state=True)
    attention = tf.keras.layers.Attention()
    token_inputs = [tf.keras.Input(shape=(1,)) for _ in range(max_caption_length)]
    input_embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim = embedding_size)
    output_net = tf.keras.layers.Dense(units = vocab_size, activation = 'softmax')
    input_transformation = tf.keras.layers.Dense(units=lstm_units)
    output_symbols = []
    for i in range(max_caption_length - 1):
        attention_input = attention([tf.keras.layers.Reshape(target_shape=(1, 512))(h_init), conv_features])
        token_input = input_embedding(token_inputs[i])
        token_input  = tf.keras.layers.Reshape(target_shape=(1, embedding_size))(token_input)
        lstm_input = tf.keras.layers.concatenate([attention_input, token_input])
        output, h_init, c_init = lstm(inputs=lstm_input, initial_state=[h_init, c_init])
        output_symbols.append(tf.keras.layers.Reshape(target_shape=(1, vocab_size))(output_net(output)))
    output_array = tf.keras.layers.concatenate(output_symbols, axis = -2)
    final_model = tf.keras.Model(inputs=[base_model.input] + token_inputs, outputs=output_array)
    final_model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy')
    return final_model

# Quick Test
def quick_test():
    def display_results(img_array, input_tokens, output_tokens, index):
        print([token_to_word[int(t[index])] for t in input_tokens])
        print([token_to_word[np.argmax(output_tokens[index][i])] for i in range(output_tokens.shape[1])])
    vocab, image_to_tokens = data.build_annotations_vocab('/home/aroy/notebooks/experiments/Flickr8k_text/Flickr8k.token.txt')
    with open('/home/aroy/notebooks/experiments/Flickr8k_text/Flickr8k.vocab.txt', 'w') as f:
        for w, index in vocab.items():
            f.write('{},{}\n'.format(w, index))
    with open('/home/aroy/notebooks/experiments/Flickr8k_text/Flickr8k.image_to_tokens.txt', 'w') as f:
        for w, tokens in image_to_tokens.items():
            f.write('{}'.format(w))
            for t in tokens:
                f.write(',{}'.format(t))
            f.write('\n')
    token_to_word = data.load_annotations_vocab('/home/aroy/notebooks/experiments/Flickr8k_text/Flickr8k.vocab.txt')
    vocab_size=len(token_to_word)
    stop_symbol=vocab_size - 1
    image_to_tokens = data.load_annotations_tokens('/home/aroy/notebooks/experiments/Flickr8k_text/Flickr8k.image_to_tokens.txt', stop_symbol)
    max_caption_length=max([len(t) for t in image_to_tokens.values()])
    model = make_model(512, 8, max_caption_length, vocab_size)
    img_array1 = data.load_image('/home/aroy/notebooks/experiments/Flickr8k_Dataset/Flicker8k_Dataset/1001773457_577c3a7d70.jpg')
    img_array2 = data.load_image('/home/aroy/notebooks/experiments/Flickr8k_Dataset/Flicker8k_Dataset/760180310_3c6bd4fd1f.jpg')
    img_array = np.concatenate([img_array1, img_array2], axis=0)
    token_array1 = data.pad(image_to_tokens['1001773457_577c3a7d70.jpg'], max_caption_length, stop_symbol)
    token_array2 = data.pad(image_to_tokens['760180310_3c6bd4fd1f.jpg'], max_caption_length, stop_symbol)
    token_array = []
    for i in range(len(token_array1)):
        token_array.append(np.asarray([[token_array1[i]],[token_array2[i]]]))
    test_input = [img_array] + token_array
    test_output_list = token_array[1:]
    test_output = np.concatenate(test_output_list, axis = -1)
    model.fit(test_input, test_output, batch_size=2, epochs=10)
    predict_output = model.predict(test_input, batch_size=2)
    display_results(img_array, token_array, predict_output, 0)


if __name__ == '__main__':
    quick_test()
