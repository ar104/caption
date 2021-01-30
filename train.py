import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import model
import data


def make_vocab():
    vocab, image_to_tokens = data.build_annotations_vocab('/home/aroy/notebooks/experiments/Flickr8k_text/Flickr8k.token.txt')
    with open('/datadrive/flickr8k/Flickr8k.vocab.txt', 'w') as f:
        for w, index in vocab.items():
            f.write('{},{}\n'.format(w, index))
    with open('/datadrive/flickr8k/Flickr8k.image_to_tokens.txt', 'w') as f:
        for w, tokens in image_to_tokens.items():
            f.write('{}'.format(w))
            for t in tokens:
                f.write(',{}'.format(t))
            f.write('\n')

    print('Vocab built.')

def preprocess(train_fname, test_fname):
    print('Preprocessing data. Executing eagerly:{}'.format(tf.executing_eagerly()))
    token_to_word = data.load_annotations_vocab('/datadrive/flickr8k/Flickr8k.vocab.txt')
    vocab_size=len(token_to_word)
    stop_symbol=vocab_size - 1
    image_to_tokens = data.load_annotations_tokens('/datadrive/flickr8k/Flickr8k.image_to_tokens.txt', stop_symbol)
    max_caption_length=max([len(t) for t in image_to_tokens.values()])
    dataset = tf.data.Dataset.list_files('/datadrive/flickr8k/Flicker8k_Dataset/*.jpg')
    dataset = dataset.shuffle(96)
    base_model = model.make_vgg16_model()
    def load_image(fname):
        img_path = bytes.decode(fname.numpy())
        img_array = tf.convert_to_tensor(data.load_image(img_path, include_batch=True))
        img_name = img_path.split('/')[-1]
        ret = [tf.convert_to_tensor(np.asarray([len(image_to_tokens[img_name])]))]
        token_list = data.pad(image_to_tokens[img_name], max_caption_length, stop_symbol)
        ret.append(img_array)
        for t in token_list:
            ret.append(tf.convert_to_tensor(np.asarray([t])))
        return ret
    def save_items(ds, fp_image, fp_caption):
        saved = 0
        for one in ds:
            caption_len = one[0][0]
            conv_features = base_model.predict(one[1], batch_size=1)
            conv_features = np.reshape(conv_features, newshape = (196, 512))
            np.save(fp_image, conv_features, allow_pickle=True)
            np.save(fp_caption, one[2][:caption_len], allow_pickle=True)
            saved += 1
            print("Saved = {}".format(saved))
    
    dataset = dataset.map(lambda x: tf.py_function(load_image, [x], [tf.int64, tf.float32] + [tf.int64]*max_caption_length))
    dataset = dataset.map(lambda *x: (x[0], x[1], tf.concat(x[2:], axis=-1)))
    val_dataset = dataset.take(192)
    train_dataset = dataset.skip(192)
    train_image_fp = open(train_fname + '_image', "wb")
    train_caption_fp = open(train_fname + '_caption', "wb")
    test_image_fp = open(test_fname + '_image', "wb")
    test_caption_fp = open(test_fname + '_caption', "wb")
    save_items(val_dataset, test_image_fp, test_caption_fp)
    save_items(train_dataset, train_image_fp, train_caption_fp)
    # print(dataset.cardinality().numpy())


def ds_gen(file_image, file_caption, max_caption_length, stop_symbol, batch_size):
    data_buffers = {}
    data_images = []
    data_captions = []

    def preload(max_count=64):
        # Preload loads the data and batches so that each batch has a uniform caption length.
        fp_image = open(file_image, 'rb')
        fp_caption = open(file_caption, 'rb')
        count = 0
        def shiftbuf(buflist):
            data_captions.append(np.stack([item[0] for item in buflist], axis=0))
            data_images.append(np.stack([item[1] for item in buflist], axis=0))
        
        while count != max_count:
            try:
                image = np.load(fp_image, allow_pickle=True)
                captions = np.load(fp_caption, allow_pickle=True)
                bufkey = captions.shape[0]
                if bufkey not in data_buffers:
                    data_buffers[bufkey] = []
                if len(data_buffers[bufkey]) == batch_size:
                    shiftbuf(data_buffers[bufkey])
                    data_buffers[bufkey] = []
                padded_captions = np.pad(captions, pad_width=(0, max_caption_length - captions.shape[0]), constant_values=stop_symbol)
                data_buffers[bufkey].append((padded_captions, image))
            except (OSError, IOError):
                break
            count += 1
        sponge = []
        for bufkey in data_buffers.keys():
            while len(data_buffers[bufkey]) > 0:
                sponge.append(data_buffers[bufkey].pop())
                if len(sponge) == batch_size:
                    shiftbuf(sponge)
                    sponge = []
        if len(sponge) == batch_size:
            shiftbuf(sponge)
            sponge = []
        # Note: Drops remaining items in sponge that don't make a full batch.
            
    def generator():
        for (im, cap) in zip(data_images, data_captions):
            yield (im, cap)
        return
    print("preloading ...")
    preload(max_count=-1)
    print("done!")
    return lambda: generator()
    
def train():
    print(device_lib.list_local_devices())
    token_to_word = data.load_annotations_vocab('/datadrive/flickr8k/Flickr8k.vocab.txt')
    vocab_size=len(token_to_word)
    stop_symbol=vocab_size - 1
    image_to_tokens = data.load_annotations_tokens('/datadrive/flickr8k/Flickr8k.image_to_tokens.txt', stop_symbol)
    max_caption_length=max([len(t) for t in image_to_tokens.values()])
    caption_model = model.make_model(1024, 100, stop_symbol, max_caption_length, vocab_size, dropout_rate=0.3)
    train_dataset = tf.data.Dataset.from_generator(ds_gen('blah_train_image', 'blah_train_caption', max_caption_length, stop_symbol, 32),
                                                   output_types=(tf.float32, tf.int64), output_shapes=((32, 196, 512), (32, max_caption_length)))
                                                   #output_signature = (tf.TensorSpec(shape=(1, None), dtype=tf.float32), tf.TensorSpec(shape=(max_caption_length,))))
    val_dataset = tf.data.Dataset.from_generator(ds_gen('blah_test_image', 'blah_test_caption', max_caption_length, stop_symbol, 32),
                                                   output_types=(tf.float32, tf.int64), output_shapes=((32, 196, 512), (32, max_caption_length)))
                                                   #output_signature = (tf.TensorSpec(shape=(1, None), dtype=tf.float32), tf.TensorSpec(shape=(max_caption_length,))))
    train_dataset = train_dataset.map(lambda *x: tuple([tuple([x[0], x[1]]), x[1][:, 1:]]))
    #for v in train_dataset:
    #    print(v)
    #    exit(-1)
    val_dataset = val_dataset.map(lambda *x: tuple([tuple([x[0], x[1]]), x[1][:, 1:]]))
    train_dataset = train_dataset.apply(tf.data.experimental.prefetch_to_device('/gpu:0'))
    val_dataset = val_dataset.apply(tf.data.experimental.prefetch_to_device('/gpu:0'))

    caption_model.fit(train_dataset, epochs=200,
                      callbacks = [tf.keras.callbacks.TensorBoard('./logs', update_freq=1),
                                   tf.keras.callbacks.ModelCheckpoint(filepath='caption_model.h5', monitor='val_loss', verbose=1, save_best_only=True)],
                      validation_data=val_dataset)

def check_perf():
    print(device_lib.list_local_devices())
    token_to_word = data.load_annotations_vocab('/datadrive/flickr8k/Flickr8k.vocab.txt')
    vocab_size=len(token_to_word)
    stop_symbol=vocab_size - 1
    image_to_tokens = data.load_annotations_tokens('/datadrive/flickr8k/Flickr8k.image_to_tokens.txt', stop_symbol)
    max_caption_length=max([len(t) for t in image_to_tokens.values()])
    caption_model = model.make_model(1024, 100, stop_symbol, max_caption_length, vocab_size, dropout_rate=0.3)
    caption_model.load_weights('caption_model.269-1.12.h5')
    val_dataset = tf.data.Dataset.from_generator(lambda: ds_gen('blah_test_image', 'blah_test_caption', max_caption_length, stop_symbol, 1),
                                                   output_types=(tf.float32, tf.int64), output_shapes=((1, 196, 512), (1, max_caption_length)))
    val_dataset = val_dataset.map(lambda *x: tuple([tuple([x[0], x[1]]), x[1][:, 1:]]))
    for ex in val_dataset:
        #print(ex[1])
        out  = ''
        for i in range(max_caption_length - 1):
            sym = ex[1].numpy()[0][i]
            #print(sym, token_to_word[sym])
            out = out + token_to_word[sym] + ' '
        print(out)
        out = ''
        r = caption_model.predict(ex)
        #print(r)
        for i in range(max_caption_length - 1):
            sym = np.argmax(r[0][i])
            #print(sym, r[0][i][sym], token_to_word[sym])
            out = out + token_to_word[sym] + ' '
        print(out)
        r = model.beam_search(caption_model, ex[0][0], max_caption_length, 0, stop_symbol, vocab_size, beam_width=5)
        for entry in r[0:5]:
            print([token_to_word[t] for t in entry[1]])
        print('------------------------')
        #break
    
    
if __name__ == '__main__':
    #preprocess('blah_train', 'blah_test') 
    #make_vocab()
    train()
    #check_perf()
