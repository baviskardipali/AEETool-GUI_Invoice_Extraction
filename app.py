"""
This is the main file to initialize flask and render html files through flask. It is the master file and calls all other secondary file to perform the extraction function.
"""

from flask import Flask, render_template, request, send_from_directory, jsonify, current_app as app
import os
import pandas as pd
import shutil
import config
import predict
import joblib
from transformers import TFRobertaModel, TFBertModel, TFDistilBertModel, TFAlbertModel
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import pdf_to_img
import warnings
from gevent.pywsgi import WSGIServer

warnings.filterwarnings("ignore")

app = Flask(__name__)

max_len = 512


def masked_ce_loss(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 17))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def create_model_roberta(num_tags):
    # BERT encoder
    encoder = TFRobertaModel.from_pretrained('roberta-base')
    # encoder = TFBertModel.from_pretrained("bert-base-uncased")
    # encoder = TFRobertaForTokenClassification.from_pretrained('roberta-base')

    # NER Model
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
    embedding = encoder(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )[0]
    embedding = layers.Dropout(0.2)(embedding)
    embedding = layers.BatchNormalization()(embedding)
    embedding = layers.Dropout(0.2)(embedding)
    embedding = layers.BatchNormalization()(embedding)
    tag_logits = layers.Dense(num_tags+1, activation='softmax')(embedding)

    model = keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[tag_logits],
    )
    optimizer = keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(optimizer=optimizer, loss=masked_ce_loss,
                  metrics=['accuracy'])
    return model


def create_model_bert(num_tags):
    # BERT encoder
    # encoder = TFRobertaModel.from_pretrained('roberta-base')
    encoder = TFBertModel.from_pretrained("bert-base-uncased")
    # encoder = TFRobertaForTokenClassification.from_pretrained('roberta-base')

    # NER Model
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
    embedding = encoder(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )[0]
    embedding = layers.Dropout(0.2)(embedding)
    embedding = layers.BatchNormalization()(embedding)
    embedding = layers.Dropout(0.2)(embedding)
    embedding = layers.BatchNormalization()(embedding)
    tag_logits = layers.Dense(num_tags+1, activation='softmax')(embedding)

    model = keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[tag_logits],
    )
    optimizer = keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(optimizer=optimizer, loss=masked_ce_loss,
                  metrics=['accuracy'])
    return model


def create_model_distilbert(num_tags):
    # encoder
    encoder = TFDistilBertModel.from_pretrained("distilbert-base-uncased")

    # NER Model
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    #token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
    embedding = encoder(
        input_ids, attention_mask=attention_mask
    )[0]
    tag_logits = layers.Dense(num_tags+1, activation='softmax')(embedding)

    model = keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=[tag_logits],
    )
    optimizer = keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(optimizer=optimizer, loss=masked_ce_loss,
                  metrics=['accuracy'])
    return model

def create_model_albert(num_tags):
    # encoder
    encoder = TFAlbertModel.from_pretrained("albert-base-v2")
    # encoder = TFBertModel.from_pretrained("bert-base-uncased")
    # encoder = TFRobertaForTokenClassification.from_pretrained('roberta-base')

    ## NER Model
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
    embedding = encoder(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )[0]
    embedding = layers.Dropout(0.2)(embedding)
    embedding =layers.BatchNormalization()(embedding)
    embedding = layers.Dropout(0.2)(embedding)
    embedding =layers.BatchNormalization()(embedding)
    tag_logits = layers.Dense(num_tags+1, activation='softmax')(embedding)
    
    model = keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[tag_logits],
    )
    optimizer = keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(optimizer=optimizer, loss=masked_ce_loss, metrics=['accuracy'])
    return model


model_bert = create_model_bert(num_tags=23)
model_roberta = create_model_roberta(num_tags=23)
model_distilbert = create_model_distilbert(num_tags=23)
model_albert = create_model_albert(num_tags=23)
print("model created")

model_bert.load_weights(config.MODEL_PATH_BERT)
model_roberta.load_weights(config.MODEL_PATH_ROBERTA)
model_distilbert.load_weights(config.MODEL_PATH_DISTILBERT)
model_albert.load_weights(config.MODEL_PATH_ALBERT)
print("weights loaded")

meta_data_bert = joblib.load(config.META_MODEL_PATH_BERT)
meta_data_roberta = joblib.load(config.META_MODEL_PATH_ROBERTA)
meta_data_distilbert = joblib.load(config.META_MODEL_PATH_DISTILBERT)
meta_data_albert = joblib.load(config.META_MODEL_PATH_ALBERT)
print("bin file loaded")

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
def main():
    """Loads the main paage - Index.html
    """
    return render_template('index.html')


@app.route('/upload1', methods=['POST'])
def upload1():
    """Helps upload the pdf to the OCR functions. It ensures that the pdf is passed to image conversion function and then the latter  is passed through the OCR extractor.
    """
    global result
    pdf_target = os.path.join(APP_ROOT, 'static/pdf')
    img_target = os.path.join(APP_ROOT, 'static/pdf-images')

    # Preparing directory - pdf
    if not os.path.isdir(pdf_target):
        os.mkdir(pdf_target)

    # Preparing directory - image
    if not os.path.isdir(img_target):
        os.mkdir(img_target)

    # Uploading File
    for file in request.files.getlist('file'):
        filename = file.filename
        destination = "/".join([pdf_target, filename])
        file.save(destination)
        sentence = pdf_to_img.extractor_pytess(destination)
        final_info = predict.get_mapping_bert(
            [sentence], meta_data_bert, model_bert=model_bert)

    # Delete file
    if os.path.isdir(pdf_target):
        shutil.rmtree(pdf_target)

    if os.path.isdir(img_target):
        shutil.rmtree(img_target)

    #final_df = pd.DataFrame.from_dict(final_info,)
    # final_info.to_html('result.html')

    result = final_info
    # return jsonify(final_info)
    return render_template('result.html', result=final_info)


@app.route('/upload2', methods=['POST'])
def upload2():
    """Helps upload the pdf to the OCR functions. It ensures that the pdf is passed to image conversion function and then the latter  is passed through the OCR extractor.
    """
    global result
    pdf_target = os.path.join(APP_ROOT, 'static/pdf')
    img_target = os.path.join(APP_ROOT, 'static/pdf-images')

    # Preparing directory - pdf
    if not os.path.isdir(pdf_target):
        os.mkdir(pdf_target)

    # Preparing directory - image
    if not os.path.isdir(img_target):
        os.mkdir(img_target)

    # Uploading File
    for file in request.files.getlist('file'):
        filename = file.filename
        destination = "/".join([pdf_target, filename])
        file.save(destination)
        sentence = pdf_to_img.extractor_pytess(destination)
        final_info = predict.get_mapping_roberta(
            [sentence], meta_data_roberta, model_bert=model_roberta)

    # Delete file
    if os.path.isdir(pdf_target):
        shutil.rmtree(pdf_target)

    if os.path.isdir(img_target):
        shutil.rmtree(img_target)

    #final_df = pd.DataFrame.from_dict(final_info,)
    # final_info.to_html('result.html')

    result = final_info
    # return jsonify(final_info)
    return render_template('result.html', result=final_info)


@app.route('/upload3', methods=['POST'])
def upload3():
    """Helps upload the pdf to the OCR functions. It ensures that the pdf is passed to image conversion function and then the latter  is passed through the OCR extractor.
    """
    global result
    pdf_target = os.path.join(APP_ROOT, 'static/pdf')
    img_target = os.path.join(APP_ROOT, 'static/pdf-images')

    # Preparing directory - pdf
    if not os.path.isdir(pdf_target):
        os.mkdir(pdf_target)

    # Preparing directory - image
    if not os.path.isdir(img_target):
        os.mkdir(img_target)

    # Uploading File
    for file in request.files.getlist('file'):
        filename = file.filename
        destination = "/".join([pdf_target, filename])
        file.save(destination)
        sentence = pdf_to_img.extractor_pytess(destination)
        final_info = predict.get_mapping_distilbert(
            [sentence], meta_data_distilbert, model_bert=model_distilbert)

    # Delete file
    if os.path.isdir(pdf_target):
        shutil.rmtree(pdf_target)

    if os.path.isdir(img_target):
        shutil.rmtree(img_target)

    #final_df = pd.DataFrame.from_dict(final_info,)
    # final_info.to_html('result.html')

    result = final_info
    # return jsonify(final_info)
    return render_template('result.html', result=final_info)

@app.route('/upload4', methods=['POST'])
def upload4():
    """Helps upload the pdf to the OCR functions. It ensures that the pdf is passed to image conversion function and then the latter  is passed through the OCR extractor.
    """
    global result
    pdf_target = os.path.join(APP_ROOT, 'static/pdf')
    img_target = os.path.join(APP_ROOT, 'static/pdf-images')

    # Preparing directory - pdf
    if not os.path.isdir(pdf_target):
        os.mkdir(pdf_target)

    # Preparing directory - image
    if not os.path.isdir(img_target):
        os.mkdir(img_target)

    # Uploading File
    for file in request.files.getlist('file'):
        filename = file.filename
        destination = "/".join([pdf_target, filename])
        file.save(destination)
        sentence = pdf_to_img.extractor_pytess(destination)
        final_info = predict.get_mapping_albert(
            [sentence], meta_data_albert, model_bert=model_albert)

    # Delete file
    if os.path.isdir(pdf_target):
        shutil.rmtree(pdf_target)

    if os.path.isdir(img_target):
        shutil.rmtree(img_target)

    #final_df = pd.DataFrame.from_dict(final_info,)
    # final_info.to_html('result.html')

    result = final_info
    # return jsonify(final_info)
    return render_template('result.html', result=final_info)

@app.route('/download', methods=['POST'])
def download():
    """Displays the key-values successfully and allows download of the file in excel format. Returns the message 'Downloaded Succesfully' on downloading the file.
    """
    if request.method == 'POST':
        final_df = pd.DataFrame.from_dict(result)
        final_df.to_excel('result.xlsx', index=False)
    return 'Downloaded successfully!'


if __name__ == '__main__':
    # app.run(debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 4000), app)
    http_server.serve_forever()
