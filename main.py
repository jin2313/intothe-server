import os
import torch
import gluonnlp as nlp

from flask import Flask, request, jsonify
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
from text_sentiment_detect import predict
from text_sentiment_detect import BERTDataset
from text_sentiment_detect import BERTClassifier



app = Flask(__name__)
host_addr = '0.0.0.0'
port_num = '5000'



def load_model():
    global model, tok, device
    device = torch.device("cpu") # cpu 사용
    bertmodel, vocab = get_pytorch_kobert_model()
    PATH = './model_text_senti/'
    model = torch.load(PATH + 'Text_Sentiment_KoBERT.pt', map_location=device)
    model.load_state_dict(torch.load(PATH + 'Text_Sentiment_state_dict.pt', map_location=device))
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)



@app.route('/text_sentiment/', methods=['GET'])
def text_senti():
    parameter_dict = request.args.to_dict()
    if len(parameter_dict) != 1:
        return '에러 발생'

    value = request.args.get('text', type=str)
    emotion = predict(model, tok, device, value)
    return emotion


@app.route('/face_recognition/', methods=['GET', 'POST'])
def face_recog():
    if request.method == 'GET':
        return '''<form action="/face_recognition/" method="POST" enctype = "multipart/form-data">
            <input type = "file" name = "file" />            
            <input type='submit'>
            '''
    else:
        file = request.files['file']
        f1 = file.read()
        name = file.filename
        img_path = './image/recog_img'
        model_path = './model_face_recog/runs'
        with open(f'{img_path}/{name}', 'wb') as F:
            F.write(f1)
        os.system(f'python ./model_face_recog/detect.py --source "{img_path}/{name}" --save-conf --save-txt --exist-ok --weights "{model_path}/train/exp/weights/best.pt"')
        
        try:
            f2 = open(f'{model_path}/detect/exp/labels/{name[:-3]}txt', 'r', encoding='UTF-8')
        except:
            return jsonify({})
        lines = f2.readlines()
        json_response = {}
        line_li = []
        for line in lines:
            line_li.append(line.split())
        for i in range(5):
            class_li = []
            for result in line_li:
                if result[0] == str(i):
                    class_li.append(result[5])
            if len(class_li) == 0:
                continue
            json_response[str(i)] = class_li
                       
        return jsonify(json_response)



@app.route('/face_sentiment/', methods=['GET', 'POST'])
def face_senti():
    if request.method == 'GET':
        return '''<form action="/face_sentiment/" method="POST" enctype = "multipart/form-data">
            <input type = "file" name = "file" />            
            <input type='submit'>
            '''
    else:
        file = request.files['file']
        f1 = file.read()
        name = file.filename
        img_path = './image/senti_img'
        model_path = './model_face_senti/runs'
        with open(f'{img_path}/{name}', 'wb') as F:
            F.write(f1)
        os.system(f'python ./model_face_senti/detect.py --source "{img_path}/{name}" --save-conf --save-txt --exist-ok --weights "{model_path}/train/exp/weights/best.pt"')
        try:
            f2 = open(f'{model_path}/detect/exp/labels/{name[:-3]}txt', 'r', encoding='UTF-8')
        except:
            return jsonify({})
        lines = f2.readlines()
        json_response = {}
        line_li = []
        for line in lines:
            line_li.append(line.split())
        for i in range(7):
            class_li = []
            for result in line_li:
                if result[0] == str(i):
                    class_li.append(result[5])
            if len(class_li) == 0:
                continue
            json_response[str(i)] = class_li
                       
        return jsonify(json_response)



if __name__ == "__main__":
    load_model()
    app.run(host=host_addr, port=port_num, debug=True)