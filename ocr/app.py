import io
import os

from ocr import ocr
#import ocr
import time
import shutil
import base64
import numpy as np
from PIL import Image, ImageFilter
from glob import glob
import sys

from flask import Flask, request, render_template, session, send_from_directory, jsonify

#import your_python_module  # 这里替换成你实际编写的Python模块




app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    output_text = ""
    if request.method == "POST":

       # image_bytes = image_file.read()
        #image_format = image_file.content_type.split("/")[-1]
        #output_text = your_python_module.process_image(image_bytes, image_format)

        #获取上传的文件图片
        #image_file = request.files["image"]
        #image_data = base64.b64encode(image_file.read()).decode("utf-8")
        #print(type(image_data))
        #print(type(image_file.read()))


        #image_img = request.files['prntimg']
        testimage = request.files['image']

        result,image_framed = single_pic_proc(testimage)

        test = Image.fromarray(image_framed)

        #假设有一个PIL.Image.Image对象test
        img_bytes_io = io.BytesIO()
        test.save(img_bytes_io, format='PNG')
        img_bytes = img_bytes_io.getvalue()
        image_data = base64.b64encode(img_bytes).decode("utf-8")
        #print(type(test))



        for key in result:
            output_text = output_text + result[key][1] + '\n'



        return render_template("index.html", output_text=output_text,image_data=image_data)
    else:
        return render_template("index.html")





def single_pic_proc(image_file):
    image = np.array(Image.open(image_file).convert('RGB'))
    result, image_framed = ocr(image)
    return result,image_framed





if __name__ == "__main__":
    output_text = ""
    app.run(debug=True)