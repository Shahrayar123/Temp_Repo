from crypt import methods
from fileinput import filename
from sqlalchemy import false
# from email.mime import image
from flask import Flask, redirect, url_for, render_template, request, flash, send_file, jsonify
# from cartoonize import generate_cartoonize_img
from PIL import Image
from matplotlib import pyplot as plt
from werkzeug.utils import secure_filename
import base64
import numpy as np
import io, os, cv2
import time
import uuid
from styleTransfer import generateNeuralStyleImage

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/"
app.secret_key = "secret_key"


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# img_path = "./horse.jpg"
LINE_WIDTH  = 7
BLUR_VALUE = 5
TOTAL_COLORS = 4
EPOCHS = 50
ACCURACY = 0.02


# return boolean value after checking whether the file extension is valid or not
def allowed_file(filename):
    return ('.' in filename) and (filename.split('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


# this function will take image name and return unique image name, which help to avoid name conflict
def uniqueImgName(imgName, isContent=False, isStyle=False, isGenerated=False):     
        # split img name after from extension
        imgName_list = imgName.split('.')       


        if isContent:
            unique_string = str(uuid.uuid4())
            ImgName_unique = imgName_list[0] + f"_{unique_string}"+ "_CONTENT" + f".{imgName_list[1]}"
        
        
        if isStyle:
            unique_string = str(uuid.uuid4())
            ImgName_unique = imgName_list[0] + f"_{unique_string}"+ "_STYLE" + f".{imgName_list[1]}"

        
        if isGenerated:
            ImgName_unique = imgName_list[0] + "_GENERATED" + f".{imgName_list[1]}"
    

        # else:
        #     unique_string = str(uuid.uuid4())
        #     ImgName_unique = imgName_list[0] + f"_{unique_string}" + f".{imgName_list[1]}"

        return ImgName_unique



@app.route('/')
def home():    
    return render_template('index.html')



@app.route('/submit', methods=['POST', 'GET'])
def submit():
    print("Submit method invoked")

    global uploadedImage_content, uploadedImage_style
    global generated_art_path
    global generated_art_name
    
    # checking from all file, whether 'content_img' and 'style_img' is selected or not
    if ('content_img' not in request.files) and ('style_img' not in request.files):        
        resp = jsonify({'message':'No file part in the request'})
        resp.status_code = 400
        return resp
    
    uploadedImage_content = request.files["content_img"]
    uploadedImage_style = request.files["style_img"]


    if (uploadedImage_content.filename == '') and (uploadedImage_style.filename == ''):        
        resp = jsonify({'message':'No image selected for uploading'})
        resp.status_code = 400
        return resp


    if (uploadedImage_content and allowed_file(uploadedImage_content.filename)) and (uploadedImage_style and allowed_file(uploadedImage_style.filename)):        
        content_ImgName = secure_filename(uploadedImage_content.filename)

    # if uploadedImage_style and allowed_file(uploadedImage_style.filename):
        style_ImgName = secure_filename(uploadedImage_style.filename)
    

#   ------------------------------------------------        
#     
    # save image with unique name, So to not face conflict
    #  while selecting image


        content_ImgName_unique = uniqueImgName(content_ImgName, isContent=True)
        style_ImgName_unique = uniqueImgName(style_ImgName, isStyle=True)
        
        
        
#   ------------------------------------------------

        uploadedImage_content.save(os.path.join(app.config['UPLOAD_FOLDER'], content_ImgName_unique))
        uploadedImage_style.save(os.path.join(app.config['UPLOAD_FOLDER'], style_ImgName_unique))
        flash('Image successfully uploaded and displayed below')


        # getting image path from static folder
        content_img_path = f"./static/{content_ImgName_unique}"
        style_img_path = f"./static/{style_ImgName_unique}"

        art_img = generateNeuralStyleImage(content_img_path, style_img_path)
        

        # using content ImgName for generated art name with _generated as end of name
        generated_art_name = uniqueImgName(content_ImgName_unique, isGenerated=True)

        
        # generated_art_path = f"./art_generated/{generated_art_name}"
        generated_art_path = f"./static/{generated_art_name}"

        # cartoon_img = Image.fromarray(cartoon_img)
        art_img = art_img.astype(np.uint8)
        art_img = Image.fromarray(art_img)
        art_img.save(generated_art_path)

        # plt.figure(figsize=(10,7))
        # plt.imshow(art_img)
        # plt.axis('off')
        # plt.show();
        
    


        return render_template('index.html', art_file = generated_art_name)            
            # filename = ImgName_unique

    else:        
        return jsonify({"Message":"Allowed image types are - png, jpg, jpeg"})
        
        
        

@app.route('/download', methods=['POST', 'GET'])
def download_art():    
    # if generated_art_path:    
    return send_file(generated_art_path, as_attachment=True, cache_timeout=0, attachment_filename=generated_art_name)
        

@app.route('/share')
def share_art():
    """
    Display social media platform (facebook, gmail, twitter) link to
    share generated art on it
    """    
    pass


@app.route('/buy')
def buy_art():
    """
    Display form that contains information about required frame type.
    When user submit form then email will be recieved to developer  
    """

    # if art will generated then try block will be executes,
    #  otherwise except block will execute    
    try:
        return render_template("buy_art_form.html", art_file = generated_art_name)
    except:
        return render_template("buy_art_form.html",)
    


if __name__ == "__main__":        
    app.run(debug=True)

