from crypt import methods
from fileinput import filename
from flask import Flask, redirect, url_for, render_template, request, flash, send_file, jsonify
from cartoonize import generate_cartoonize_img
from PIL import Image
from matplotlib import pyplot as plt
from werkzeug.utils import secure_filename
import base64
import numpy as np
import io, os, cv2
import time
import uuid

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
def uniqueImgName(imgName, isGenerated:bool):     
        # split img name after from extension
        imgName_list = imgName.split('.')       

        # function is call for generated image
        if isGenerated:
            ImgName_unique = imgName_list[0] + "_generated" + f".{imgName_list[1]}"
        
        # if function is call first time(for make uploaded image name unique)
        else:
            unique_string = str(uuid.uuid4())
            ImgName_unique = imgName_list[0] + f"_{unique_string}" + f".{imgName_list[1]}"

        return ImgName_unique


# def ndarray_to_b64(ndarray):
#     """
#     converts a np ndarray to a b64 string readable by html-img tags 
#     """
#     img = cv2.cvtColor(ndarray, cv2.COLOR_RGB2BGR)
#     _, buffer = cv2.imencode('.png', img)
#     return base64.b64encode(buffer).decode('utf-8')        


@app.route('/')
def home():    
    return render_template('index.html')


@app.route('/submit', methods=['POST', 'GET'])
def submit():

    global uploadedImage
    global generated_art_path
    global generated_art_name
    
    # checking from all file, whether 'img' is selected or not
    if 'img' not in request.files:
        resp = jsonify({'message':'No file in the request'})
        resp.status_code = 400   # 400 ---> Bad Request
        return resp

    
    uploadedImage = request.files["img"]

    if uploadedImage.filename == '':        
        resp = jsonify({'message':'No image selected for uploading'})
        resp.status_code = 400
        return resp


# secure_filename ---> pass it a filename and it will return
#  a secure version of it. 
# secure_filename("../../../etc/passwd")
#  ---- 'etc_passwd'

    if uploadedImage and allowed_file(uploadedImage.filename):
        ImgName = secure_filename(uploadedImage.filename)

#     
    # save image with unique name, So to not face conflict
    #  while selecting image

        ImgName_unique = uniqueImgName(ImgName, isGenerated=False)
        

        uploadedImage.save(os.path.join(app.config['UPLOAD_FOLDER'], ImgName_unique))
        print('Image successfully uploaded')
        # flash('Image successfully uploaded and displayed below')

        # getting image path from static folder
        img_path = f"./static/{ImgName_unique}"

        # loading img in numpy array and converting datatype to uint8
        img_numpy = plt.imread(img_path)
        img_numpy = img_numpy.astype(np.uint8)   
        # print(f"\nShape of img before reshaping is: {img_numpy.shape}\n")

        # if img_numpy.shape[2] > 3:
        #     img_numpy.reshape((-1, img_numpy.shape[1], 3))

        # print(f"\nShape of img after reshaping is: {img_numpy.shape}\n")

        # call generate_cartoonize_img, it returns image in numpy array            
        cartoon_img = generate_cartoonize_img(img_numpy, LINE_WIDTH, BLUR_VALUE, TOTAL_COLORS, EPOCHS, ACCURACY)
        

        # setting unique name for cartoon_img, to avoid name conflict
        generated_art_name = uniqueImgName(ImgName_unique, isGenerated=True)

        
        # generated_art_path = f"./art_generated/{generated_art_name}"
        generated_art_path = f"./static/{generated_art_name}"

        # gen_img_path = f"./art_generated/{generated_art_name}"

        cartoon_img = Image.fromarray(cartoon_img)
        cartoon_img.save(generated_art_path)
        

        # print(f"\nType of cartoon image is: {type(cartoon_img,)}\n")
        
        # print(f"\n{(ImgName_unique)}\n")
        # print(f"\n{(generated_art_name)}\n")
        # print(f"\n{(generated_art_path)}\n")

        return render_template('index.html', art_file = generated_art_name, filename=ImgName_unique)            
            # filename = ImgName_unique

    else:
        return jsonify({"Message":"Allowed image types are - png, jpg, jpeg"})
        # flash('Allowed image types are - png, jpg, jpeg')
        # return redirect(request.url)
        
        

# @app.route('/art_generated/<filename>')
# def display_image(ImgName):        
#     return redirect(url_for('art_generated', filename= ImgName), code=301)



# @app.route('/cartoonized/<ImgFileName>')
# def cartoonizedImg(ImgFileName):
#     pass


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

