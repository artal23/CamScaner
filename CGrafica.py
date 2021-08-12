from flask import Flask
from flask import current_app, flash, jsonify, make_response, redirect, request, url_for
from flask_restful import Resource, Api, reqparse
from imageio import imread
import werkzeug
import numpy as np
import io
import cv2
import base64
import requests
import sys

app = Flask(__name__)
api = Api(app)

def blur_and_threshold(gray):
    gray = cv2.GaussianBlur(gray,(3,3),2)
    threshold = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    threshold = cv2.fastNlMeansDenoising(threshold, 11, 31, 9)
    return threshold

#Contorno mas grande para evitar errores
def biggest_contour(contours,min_area):
    biggest = None
    max_area = 0
    biggest_n=0
    approx_contour=None
    for n,i in enumerate(contours):
            area = cv2.contourArea(i)
                     
            if area > min_area/10:
                    peri = cv2.arcLength(i,True)
                    approx = cv2.approxPolyDP(i,0.02*peri,True)
                    if area > max_area and len(approx)==4:
                            biggest = approx
                            max_area = area
                            biggest_n=n
                            approx_contour=approx
                            
                                                   
    return biggest_n,approx_contour


def order_points(pts):
    # ordena los puntos en caso de que el usuario los cruce
    pts=pts.reshape(4,2)
    ordered = np.zeros((4, 2), dtype = "float32")

    # se calcula con la suma de sus componentes
    s = pts.sum(axis = 1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]

    # se calcula como la diferencia de los puntos 
    diff = np.diff(pts, axis = 1)
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered

def four_point_transform(image, pts):
    
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    #Distancia de los puntos de  arriba izquierda y derecha para calcular los anchos
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
   
    #Distancia de los puntos de  arriba izquierda y derecha para calcular los altos
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    #recortamos la imagen
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # aplicamos la transformacion perpectiva apartir de los puntos
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def get_points(image):
    image=image.copy()  
    height, width, channels = image.shape
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_size=gray.size
    threshold=blur_and_threshold(gray)
    # se convierte a escala de grises y se hace un suavizado con thresholding
    # para hallar los contornos de la imagen
    edges = cv2.Canny(threshold,50,150,apertureSize = 7)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    simplified_contours = []

    for cnt in contours:
        hull = cv2.convexHull(cnt)
        simplified_contours.append(cv2.approxPolyDP(hull,
                                0.001*cv2.arcLength(hull,True),True))
    simplified_contours = np.array(simplified_contours)
    # encontramos el contorno mas grande
    biggest_n,approx_contour = biggest_contour(simplified_contours,image_size)
    threshold = cv2.drawContours(image, simplified_contours ,biggest_n, (255, 255, 255), 1)
    
    dst = 0
    if approx_contour is not None and len(approx_contour)==4:
        approx_contour=np.float32(approx_contour)
        return approx_contour
    return None

def transformation(image):
    image=image.copy()  
    height, width, channels = image.shape
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_size=gray.size
    threshold=blur_and_threshold(gray)
    # se convierte a escala de grises y se hace un suavizado con thresholding
    # para hallar los contornos de la imagen
    edges = cv2.Canny(threshold,50,150,apertureSize = 7)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    simplified_contours = []

    for cnt in contours:
        hull = cv2.convexHull(cnt)
        simplified_contours.append(cv2.approxPolyDP(hull,
                                0.001*cv2.arcLength(hull,True),True))
    simplified_contours = np.array(simplified_contours)
    # encontramos el contorno mas grande
    biggest_n,approx_contour = biggest_contour(simplified_contours,image_size)
    threshold = cv2.drawContours(image, simplified_contours ,biggest_n, (255, 255, 255), 1)
    
    dst = 0
    if approx_contour is not None and len(approx_contour)==4:
        approx_contour=np.float32(approx_contour)
        # cortamos la imagen
        dst=four_point_transform(threshold,approx_contour)
    croppedImage = dst
    return croppedImage


# correcion del brillo

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img  


#Darle mas definicion utilizando un kernel 


def final_image(rotated):
    kernel_sharpening = np.array([[0,-1,0], 
                                [-1, 5,-1],
                                [0,-1,0]])
    #Se aplica el kernel
    sharpened = cv2.filter2D(rotated, -1, kernel_sharpening)
    sharpened=increase_brightness(sharpened,30)  
    return sharpened

def delete_shadow(cleaned_image):
    shadow = cleaned_image 
    rgb_planes = cv2.split(shadow)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    return result_norm

class EdgesProcessImageEndpoint(Resource):
  
    # This method is called when we send a POST request to this endpoint
    def post(self):
        response = request.json
        datauri = response['image']
        header, encoded = datauri.split(",", 1)
        imgbits = base64.b64decode(encoded)
        im_arr = np.frombuffer(imgbits, dtype=np.uint8) 
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        pts=get_points(img)
        print(pts[0][0])
        print(pts[1][0])
        print(pts[2][0])
        print(pts[3][0])    
        result = np.array(pts).tolist()
        print (result)
        #cv2.imwrite("saved_file.jpg", img)
        return jsonify({"image": "result"})




class ProcessImageEndpoint(Resource):
  
    # This method is called when we send a POST request to this endpoint
    def post(self):
        response = request.json
        datauri = response['image']
        header, encoded = datauri.split(",", 1)
        imgbits = base64.b64decode(encoded)
        im_arr = np.frombuffer(imgbits, dtype=np.uint8) 
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        points = response['points']
        #pts= a
        pts=np.zeros((4,(1) , 2))
        pts[0][0]=points[0]
        pts[1][0]=points[1]
        pts[2][0]=points[2]
        pts[3][0]=points[3]
        # read like a stream
        blurred_threshold = four_point_transform(img, pts)
        cleaned_image = final_image(blurred_threshold)
        buffer = cv2.imencode('.jpg', cleaned_image)[1]
        encoded_image1 = base64.b64encode(buffer).decode() #codificando imagen con sombra 
        dict = {
            'img': encoded_image1
        }
        no_shadow = delete_shadow(cleaned_image)  #imagen sin sombra
        buffer2 = cv2.imencode('.jpg', no_shadow)[1]
        encoded_image2 = base64.b64encode(buffer2)#codificando imagen sin sombra

        return jsonify({"image1": dict  })


               
api.add_resource(ProcessImageEndpoint, '/image')
api.add_resource(EdgesProcessImageEndpoint, '/imagepoints')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000 ,debug=True)

