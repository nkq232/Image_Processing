import numpy
import cv2


#Hàm resize lại ảnh 
def image_resize(image,width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
       
    else:
        r = width / float(w)
        dim = (width, int(h * r))
       

    resized_img = cv2.resize(image, dim, interpolation = inter)
    

    return resized_img

# Resize ảnh image, cắt lấy template
image = cv2.imread("1.jpg");
image = image_resize(image, height = 600);

#Danh sách template
horse = image[385:450,487:568]
ice_cream = image[298:374, 488:536]
boat = image[288:380, 561:632]
car = image[301:362, 640:714]
balloon = image[297:367, 732:805]
wa_melon = image[399:441, 598:666]
train = image[371:441, 685:761]
grape = image[461:512, 491:558]
tie = image[461:501, 577:638]
cake = image[462:508, 677:729]
bear = image[432:507, 760:816]
tennis = image[526:565, 497:539]
straw = image[522:566, 575:618]
rabbit = image[508:581, 647:711]
duck = image[518:574, 739:805]
listTemplate = [horse, ice_cream, boat, car, balloon, wa_melon, train, grape, tie, tennis, cake, bear, straw, rabbit, duck];

#Xử lý ảnh

imagex = image[0:600,0:465]
image_gray = cv2.cvtColor(imagex, cv2.COLOR_BGR2GRAY);

#image_canny = cv2.Canny(image_gray, 50, 200);

#cv2.imshow("Gray Image", image_gray)

(H,W) = image_gray.shape[::]

for temp in listTemplate:
    # Xử lý template

    template = temp
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY);
    template_canny = cv2.Canny(template_gray, 50, 200);
    (tH,tW) = template_gray.shape[::];
    #cv2.imshow("Edge Detect Template", template_canny);    

    #Xử lý scale
    catch = None 
    for scale in numpy.linspace(0.2, 1.0, 60)[::-1]:
        resized= image_resize(image_gray, int(W/scale))
        r = W / float(W/scale)
        
        #resize_template = cv2.Canny(template, 50, 200);
        
        canny = cv2.Canny(resized, 50, 200);
        result = cv2.matchTemplate(canny, template_canny,cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        if catch is None or maxVal > catch[0]:
            catch = (maxVal, maxLoc, r)

    (_,maxLoc,r) = catch
    (startX, startY) = (int(maxLoc[0]*r), int(maxLoc[1]*r))
    (endX, endY) = (int((maxLoc[0] + tW)*r), int((maxLoc[1] + tH)*r))
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)


cv2.imshow("Matched image", image)
cv2.waitKey(0)