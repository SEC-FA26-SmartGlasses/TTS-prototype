import pytesseract
from PIL import Image


config_oem_psm = r'--oem 3 --psm 11' 
# psm 6 for "single uniform block of text, best for paragraphs/labels" 
# assuming input "blocks" are boxed text in image

# if there are multiple boxes of text in the image,
# return an array of strings where each element is the text from that box
def img_to_text(boxes):
    strings = []
    for n in boxes:
        # timeout=10 (seconds)
        # output_type=pytesseract.Output.STRING
        # n can be "either PIL Image, NumPy array or file path of the image to be processed by Tesseract"
        strings.append(pytesseract.image_to_string(n["processed_crop"], config=config_oem_psm))
    return strings

def unprocessed_img_to_text(img):
    return pytesseract.image_to_string(img, config=config_oem_psm)

if __name__ == "__main__":
    test_img = Image.open(r"C:\Users\zoome\Downloads\tesseract_test2.jpg")

    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    print(pytesseract.__version__) # verify install
    print(unprocessed_img_to_text(test_img))