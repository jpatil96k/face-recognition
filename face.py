import cv2

# Load Haarcascade properly
a = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
b = cv2.VideoCapture(0)

while True:
    c_rec, d_img = b.read()
    if not c_rec:
        break

    # Convert to grayscale
    e = cv2.cvtColor(d_img, cv2.COLOR_BGR2GRAY)
    f = a.detectMultiScale(e, 1.3, 6)

    for (x1, y1, w1, h1) in f:
        # Use one fixed proper color (green)
        color = (0, 255, 0)
        cv2.rectangle(d_img, (x1, y1), (x1 + w1, y1 + h1), color, 3)

        # Add label text
        cv2.putText(d_img, "SMILE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Smart Face Detector', d_img)

    # Press Q to exit
    h = cv2.waitKey(40) & 0xff
    if h == ord('q'):
        break

b.release()
cv2.destroyAllWindows()
