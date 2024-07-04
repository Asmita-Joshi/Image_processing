import cv2
import numpy as np
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, HttpResponseServerError, StreamingHttpResponse
from django.contrib.auth.forms import UserCreationForm
from django.conf import settings
from django.views.decorators import gzip
from .forms import ProcessImageForm
from .models import ProcessedImage
from matplotlib import pyplot as plt
from deepface import DeepFace
import os
from django.contrib.auth import authenticate, login, logout 
from .forms import SignupForm, LoginForm
def user_signup(request):
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = SignupForm()
    return render(request, 'signup.html', {'form': form})

# login page
def user_login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user:
                login(request, user)    
                return redirect('home')
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})

# logout page
def user_logout(request):
    logout(request)
    return redirect('login')
# OpenCV video capture
cap = cv2.VideoCapture(0)

def firstpage(request):
    if request.method == 'POST':
        form = ProcessImageForm(request.POST, request.FILES)
        if form.is_valid():
            return render(request, 'firstpage.html', {'form': form, 'message': 'Form submitted successfully!'})
    else:
        form = ProcessImageForm()
    return render(request, 'firstpage.html', {'form': form})

def afterlogin(request):
    if request.method == 'POST':
        form = ProcessImageForm(request.POST, request.FILES)
        if form.is_valid():
            return render(request, 'afterlogin.html', {'form': form, 'message': 'Form submitted successfully!'})
    else:
        form = ProcessImageForm()
    return render(request, 'afterlogin.html', {'form': form})

def index(request):
    return render(request, 'miniindex.html')

def result(request):
    return render(request, 'miniresult.html')

def home(request):
    processed_image = None  # Set processed_image to None initially
    if request.method == 'POST':
        form = ProcessImageForm(request.POST, request.FILES)
        if form.is_valid():
            if 'input_image' in request.FILES:  # Check if an image file is selected
                processed_image_path = process_image(request.FILES['input_image'], form.cleaned_data['operations'])
                processed_image = ProcessedImage.objects.create(input_image=request.FILES['input_image'], processed_image=processed_image_path)
            elif 'live_image' in request.FILES:  # Check if a live image is captured
                ret, frame = cap.read()
                processed_image_path = process_image(frame, form.cleaned_data['operations'])
                processed_image = ProcessedImage.objects.create(input_image=None, processed_image=processed_image_path)
                
            context = {'form': form, 'processed_image': processed_image}
            return render(request, 'home.html', context)
    else:
        form = ProcessImageForm()
    return render(request, 'home.html', {'form': form, 'processed_image': processed_image})

def process_image(input_image, operations):
    # Get the original image name
    original_image_name = input_image.name
    
    # Read the image
    img = cv2.imdecode(np.frombuffer(input_image.read(), np.uint8), cv2.IMREAD_COLOR)

    # Perform the selected operations on the image
    for operation in operations:
        if operation == 'crop':
            height, width, _ = img.shape
            start_row, start_col = int(height * 0.25), int(width * 0.25)
            end_row, end_col = int(height * 0.75), int(width * 0.75)
            img = img[start_row:end_row, start_col:end_col]
        elif operation == 'rotate':
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif operation == 'brightness':
            img = np.clip(img + 50, 0, 255)
        elif operation == 'noise_reduction':
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif operation == 'greyscale':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif operation == 'dilation':
            kernel = np.ones((3, 3), np.uint8)
            img = cv2.dilate(img, kernel, iterations=1)
        elif operation == 'erosion':
            kernel = np.ones((3, 3), np.uint8)
            img = cv2.erode(img, kernel, iterations=1)
        elif operation == 'face_detection':
            # Perform face detection using Haar cascades
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
        elif operation == 'histogram':
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist_hue = cv2.calcHist([hsv], [0], None, [256], [0, 256])
            hist_saturation = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            hist_value = cv2.calcHist([hsv], [2], None, [256], [0, 256])
            plt.figure(figsize=(8, 6))
            plt.subplot(311)
            plt.plot(hist_hue, color='r')
            plt.title('Hue Histogram')
            plt.subplot(312)
            plt.plot(hist_saturation, color='g')
            plt.title('Saturation Histogram')
            plt.subplot(313)
            plt.plot(hist_value, color='b')
            plt.title('Value Histogram')
            plt.tight_layout()
            plt.show()


    # Save the processed image
    processed_image_path = 'images/' + original_image_name
    processed_image_full_path = os.path.join(settings.MEDIA_ROOT, processed_image_path)
    cv2.imwrite(processed_image_full_path, img)

    return processed_image_path

def download_image(request, image_id):
    processed_image = get_object_or_404(ProcessedImage, id=image_id)
    with open(processed_image.processed_image.path, 'rb') as f:
        response = HttpResponse(f.read(), content_type='image/jpeg')
        response['Content-Disposition'] = 'attachment; filename="{}"'.format(processed_image.input_image.name)
        return response

def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'registration/signup.html', {'form': form})

def dashboard(request):
    processed_images = ProcessedImage.objects.all()
    return render(request, 'dashboard.html', {'processed_images': processed_images})
    


# Function to stream video frames
def stream_camera():
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Perform operations on the frame (example: convert to grayscale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Decorator to gzip the stream
@gzip.gzip_page
def live_camera(request):
    try:
        return StreamingHttpResponse(stream_camera(), content_type="multipart/x-mixed-replace;boundary=frame")
    except HttpResponseServerError as e:
        print("aborted")

# Release the camera and
# Release the camera and destroy OpenCV windows on exit
def cleanup():
    cap.release()
    cv2.destroyAllWindows()

# Run the cleanup function on exit
import atexit
atexit.register(cleanup)
cap = cv2.VideoCapture(0)
