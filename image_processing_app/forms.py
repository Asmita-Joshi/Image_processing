from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
class SignupForm(UserCreationForm):
    class Meta:
        model = User 
        fields = ['username', 'password1', 'password2']

class LoginForm(forms.Form):
    username = forms.CharField()
    password = forms.CharField(widget=forms.PasswordInput)
class CustomUserCreationForm(UserCreationForm):
    # Add custom validation if needed
    pass
class ProcessImageForm(forms.Form):
    OPERATIONS_CHOICES = [
        ('crop', 'Crop'),
        ('rotate', 'Rotate'),
        ('brightness', 'Adjust Brightness'),
        ('noise_reduction', 'Noise Reduction'),
        ('greyscale', 'Convert to Greyscale'),
        ('dilation', 'Dilation'),
        ('erosion', 'Erosion'),
        ('emotion_detection', 'Emotion Detection'),
        ('face_detection', 'Face Detection'),
        ('histogram', 'Histogram'),

    ]
    operations = forms.MultipleChoiceField(choices=OPERATIONS_CHOICES, widget=forms.CheckboxSelectMultiple)
    image_source = forms.ChoiceField(choices=[('live', 'Live Image'), ('upload', 'Upload Image')])
