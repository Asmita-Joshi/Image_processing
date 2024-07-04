from django.db import models
class ProcessedImage(models.Model):
    input_image = models.ImageField(upload_to='images/')
    processed_image = models.ImageField(upload_to='images/', blank=True, null=True)
# Create your models here.
