from django.db import models

# Create your models here.
class Images(models.Model):
    image_low_res = models.ImageField(null = True , blank=False, upload_to="low_res")
    image_high_res = models.ImageField(null = True , blank=False, upload_to="high_res")