from django.db import models
from django.contrib.auth.models import User

class Classification(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=255)
    description = models.TextField()
    document = models.FileField(upload_to='documents/')
    author = models.CharField(max_length=255)
    depth = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

class ClassificationDetail(models.Model):
    classification = models.ForeignKey(Classification, on_delete=models.CASCADE)
    detail = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Detail for {self.classification.title}"
