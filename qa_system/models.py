from django.db import models


# Create your models here.
class KnowledgeBase(models.Model):
    name = models.CharField(max_length=100)
    create_time = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name