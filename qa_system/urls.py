from django.urls import path
from . import views
urlpatterns = [
    path('', views.chat, name='chat'),
    path('person_setting/', views.person_setting, name='person_setting'),
    path('qa/knowledgebase_create/', views.create_knowledgebase, name='knowledgebase_create'),
    path('qa/knowledgebase_list/', views.get_knowledgebases, name='knowledgebase_list'),
    path('qa/update_params/', views.update_params, name='update_params'),
    path('qa/load_knowledgebase/', views.load_knowledgebase, name='load_knowledgebase'),
    path('qa/unload_knowledgebase/', views.unload_knowledgebase, name='unload_knowledgebase'),
    path('qa/smart_query/',views.smart_query, name='smart_query'),
]