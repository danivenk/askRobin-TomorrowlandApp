from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.urls import get_resolver

def index(request):
    print(get_resolver().url_patterns)
    # template = loader.get_template("tomorrow/index.html")
    return render(request, "tomorrow/index.html")
