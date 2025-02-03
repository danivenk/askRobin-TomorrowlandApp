from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from django.urls import get_resolver

# save the messages of the chat
messages = []


def index(request):
    """
    show the chat with input view
    """

    if request.method == "POST":
        query = request.POST.get("query", None)
        try:
            new_chat = int(request.POST.get("new", 0))
        except ValueError:
            new_chat = 0
        if query is not None:
            messages.append({"role": "user", "content": query})
        if new_chat:
            messages.clear()
        return redirect("index")
    # print(get_resolver().url_patterns)
    # template = loader.get_template("tomorrow/index.html")
    context = {"messages": reversed(messages)}
    print(context)
    return render(request, "tomorrow/index.html", context)
