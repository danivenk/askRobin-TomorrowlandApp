from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from django.urls import get_resolver
from .llm_script import query


def index(request):
    """
    show the chat with input view
    """

    messages = query.chat_messages

    if request.method == "POST":
        post_query = request.POST.get("query", None)
        try:
            new_chat = int(request.POST.get("new", 0))
        except ValueError:
            new_chat = 0
        if post_query is not None:
            messages.append({"role": "user", "content": post_query})
            messages = query.generate(messages)
        if new_chat:
            messages.clear()
        return redirect("index")

    query.chat_messages = messages
    context = {"messages": reversed(messages)}
    return render(request, "tomorrow/index.html", context)
