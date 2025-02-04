from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from django.urls import get_resolver
from .llm_script import query


def index(request):
    """
    show the chat with input view
    """

    # set the local messages to the global messages
    messages = query.chat_messages

    # check request method
    if request.method == "POST":

        # get post value
        post_query = request.POST.get("query", None)

        # if new chat requested set to true
        try:
            new_chat = int(request.POST.get("new", 0))
        except ValueError:
            new_chat = 0

        # if there is a query then generate an answer
        if post_query is not None:
            messages.append({"role": "user", "content": post_query})
            messages = query.generate(messages)

        # if new chat is requested clear messages
        if new_chat:
            messages.clear()

        return redirect("index")

    # put local messages in global and prepare for template context
    query.chat_messages = messages
    context = {"messages": reversed(messages)}

    return render(request, "tomorrow/index.html", context)
