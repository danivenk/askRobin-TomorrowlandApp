from django.apps import AppConfig
from django.conf import settings
from .llm_script import query
from dotenv import load_dotenv
import os


class TomorrowConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'tomorrow'

    def ready(self):
        """
        preload the json & .env files
        """
        load_dotenv(dotenv_path=os.path.join(settings.BASE_DIR, settings.ENV))
        query.load_json(os.path.join(settings.BASE_DIR, 'data/tomorrowland.json'))
