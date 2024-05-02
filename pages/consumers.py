from channels.generic.websocket import AsyncWebsocketConsumer
import json
from django.template.loader import get_template


class UploadProgressConsumer(AsyncWebsocketConsumer):

    async def connect(self):

        self.group_name = 'upload'

        await self.channel_layer.group_add(self.group_name, self.channel_name)

        await self.accept()

    async def disconnect(self, close_code):

        await self.channel_layer.group_discard(self.group_name, self.channel_name)


    async def receive(self, text_data):

        text_data_json = json.loads(text_data)

        message = text_data_json["message"]

        await self.channel_layer.group_send(

            self.group_name, {"type": "chat.message", "message": message}

        )


    async def send_message(self, event):

        html = get_template("components/upload-notification.html").render(

            context={"progress": event['progress'], 'uploaded': event['uploaded']}

        )

        await self.send(text_data=html)


    async def send_process_message(self, event):

        html = get_template("components/process-notification.html").render(

            context={"progress": event['progress'], 'uploaded': event['uploaded'], 'id': event['id']}

        )

        await self.send(text_data=html)

    async def send_sub_process_message(self, event):

        html = get_template("components/sub-process-notification.html").render(

            context={"progress": event['progress'], 'uploaded': event['uploaded'], 'id': event['id']}

        )

        await self.send(text_data=html)