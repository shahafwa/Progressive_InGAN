import os

import time

import re

from slackclient.slackclient.client import SlackClient

from io import BytesIO

from utils import *
from util import *

import math

SLACK_BOT_TOKEN = "xoxb-720593983683-731575153188-bIL4gsR90gejUCfun8jCrTUP"
SLACK_OAUTH_TOKEN = "xoxp-720593983683-732043767109-732046846693-421b43deb5e80991ce09bbf8349b0868"



class slack_weizmann(object):

    # instantiate Slack client

    slack_client = SlackClient(SLACK_BOT_TOKEN)

    # starterbot's user ID in Slack: value is assigned after the bot starts up

    starterbot_id = None



    # constants

    RTM_READ_DELAY = 0.4

    EXAMPLE_COMMAND = "do"

    MENTION_REGEX = "^<@(|[WU].+?)>(.*)"



    def __init__(self,conf=None):

        self._last_time = time.time()

        self._channels_list = None
        self.conf = conf

        # self._RTM_READ_DELAY = 1  # 1 second delay between reading from RTM

        # self._EXAMPLE_COMMAND = "do"

        # self._MENTION_REGEX = "^<@(|[WU].+?)>(.*)"



    def _delay_api_calls(self):

        t_end = self._last_time + self.RTM_READ_DELAY

        while time.time() < t_end:

            # print("sleeping for slack rate-limit")

            time.sleep(0.1)



    def _parse_bot_commands(self, slack_events):

        """

            Parses a list of events coming from the Slack RTM API to find bot commands.

            If a bot command is found, this function returns a tuple of command and channel.

            If its not found, then this function returns None, None.

        """

        for event in slack_events:

            if event["type"] == "message" and not "subtype" in event:

                user_id, message = self._parse_direct_mention(event["text"])

                if user_id == self.starterbot_id:

                    return message, event["channel"]

        return None, None



    def _parse_direct_mention(self, message_text):

        """

            Finds a direct mention (a mention that is at the beginning) in message text

            and returns the user ID which was mentioned. If there is no direct mention, returns None

        """

        matches = re.search(self.MENTION_REGEX, message_text)

        # the first group contains the username, the second group contains the remaining message

        return (matches.group(1), matches.group(2).strip()) if matches else (None, None)



    def _list_channels(self):

        if not self._channels_list:

            self._delay_api_calls()

            channels_call = self.slack_client.api_call("channels.list")

            self._last_time = time.time()

            if channels_call['ok']:

                self._channels_list = channels_call['channels']

        return self._channels_list



    def _handle_command(self, command, channel):

        """

            Executes bot command if the command is known

        """

        # Default response is help text for the user

        default_response = "Not sure what you mean. Try *{}*.".format(self.EXAMPLE_COMMAND)



        # Finds and executes the given command, filling in response

        response = None

        # This is where you start to implement more commands!

        if command.startswith(self.EXAMPLE_COMMAND):

            response = "Sure...write some more code then I can do that!"



        # Sends the response back to the channel

        self._delay_api_calls()

        self.slack_client.api_call(

            "chat.postMessage",

            channel=channel,

            text=response or default_response

        )

        self._last_time = time.time()



    def _send_message(self, channel_id, message):

        success_res = ""

        self._delay_api_calls()

        res = self.slack_client.api_call(

            "chat.postMessage",

            channel=channel_id,

            text=message,

            username='pythonbot',

            icon_emoji=':ninja:'

        )

        self._last_time = time.time()

        try:

            success_res = res['ts']

        except:

            pass

        return success_res



    def _update_message(self, channel_id, message, timestamp):

        success_res = ""

        self._delay_api_calls()

        res = self.slack_client.api_call(

            "chat.update",

            channel=channel_id,

            text=message,

            username='pythonbot',

            icon_emoji=':ninja:',

            ts=timestamp

        )

        self._last_time = time.time()

        try:

            success_res = res['ts']

        except:

            pass

        return success_res



    def _create_public_channel(self, ch_name):

        ch_list = self._list_channels()

        channel_exist = False

        response = None

        if ch_list:

            for channel in ch_list:

                if channel['name'] == ch_name:

                    channel_exist = True

                    break

        if not channel_exist:

            self._delay_api_calls()

            response = self.slack_client.api_call(

                "channels.create",

                token=SLACK_OAUTH_TOKEN,

                name=ch_name,

                validate="true"

            )

            self._last_time = time.time()

            if response["ok"] == True:

                if self._channels_list:

                    self._channels_list.append(response["channel"])

                else:

                    self._channels_list = [response["channel"]]

                return True

            else:

                return False

        else:

            return True

    def upload_tensor_image(self,image_tensor,ch_name="gan"):

        pil_im = Image.fromarray(tensor2im(image_tensor), 'RGB')
        x = BytesIO()
        pil_im.save(x, 'png')
        im_bytes = x.getvalue()

        self._delay_api_calls()

        response = self.slack_client.api_call(

            "files.upload",
            file=im_bytes,
            content="Something among man",
            channels=ch_name,
            title="Test upload",
        )
        return response["file"]


    def make_bar(self, percent_done):

        #9608

        # return '%s%s%%' % ((round(percent_done / 5) * chr(1)), percent_done)

        total_chars = 20.0

        full_bar = '#'*int(math.trunc((float(percent_done)*4.0/float(total_chars))))

        empty_bar = '_'*int(math.trunc((float(total_chars) - float(len(full_bar)))))

        return "Epoch progress: [{}{}] {}% ".format(full_bar, empty_bar, percent_done)


    def create_public_ch(self,ch_name):
        self._create_public_channel(ch_name)

    def handle_message(self, message, channel_name, timestamp=None):

        rec_timestamp = ""

        if not self.slack_client.server.connected:

            if self.slack_client.rtm_connect(with_team_state=False):

                channels = self._list_channels()

                if channels:

                    for channel in channels:

                        if channel['name'] == channel_name:

                            if not timestamp:

                                rec_timestamp = self._send_message(channel['id'], message)

                            else:

                                rec_timestamp = self._update_message(channel['id'], message, timestamp)

                            return rec_timestamp

        else:

            channels = self._list_channels()

            if channels:

                for channel in channels:

                    if channel['name'] == channel_name:

                        if not timestamp:

                            rec_timestamp = self._send_message(channel['id'], message)

                        else:

                            rec_timestamp = self._update_message(channel['id'], message, timestamp)

                        return rec_timestamp



if __name__ == "__main__":


    slack._create_public_channel("test2")

    ts1 = slack.handle_message("shay","test2")

    slack = slack_weizmann()

    slack._create_public_channel("test2")

    ts2 = slack.handle_message("shay2", "test2")

    temp = 3

