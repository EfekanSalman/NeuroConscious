#
# Copyright (c) 2025 Efekan Salman
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
This module contains the Message class used for communication between agents.
"""

class Message:
    """
    A data class representing a message sent between agents.
    """
    def __init__(self, sender: str, recipient: str, content: str):
        """
        Initializes the message object.

        Args:
            sender (str): The name of the entity sending the message.
            recipient (str): The name of the entity receiving the message.
            content (str): The textual content of the message.
        """
        self.sender = sender
        self.recipient = recipient
        self.content = content

    def __str__(self):
        """
        Returns a human-readable string representation of the message.
        """
        return f"From: {self.sender}, To: {self.recipient}, Content: '{self.content}'"
