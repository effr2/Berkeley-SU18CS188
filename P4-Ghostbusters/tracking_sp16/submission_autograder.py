#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from codecs import open
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

"""
CS 188 Local Submission Autograder
Written by the CS 188 Staff

==============================================================================
   _____ _              _ 
  / ____| |            | |
 | (___ | |_ ___  _ __ | |
  \___ \| __/ _ \| '_ \| |
  ____) | || (_) | |_) |_|
 |_____/ \__\___/| .__/(_)
                 | |      
                 |_|      

Modifying or tampering with this file is a violation of course policy.
If you're having trouble running the autograder, please contact the staff.
==============================================================================
"""
import bz2, base64
exec(bz2.decompress(base64.b64decode('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')))

