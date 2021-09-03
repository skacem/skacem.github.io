#!/usr/bin/env python

'''
This script creates tags for your Jekyll blog hosted by Github page.
Only if tags are written as follows:
---
tags:
  - tag1
  - tag2
  ...
---
No plugins required.
'''

import glob
import os
import re

post_dir = '_posts/'
tag_dir = 'tag/'

filenames = glob.glob(post_dir + '*md')

total_tags = []
for filename in filenames:
    if filename.split("/")[1].startswith("_"):
        continue
    print(filename)
    f = open(filename, 'r', encoding="utf8")
    crawl = False
    next_line_tag = False
    for line in f:
        content = line.strip()
        if (line.strip() == '---' and crawl == False):
            crawl = True
            continue
        if line.strip() == 'tags:':
            next_line_tag = True
            continue

        if (re.search(r'\s*?-{1}\s+\S', line) and next_line_tag == True):
            tag = re.search('(?<=-\s)(\S.*)', line).string.strip()
            tag = [tag.replace('- ', '')]
            total_tags += tag

        if (line.strip() == '---' and crawl == True):
            break
    total_tags

    f.close()
total_tags = set(total_tags)

old_tags = glob.glob(tag_dir + '*.md')
for tag in old_tags:
    os.remove(tag)

for tag in total_tags:
    tag_filename = tag_dir + tag + '.md'
    f = open(tag_filename, 'a')
    write_str = '---\nlayout: tagpage\ntitle: \"Tag: ' + tag + \
        '\"\ntag: ' + tag + '\ncategory: tag' + '\nrobots: noindex\n---\n'
    f.write(write_str)
    f.close()
print("Tags generated, count", total_tags.__len__())
