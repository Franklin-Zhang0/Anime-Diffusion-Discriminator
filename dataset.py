from __future__ import unicode_literals
from pybooru import Danbooru
from random import randint
import urllib.request
import os


x = []  # link storage


def download(tags, pages):
    try:
        print("Logging in...")
        client = Danbooru('danbooru', username='Franklin_zhang', api_key='Ddb9G3cfGweHakS3FDu5noj2')
        print("Logged in!")
        # Collect links
        while len(x) != 2000:  # Checks if the list is full
            print("Collecting links...", len(x), "/ 2000")
            randompage = randint(1, pages)
            posts = client.post_list(tags=tags, page=randompage, limit=100)
            for post in posts:
                try:
                    fileurl = post['file_url']
                except:
                    fileurl = 'https://danbooru.donmai.us' + post['source']
                x.append(fileurl)

        dirlist = os.listdir('./dataset/human')
        dirlist = [int(i.split('.')[0]) for i in dirlist]
        if len(dirlist) == 0:
            baseindex = 0
        else:
            baseindex = sorted(dirlist)[-1]
            
        # Download images
        for i,url in enumerate(x):
            try:
                print("Downloading image {0} of {1}, index:{2} ,url: {3}".format(i, len(x), i+baseindex, url))
                urllib.request.urlretrieve(url, "./dataset/human/{0}.jpg".format(i+baseindex))
            except:
                continue
    except Exception as e:
        raise e


def main():
    # pages: 2000 Gold account limit. Basic Users should have 1000
    download(tags='1girl rating:s', pages=1000)


main()