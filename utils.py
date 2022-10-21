#@title Imports and Notebook Utilities {vertical-output: true}

import os
import io
import PIL
import PIL.Image, PIL.ImageDraw
# import base64
import numpy as np
from datetime import datetime

# import torch
# import requests
# from io import BytesIO

from time import time
import cv2

# device = torch.device('cuda')

def draw_text_on_image(img, text):
    img = img.transpose((1,2,0))
    img = PIL.Image.fromarray((img*255.).astype('uint8'), 'RGB')

    # Call draw Method to add 2D graphics in an image
    I1 = PIL.ImageDraw.Draw(img)
    # try:
    #     font = PIL.ImageFont.truetype(r'/usr/share/fonts/truetype/humor-sans/Humor-Sans.ttf', 17)
    # except:
    #     pass
    if os.path.exists('/usr/share/fonts/truetype/humor-sans/Humor-Sans.ttf'):
        font = PIL.ImageFont.truetype(r'/usr/share/fonts/truetype/humor-sans/Humor-Sans.ttf', 17)
    else:
        try:
            from matplotlib.font_manager import findfont, FontProperties
            font = findfont(FontProperties(family=['monospace']))
            font = PIL.ImageFont.truetype(font, 17)
        except:
            font = None


    # Add Text to an image
    I1.text((5, 5), text, fill=(255, 255, 255), font=font)

    # Display edited image
    # img.show()

    return np.array(img).transpose(2,0,1)/255.



def to_gif(canvases, fn='/animation.gif', duration=250):
    #imgs = [PIL.Image.fromarray((img.transpose((1,2,0))*255.).astype(np.uint8)) for img in canvases]
    imgs = []
    for i in range(len(canvases)):
      if True:
          np_img = (np.clip(canvases[i], 0, 1).transpose((1,2,0))*255.).astype(np.uint8)

          imgs.append(PIL.Image.fromarray(np_img))
    # duration is the number of milliseconds between frames; this is 40 frames per second
    # imgs[0].save(fn, save_all=True, append_images=imgs[1:], duration=50, loop=0)
    imgs[0].save(fn, save_all=True, append_images=imgs[1:], duration=duration, loop=0)

def to_video(frames, fn=None, frame_rate=4):
    #if fn is None: fn = '/content/drive/MyDrive/animations/{}.mp4'.format(time())
    if fn is None:
        import datetime
        date_and_time = datetime.datetime.now()
        run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
        fn = '/content/{}.mp4'.format(run_name)
    h, w = frames[0].shape[1], frames[0].shape[2]
    print(h,w)
    _fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # _fourcc = cv2.VideoWriter_fourcc(*'H264')
    # _fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(fn, _fourcc, frame_rate, (w,h))
    for frame in frames:
        cv2_frame = np.clip(frame, a_min=0, a_max=1)
        cv2_frame = (cv2_frame * 255.).astype(np.uint8).transpose((1,2,0))[:,:,::-1]
        out.write(cv2_frame)
    out.release()
    return fn

# def imread(url, max_size=None, mode=None):
#   if url.startswith(('http:', 'https:')):
#     r = requests.get(url)
#     f = io.BytesIO(r.content)
#   else:
#     f = url
#   img = PIL.Image.open(f)
#   if max_size is not None:
#     img = img.resize((max_size, max_size))
#   if mode is not None:
#     img = img.convert(mode)
#   img = np.float32(img)/255.0
#   return img

# def np2pil(a):
#   if a.dtype in [np.float32, np.float64]:
#     a = np.uint8(np.clip(a, 0, 1)*255)
#   return PIL.Image.fromarray(a)

# def imwrite(f, a, fmt=None):
#   a = np.asarray(a)
#   if isinstance(f, str):
#     fmt = f.rsplit('.', 1)[-1].lower()
#     if fmt == 'jpg':
#       fmt = 'jpeg'
#     f = open(f, 'wb')
#   np2pil(a).save(f, fmt, quality=95)

# def imencode(a, fmt='jpeg'):
#   a = np.asarray(a)
#   if len(a.shape) == 3 and a.shape[-1] == 4:
#     fmt = 'png'
#   f = io.BytesIO()
#   imwrite(f, a, fmt)
#   return f.getvalue()

# def im2url(a, fmt='jpeg'):
#   encoded = imencode(a, fmt)
#   base64_byte_string = base64.b64encode(encoded).decode('ascii')
#   return 'data:image/' + fmt.upper() + ';base64,' + base64_byte_string

# def imshow(a, fmt='jpeg'):
#   display(Image(data=imencode(a, fmt)))



# from torchvision import utils
# def show_img(img):
#     img = np.transpose(img, (1, 2, 0))
#     img = np.clip(img, 0, 1)
#     img = np.uint8(img * 254)
#     # img = np.repeat(img, 4, axis=0)
#     # img = np.repeat(img, 4, axis=1)
#     pimg = PIL.Image.fromarray(img, mode="RGB")
#     imshow(pimg)

# def zoom(img, scale=4):
#   img = np.repeat(img, scale, 0)
#   img = np.repeat(img, scale, 1)
#   return img

# class VideoWriter:
#   def __init__(self, filename='_autoplay.mp4', fps=30.0, **kw):
#     self.writer = None
#     self.params = dict(filename=filename, fps=fps, **kw)

#   def add(self, img):
#     img = np.asarray(img)
#     if self.writer is None:
#       h, w = img.shape[:2]
#       self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
#     if img.dtype in [np.float32, np.float64]:
#       img = np.uint8(img.clip(0, 1)*255)
#     if len(img.shape) == 2:
#       img = np.repeat(img[..., None], 3, -1)
#     self.writer.write_frame(img)

#   def close(self):
#     if self.writer:
#       self.writer.close()

#   def __enter__(self):
#     return self

#   def __exit__(self, *kw):
#     self.close()
#     if self.params['filename'] == '_autoplay.mp4':
#       self.show()

#   def show(self, **kw):
#       self.close()
#       fn = self.params['filename']
#       display(mvp.ipython_display(fn, **kw))
