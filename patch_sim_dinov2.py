import os
from PIL import Image
from skimage.registration import optical_flow_ilk
import skimage
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import PIL.ImageColor as ImageColor
import numpy as np
from transformers import AutoImageProcessor, Dinov2Model
import torch
import torch.nn as nn
import pandas as pd
import time

cos = nn.CosineSimilarity(dim=1, eps=1e-6)

thickness = 10
det_str = ''

color1 = ImageColor.getrgb('green')


# color1 = [mcolors.hsv_to_rgb(color_base + np.array([1-soft, 0, 0])).astype('int') for soft in softmax_score_good_cls_target]

color_tup = (color1[0], color1[1], color1[2])


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
  """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top + total_display_str_height
  else:
    text_bottom = bottom # + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin


def cos_sim_crop_to_patch(im_crop, features_semantic_crop, model):
  inputs = image_processor(im_crop, return_tensors="pt")
  with torch.no_grad():
    outputs = model(**inputs)

  features_semantic_part = outputs.pooler_output
  return cos(features_semantic_part, features_semantic_crop)

result_dir = '/content/drive/MyDrive/Colab Notebooks/mask_grounding'
cropped_dir = '/content/drive/MyDrive/Colab Notebooks/mask_grounding/data/cropped_tiff'
original_dir = '/content/drive/MyDrive/Colab Notebooks/mask_grounding/data/original_tiff'

filenames_crop = [os.path.join(cropped_dir, x) for x in os.listdir(cropped_dir)
                  if x.endswith('tif')]
filenames_orig = [os.path.join(original_dir, x) for x in os.listdir(original_dir)
                  if x.endswith('tif')]
# g = 'DF148_P5_L111_x50xH_HS2Y-01_proc-Scene-010-M279-E03'
# [i for i, x in enumerate(filenames_crop) if g.lower() in x.lower()]

image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
backbone=False
if backbone:
  model = Dinov2Backbone.from_pretrained("facebook/dinov2-base", out_indices=[0,1,2,3])
else:
  model = Dinov2Model.from_pretrained("facebook/dinov2-base")

unique_run_name = str(int(time.time()))
mask_en = False
debug = True
box_dict = dict()
res = list()

for inx, file in enumerate(filenames_crop):
  # if inx != 33 :
  #   continue

  print(file) 
  qual_crop = file.split('proc-Scene-')[-1].split('Create Image Subset')[0].lower()
  orig_file = [x for x in filenames_orig if qual_crop[:-1] in x.lower()][0]
  print(os.path.basename(orig_file))
  print(os.path.basename(file))

  im_orig = Image.open(orig_file)
  im_orig = im_orig.convert('RGB')
  im_crop = Image.open(file)
  im_crop = im_crop.convert('RGB')
  print(im_crop.size, im_orig.size)

  if (im_crop.size[0]<100 or im_crop.size[1]<100):
    print('Too small crop')

  n_subblocks_w = 1 + 1 + im_orig.size[0]//(im_crop.size[0])
  n_subblocks_h = 1 + 1 + im_orig.size[1]//(im_crop.size[1])

  inputs = image_processor(im_crop, return_tensors="pt")

  with torch.no_grad():
      outputs = model(**inputs)
  features_semantic_crop = outputs.pooler_output

  # coarse loop
  sim_all = list()
  coordination_coarse = list()
  for i in range(n_subblocks_w):
    for j in range(n_subblocks_h):
      left = i*im_crop.size[0]//2
      up = j*im_crop.size[1]//2
      right = int(min((i+1)*im_crop.size[0]*1.5, im_orig.size[0]))
      down = int(min((j+1)*im_crop.size[1]*1.5, im_orig.size[1]))
      
      if (left>=right) or (down-up) < im_crop.size[1] or ((right-left) < im_crop.size[0]):
        print('skip ROI',(left,up,right,down))
        continue

      coordination_coarse.append((left,up,right,down))
      inputs = image_processor(im_orig.crop((left,up,right,down)), return_tensors="pt")
      with torch.no_grad():
        outputs = model(**inputs)

      features_semantic_part = outputs.pooler_output
      cos_sim = cos(features_semantic_part, features_semantic_crop)
      print((left,up,right,down), cos_sim)

      sim_all.append(cos_sim)
  
  coarse_roi_ind = np.argmax(sim_all)
  left_, up_, right_, down_= coordination_coarse[coarse_roi_ind]
  print('cand area', coordination_coarse[coarse_roi_ind])
  max_score = max(sim_all)

  # binary search x axis left wise
  if left_>0:
    low = 0
    high = max(0, left_ - 1)
    mid = high // 2

    max_score = max(sim_all)
    print('Init low ; high; max_score', low, high, max_score)

    while low <= high:

        mid_elem_score = cos_sim_crop_to_patch(im_orig.crop((mid, up_, mid+im_crop.size[0], down_)), 
                                          features_semantic_crop, model)
        print('mid_elem_score', mid_elem_score)
        if mid_elem_score < max_score:
          low = mid + 1
        else:
          max_score = mid_elem_score
          high = mid - 1
        print('low ; high; max_score', low, high, max_score)
        
        mid = low + (high - low)//2
        
        print('mid', mid)
  
  if max_score <= max(sim_all): # no better x axis location then try on the right
    # binary search x axis right wise
    if right_ < im_orig.size[0]: # in the image boundary
      low = left_
      high = im_orig.size[0] - im_crop.size[0] -1
      mid = low + (high-low) // 2

      
      print('Init rightwise :  low ; high; max_score', low, high, max_score)

      while low <= high:

          mid_elem_score = cos_sim_crop_to_patch(im_orig.crop((mid, up_, mid+im_crop.size[0], down_)), 
                                            features_semantic_crop, model)
          assert(mid+im_crop.size[0] <im_orig.size[0])
          if mid_elem_score < max_score:
            low = mid + 1
          else:
            max_score = mid_elem_score
            high = mid - 1
          
          print('low ; high; max_score', low, high, max_score)
          
          mid_hyp = low + (high - low)//2
          # binary search till rightmost - crop width
          if (mid_hyp+im_crop.size[0] >= im_orig.size[0]):
            break
          else:
            mid = low + (high - low)//2

          print('mid', mid)

  # inputs = image_processor(im_orig, return_tensors="pt")
  # with torch.no_grad():
  #     outputs = model(**inputs)
  # features_semantic_orig = outputs.pooler_output

# left_, up_, right_, down_
  if debug:
    draw_bounding_box_on_image(im_orig, up_, mid ,
                              down_ +im_crop.size[1] ,  mid +im_crop.size[0],
                                color=color_tup,
                                thickness=thickness,
                                display_str_list=det_str,
                              use_normalized_coordinates=False)
    im_orig.save(os.path.join(result_dir,qual_crop + '_Act_cls_tiles_over_blind.png'))

  res.append({'file':os.path.basename(orig_file),
              'x_top_left' :mid, 'y_top_left': up_,
              'width' :im_crop.size[0], 'height':im_crop.size[1]})
  print(res)
  print('top_left_offset', mid ,up_,
                            mid +im_crop.size[0], down_ +im_crop.size[1])
  if (inx % 10) == 0:
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(result_dir, unique_run_name + 'grounding_prediction.csv'), index=False)
    
df = pd.DataFrame(res)
df.to_csv(os.path.join(result_dir, unique_run_name + 'grounding_prediction.csv'), index=False)
