from transformers import AutoImageProcessor, Dinov2Model
import torch
from datasets import load_dataset
import numpy as np
import os
from PIL import Image
from skimage.registration import optical_flow_ilk
import skimage
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import PIL.ImageColor as ImageColor
from transformers import Dinov2Backbone


thickness = 10
det_str = ''

color1 = ImageColor.getrgb('green')
color_tup = (color1[0], color1[1], color1[2])


dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
backbone=False
if backbone:
  model = Dinov2Backbone.from_pretrained("facebook/dinov2-base", out_indices=[0,1,2,3])
else:
  model = Dinov2Model.from_pretrained("facebook/dinov2-base")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

if backbone:
  last_hidden_states = outputs
  features_semantic = last_hidden_states.mean(dim=1)

else:
  features_semantic = outputs.pooler_output
  

result_dir = '/content/drive/MyDrive/Colab Notebooks/mask_grounding'
cropped_dir = '/content/drive/MyDrive/Colab Notebooks/mask_grounding/data/cropped_tiff'
original_dir = '/content/drive/MyDrive/Colab Notebooks/mask_grounding/data/original_tiff'



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



filenames_crop = [os.path.join(cropped_dir, x) for x in os.listdir(cropped_dir)
                  if x.endswith('tif')]
filenames_orig = [os.path.join(original_dir, x) for x in os.listdir(original_dir)
                  if x.endswith('tif')]

for file in filenames_crop:
  qual_crop = file.split('proc-Scene-')[-1].split('Create Image Subset')[0].lower()
  orig_file = [x for x in filenames_orig if qual_crop[:-1] in x.lower()][0]
  print(os.path.basename(orig_file))
  print(os.path.basename(file))

  im_orig = Image.open(orig_file)
  im_orig = im_orig.convert('RGB')
  im_crop = Image.open(file)
  im_crop = im_crop.convert('RGB')
  print(im_crop.size, im_orig.size)
  pad_crop = Image.new(im_crop.mode, (im_orig.size[0], im_orig.size[1]), (0,0,0))
  pad_crop.paste(im_crop, (im_orig.size[0]-im_crop.size[0], im_orig.size[1]-im_crop.size[1]))
  print('pasted at', (im_orig.size[0]-im_crop.size[0], im_orig.size[1]-im_crop.size[1]))

  if mask_en:
    mask = np.zeros_like(pad_crop)
    mask[im_orig.size[1]-im_crop.size[1]:, im_orig.size[0]-im_crop.size[0]:] = 1
    flow = skimage.registration.phase_cross_correlation(reference_image=np.array(im_orig), moving_image=np.array(pad_crop), reference_mask=mask)
  else:
  #plt.imshow(mask)
    flow = skimage.registration.phase_cross_correlation(np.array(im_orig), np.array(pad_crop))
  if isinstance(flow, tuple):
    flow = flow[0]
  if mask_en:
    top_left_offset = top_left_offset%im_orig.size

  top_left_offset = np.array((im_orig.size[0]-im_crop.size[0], im_orig.size[1]-im_crop.size[1]))+(flow[0][:2])
  draw_bounding_box_on_image(im_orig, top_left_offset[1] ,
                            top_left_offset[0],
                            top_left_offset[1] +im_crop.size[1] ,  top_left_offset[0] +im_crop.size[0],
                              color=color_tup,
                              thickness=thickness,
                              display_str_list=det_str,
                            use_normalized_coordinates=False)
  im_orig.save(os.path.join(result_dir,qual_crop + '_Act_cls_tiles_over_blind.png'))

  print(flow)
  break
  print('top_left_offset', top_left_offset)
