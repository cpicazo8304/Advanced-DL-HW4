from pathlib import Path

import fire
import json
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_kart_objects, extract_frame_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, split: str = None) -> list:
    """
    Generate caption for a specific view.
    """
    with open(info_path) as f:
      info = json.load(f)

    # Find corresponding image file
    info_path = Path(info_path)
    base_name = info_path.stem.replace("_info", "")
    if split:
      image_file = f"{split}/{base_name}_{view_index:02d}_im.jpg"
    else:
      image_file = str(list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0])
    image_file = str(image_file)

    captions = []
    # 1. Ego car
    # {kart_name} is the ego car.
    captions.append(
      {
        "caption": f"{info['karts'][view_index]} is the ego car.",
        "image_file": image_file
      }
    )

    # 2. Counting
    # There are {num_karts} karts in the scenario.
    captions.append(
      {
        "caption": f"There are {str(len(info['karts']))} karts in the scene.",
        "image_file": image_file
      }
    )

    # 3. Track name
    # The track is {track_name}.
    captions.append(
      {
        "caption": f"The track is {info['track']}.",
        "image_file": image_file
      }
    )

    # 4. Relative position
    # {kart_name} is {position} of the ego car.
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    ego_cx = img_width / 2
    ego_cy = img_height / 2
    for kart in karts:
      kart_cx, kart_cy = kart['center']
      kart_name = kart['kart_name']

      # check for left or right of ego car
      if kart_cx < ego_cx:
        answer1 = 'left'
      else:
        answer1 = 'right'

      # check for front or back of ego car
      if kart_cy < ego_cy:
          answer2 = 'front'
      else:
          answer2 = 'back'

      captions.append(
        {
          "caption": f"{kart_name} is {answer1} of the ego car.",
          "image_file": image_file
        }
      )

      captions.append(
        {
          "caption": f"{kart_name} is {answer2} of the ego car.",
          "image_file": image_file
        }
      )

    return captions

def generate_all_captions(data_dir: str):
  # get all info_json
  split = Path(data_dir).name
  data_path = Path(data_dir)
  info_files = list(data_path.glob("*_info.json"))
  all_capt_pairs = []
  for info_path in info_files:
    with open(info_path) as f:
      info = json.load(f)

    # go through each possible view_index and generate questions
    for view_index in range(len(info['karts'])):
      captions = generate_caption(str(info_path), view_index, split=split)
      all_capt_pairs.extend(captions)

  output_file = f"{data_dir}/all_captions.json"
  with open(output_file, "w") as f:
      json.dump(all_capt_pairs, f)


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption, "generate": generate_all_captions})


if __name__ == "__main__":
    main()
