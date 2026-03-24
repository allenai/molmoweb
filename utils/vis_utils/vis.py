import os
import re
import json
import argparse
import textwrap
from PIL import Image, ImageDraw, ImageFont


def draw_coord_box(image, action_str, box_size=40, outline_color="red", width=3):
    """
    If action_str contains coordinates in the format:
      mouse_click(x=371, y=252, button=left)
    parse those coordinates and draw a red rectangle on the image.
    """
    if isinstance(action_str, list):
        action_str = " ".join(str(item) for item in action_str)

    m = re.search(r"x\s*=\s*(\d+).*y\s*=\s*(\d+)", action_str)
    if m:
        x = int(m.group(1))
        y = int(m.group(2))
        # Create a drawing context.
        draw = ImageDraw.Draw(image)
        # Draw a rectangle starting at (x, y) with the given box_size.
        draw.rectangle([(x, y), (x + box_size, y + box_size)], outline=outline_color, width=width)
    return image


def annotate_image_right(image, text, font, text_width=300, padding=10,
                           bg_color=(0, 0, 0, 180), text_color=(255, 255, 255)):
    """
    Create a new image that places the original image on the left and writes text to the right.
    """
    # Ensure the input image is in RGBA mode.
    image = image.convert("RGBA")
    img_width, img_height = image.size
    new_width = img_width + text_width
    # Create a new image to hold the screenshot and the text area.
    annotated = Image.new("RGBA", (new_width, img_height), (0, 0, 0, 0))
    
    # Paste the original image on the left side.
    annotated.paste(image, (0, 0))
    
    # Create a drawing context.
    draw = ImageDraw.Draw(annotated)
    
    # Draw a background rectangle for the text area.
    draw.rectangle([(img_width, 0), (new_width, img_height)], fill=bg_color)
    
    # Wrap the text so that it fits within the text area.
    # Use getbbox to calculate average character width.
    bbox_x = font.getbbox("x")
    avg_char_width = bbox_x[2] - bbox_x[0]
    max_chars_per_line = max(1, (text_width - 2 * padding) // avg_char_width)
    wrapped_text = "\n".join(textwrap.fill(line, width=max_chars_per_line) for line in text.splitlines())
    
    # Use multiline_textbbox to get the bounding box of the wrapped text.
    bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
    text_width_calc = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the text vertically in the text area.
    text_x = img_width + padding
    text_y = (img_height - text_height) // 2
    draw.multiline_text((text_x, text_y), wrapped_text, fill=text_color, font=font)
    
    return annotated


def create_annotated_gif(directory, output_file, duration=1000, font_path=None, font_size=30, text_width=600, coord_flag=False):
    """
    Creates an animated GIF from screenshots and a trajectory.json file.
    The trajectory.json file should be located in the given directory and the images in a subfolder called "images".
    The action and thought from the trajectory will be drawn to the right of each screenshot.
    """
    # Define file paths.
    trajectory_file = os.path.join(directory, "trajectory.json")
    images_dir = os.path.join(directory, "images")
    
    # Load trajectory.json
    with open(trajectory_file, "r", encoding="utf-8") as f:
        trajectory = json.load(f)
    
    # Sort keys numerically.
    sorted_keys = sorted(trajectory.keys(), key=lambda k: int(k))
    
    # Load the font.
    try:
        if font_path is not None:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), "Arial.ttf"), font_size)
            #font = ImageFont.load_default()
    except Exception as e:
        print("Error loading font; using default font. Error:", e)
        font = ImageFont.load_default()
    
    annotated_images = []
    for key in sorted_keys:
        entry = trajectory[key]
        screenshot_filename = entry.get("screenshot")
        if not screenshot_filename:
            print(f"No screenshot for key {key}")
            continue
        
        image_path = os.path.join(images_dir, screenshot_filename)
        if not os.path.exists(image_path):
            print(f"File {image_path} not found; skipping.")
            continue
        
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
        
        # Retrieve action and thought details.
        action_str = entry.get("action", {}).get("action_str", "No action")
        thought = entry.get("action", {}).get("action_output", {}).get("thought", "No thought")
        combined_text = f"Action: {action_str}\n\nThought: {thought}"
        
        # If using --coord flag, draw a red box on the image if it contains x,y coordinates.
        if coord_flag:
            image = draw_coord_box(image, action_str)

        # Annotate the image with text on the right.
        annotated_img = annotate_image_right(image, combined_text, font, text_width=text_width)
        annotated_images.append(annotated_img.convert("RGB"))
    
    if annotated_images:
        annotated_images[0].save(
            output_file,
            save_all=True,
            append_images=annotated_images[1:],
            duration=duration,
            loop=0
        )
        print(f"Annotated GIF created and saved as {output_file}")
    else:
        print("No annotated images were created. Please check your inputs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create an annotated GIF from screenshots with actions and thoughts to the right."
    )
    parser.add_argument("directory", type=str, help="Directory containing trajectory.json and images/ folder")
    parser.add_argument("output_file", type=str, help="Filename for the output GIF (e.g., annotated.gif)")
    parser.add_argument("--duration", type=int, default=1000, help="Frame duration in milliseconds (default: 1000)")
    parser.add_argument("--font_path", type=str, default=None, help="Optional path to a TTF font file")
    parser.add_argument("--font_size", type=int, default=30, help="Font size for annotation text (default: 20)")
    parser.add_argument("--text_width", type=int, default=600, help="Width of the text area in pixels (default: 300)")
    parser.add_argument("--coord", action="store_true", help="If set, draw red boxes on images for actions with x,y coordinates.")

    args = parser.parse_args()
    create_annotated_gif(args.directory, args.output_file, args.duration, args.font_path, args.font_size, args.text_width, args.coord)

