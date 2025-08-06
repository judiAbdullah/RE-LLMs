import json
from transformers import (
    RobertaTokenizer,
    T5ForConditionalGeneration,
)
import torch
from seq_drawer import *
from dotenv import dotenv_values
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import cairosvg
import io
import traceback
from datasets import load_from_disk
import asyncio

parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(parent_dir)

config = dotenv_values("../.env")
image_size=(1000, 1000)



def fix_json(seq):
    def matchBracket(current, stack):
        if ((current == ']' and stack == '[') or 
            (current == '}' and stack == '{')):
            return True
        return False
    def getClose(stack):
        if stack == '[':
            return ']'
        elif stack == '{':
            return '}'
    stack = []
    for i, c in enumerate(list(seq)):
        if c == "[":
            stack.append("[")
        elif c == "{":
            stack.append("{") 
        elif c == "]" or c == "}":
            if len(stack) <= 0:
                raise 'error in open bracket'
            if matchBracket(c, stack[-1]):
                stack.pop()
    while len(stack) > 0:
        seq += getClose(stack[-1])
        stack.pop()                    
    return seq




def code_to_image(code, font_path='dejavu-sans/DejaVuSans.ttf', font_size=30):
    # Create a blank image with white background
    width, hight = image_size[0], image_size[1]
    image = Image.new('RGB', (width, hight), 'white')
    draw = ImageDraw.Draw(image)

    # Load a font
    font = ImageFont.load_default()
    if font_path:
        font = ImageFont.truetype(font_path, font_size)

    # Define text position and color
    text_color = 'black'
    padding = 10  # padding from the edges
    current_h = padding  # starting height position

    # Draw the code text line by line
    for line in code.split('\n'):
        draw.text((padding, current_h), line, font=font, fill=text_color)
        current_h += font_size  # move to the next line

    return image

def combine_image_sets(image_sets, output_path='testplot.jpg', padding=10):
    # Determine the dimensions of each individual image (assuming all images have the same size)
    
    image_width, image_height = image_size
    
    # Calculate the size of the final combined image
    num_rows = len(image_sets)
    num_columns = len(image_sets[0])
    
    final_width = num_columns * image_width + (num_columns + 1) * padding
    final_height = num_rows * image_height + (num_rows + 1) * padding
    
    # Create a new blank image with white background
    combined_image = Image.new('RGB', (final_width, final_height), 'white')
    
    # Paste each image into the combined image
    for row_index, image_set in enumerate(image_sets):
        for col_index, img in enumerate(image_set):
            x = padding + col_index * (image_width + padding)
            y = padding + row_index * (image_height + padding)
            combined_image.paste(img, (x, y))
    
    # Save the combined image
    combined_image.save(output_path)


async def main():
    drawSvg = SequenceDiagram()
    # filteredDatapath=os.path.join(parent_dir, config['evalDatasetpathPython'].lstrip(os.sep))
    df = pd.read_csv(os.path.join(parent_dir, config['evalstorepython'].lstrip(os.sep)))
    condition = df['Evaluation'] != 'error in cmpute_code_bleu'
    sample_df = df[condition]
    # dataset = load_from_disk(filteredDatapath)
    j = 0
    seq_png = []
    for i in range(100, 200):
        try:
            # code1 = dataset[i]['code']
            code1 = sample_df.iloc[i]['Code']
            # seq_text = model_generate(code1)
            svg1 = await drawSvg.draw_svg(json.loads(sample_df.iloc[i]['Seq']))
            
            # with open(f"test_output{i}python.svg", "wb") as f:
            #     f.write(svg1)
            png_bytes1 = cairosvg.svg2png(bytestring=svg1)
            img_py = Image.open(io.BytesIO(png_bytes1)).resize(image_size, Image.Resampling.LANCZOS)
            # img_py.save(f"test_output{i}python.png")


            
            code2 = sample_df.iloc[i]['Code']
            # seq_text = model_generate(code2)
            svg2 = await drawSvg.draw_svg(json.loads(sample_df.iloc[i]['Generated Seq']))
            
            # with open(f"test_output{i}java.svg", "wb") as f:
            #     f.write(svg2)
            png_bytes2 = cairosvg.svg2png(bytestring=svg2)
            img_ja = Image.open(io.BytesIO(png_bytes2)).resize(image_size, Image.Resampling.LANCZOS)
            # img_ja.save(f"test_output{i}java.png")

            seq_png.append([img_py, code_to_image(code=code1), img_ja, code_to_image(code=code2)])
            j += 1
            if j >= 10:
                break
        except Exception as e:
            print(f"An error occurred: {e} current: {i}, processed: {j}")
            traceback.print_exc()
    print(seq_png)
    combine_image_sets(seq_png)

if __name__ == "__main__":
    asyncio.run(main())
