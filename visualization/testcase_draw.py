import json
import math
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




def code_to_image(code, font_path='dejavu-sans/DejaVuSans.ttf', font_size=30, image_sizef=(1000, 1000)):
    width, hight = image_sizef[0], image_sizef[1]
    image = Image.new('RGB', (width, hight), 'white')
    draw = ImageDraw.Draw(image)

    font = ImageFont.load_default()
    if font_path:
        font = ImageFont.truetype(font_path, font_size)

    # Define text position and color
    text_color = 'black'
    padding = 10  # padding from the edges
    current_h = padding  # starting height position

    for line in code.split('\n'):
        draw.text((padding, current_h), line, font=font, fill=text_color)
        current_h += font_size  # move to the next line

    return image

def combine_image_sets(image_sets, size, path, padding=10):   
    image_width, image_height = size[0], size[1]

    final_width = 2 * image_width + 3 * padding
    final_height = image_height + 2 * padding
    
    combined_image = Image.new('RGB', (final_width, final_height), 'white')
    
    x = padding
    y = padding
    combined_image.paste(image_sets[0].resize((image_width, image_height)), (x, y))

    x = 2 * padding + image_width
    y = padding
    combined_image.paste(image_sets[1].resize((image_width, image_height)), (x, y))
    
    combined_image.save(path)


async def main():
    drawSvg = SequenceDiagram(basicobject='self')
    df = pd.read_csv(os.path.join(parent_dir, 'dataset/testcase.csv'))
    df['code'].str.replace('""', '\\"')
    df['seqs'].str.replace('""', '\\"')
    df['gen_dec'].str.replace('""', '\\"')
    for index, row in df.iterrows():
        solved = False 
        highmult = math.ceil(len(row['seqs']) / 300) if math.ceil(len(row['seqs']) / 300) > 0 else 1
        try:
            seq = json.loads(row['seqs'])
            seqgen = json.loads(row['gen_dec'])
        except:
            try:
                seqgen = json.loads(fix_json(row['gen_dec']))
                solved = True
            except:
                code = code_to_image(code=df.iloc[index]['code'])

                column2 = code_to_image("sample index: "+str(index) 
                                        + "\nError in generated sequence JSON form\n" 
                                        + str(row['evaluation'])
                                        + "\nsequence json length: " + str(len(row['seqs'])))
                seq_json_image = code_to_image(code=json.dumps(seq, indent=4), image_sizef=(1000, 1000*highmult))

                formatted_string = row['gen_dec'].replace('",', '",\n')  # Add newlines after each comma
                formatted_string = formatted_string.replace('{"', '\n{')  # Start new line for each new object
                formatted_string = formatted_string.replace('}', '\n}')  # Add newline before closing brace
                seqgen_json_image = code_to_image(code=formatted_string, image_sizef=(1000, 1000*highmult))

                seq_png = [code, column2]
                combine_image_sets(seq_png, size=image_size, path=f'index_{index}_1.jpg')
                seq_png = [seq_json_image, seqgen_json_image]
                combine_image_sets(seq_png, size=(1000, 1000*highmult), path=f'index_{index}_1.jpg')
                continue
        try:
            code = code_to_image(code=df.iloc[index]['code'])

            column2 = code_to_image("sample index: "+str(index) 
                                    + "\nsolved Error\n" if solved else ""
                                    + str(row['evaluation'])
                                    + "\nsequence json length: " + str(len(row['seqs'])))
            
            seq_json_image = code_to_image(code=json.dumps(seq, indent=4), image_sizef=(1000, 1000*highmult))

            seqgen_json_image = code_to_image(code=json.dumps(seqgen, indent=4), image_sizef=(1000, 1000*highmult))

            svg1 = await drawSvg.draw_svg(json.dumps(seq, indent=4))
            png_bytes1 = cairosvg.svg2png(bytestring=svg1)
            seq_image = Image.open(io.BytesIO(png_bytes1)).resize(image_size, Image.Resampling.LANCZOS)

            svg2 = await drawSvg.draw_svg(json.dumps(seqgen, indent=4))
            png_bytes2 = cairosvg.svg2png(bytestring=svg2)
            genseq_image = Image.open(io.BytesIO(png_bytes2)).resize(image_size, Image.Resampling.LANCZOS)

            seq_png = [code, column2]
            combine_image_sets(seq_png, size=image_size, path=f'index_{index}_1.jpg')
            seq_png = [seq_image, genseq_image]
            combine_image_sets(seq_png, size=image_size, path=f'index_{index}_2.jpg')
            seq_png = [seq_json_image, seqgen_json_image]
            combine_image_sets(seq_png, size=(1000, 1000*highmult), path=f'index_{index}_3.jpg')
            
        except Exception as e:
            traceback.print_exc()
    


if __name__ == "__main__":
    asyncio.run(main())
