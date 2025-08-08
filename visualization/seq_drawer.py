import sys
import json

import cairosvg
import asyncio
import json
import xml.etree.ElementTree as ET
from playwright.async_api import async_playwright
import re


class SequenceDiagram:
    def __init__(self, basicobject='self'):
        self.unk_count = 0
        self.THIS_NAME = basicobject

    def san(self, s):
        escape = ["\\", "*", "/", "+", "-", '"']
        for c in escape:
            s = s.replace(c, f"\\{c}")
        return s

    def draw_sequence_diagram(self, seq_data):
        out = ""
        out += f"title {seq_data['title']}\n"
        out += f'participant "**{self.THIS_NAME}**" as {self.THIS_NAME}\n'
        out += self.draw_elements(seq_data["sequence"])

        return out

    def draw_elements(self, elements):
        out = ""

        def append(s):
            nonlocal out
            if out:
                out += "\n"
            out += s

        for elt in elements:
            try:
                elt_type = elt["type"]

                if elt_type == "methodInvocation":
                    to = elt["to"]
                    if to == "### unk":
                        to = f"_{self.unk_count}"
                        self.unk_count += 1
                        append(f"participant \"//[...]//\" as {to}")
                    elif len(to) > 0:
                        to = self.san('.'.join(to).replace(',', ' -'))
                    else:
                        to = self.THIS_NAME
                    append(
                        f"{self.THIS_NAME}->{to}:"
                        f"{self.san(elt['method'])}"
                    )

                elif elt_type == "newInstance":
                    if self.is_used(elt["name"], elements):
                        append(f"{self.THIS_NAME}->>*{self.san(elt['name'])}://create//")

                elif elt_type == "scopedVariable":
                    if self.is_used(elt["name"], elements):
                        append(f"{self.THIS_NAME}->>*{self.san(elt['name'])}://create//")

                elif elt_type == "controlFlow":
                    append(
                        f"aboxleft over {self.THIS_NAME}:**{self.san(elt['name'])}**"
                        + (f" {self.san(elt['value'])}" if elt["value"] else "")
                    )

                elif elt_type == "blocks":
                    blocks = elt["blocks"]
                    name = elt["name"]

                    for idx, block in enumerate(blocks):
                        append(
                            f"{f'alt _{name}_' if idx == 0 else 'else '}"
                            + (f"{block['guard']}" if block["guard"] else "")
                        )
                        append(self.draw_elements(block["contents"]))

                    append("end")

                else:
                    raise Exception("unknown type", elt_type)

            except Exception as e:
                print("draw exception:", e)
                continue

        return out

    def is_used(self, name, elements):
        return True
        # for elt in elements:
        #     elt_type = elt["type"]
        #     if elt_type == "methodInvocation":
        #         if ".".join(elt["to"]) == name:
        #             return True

        #     elif elt_type == "blocks":
        #         for block in elt["blocks"]:
        #             if self.is_used(name, block["contents"]):
        #                 return True

        # return False

    async def render_sequence_diagram(self, seq_text):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto("https://sequencediagram.org/")

            await page.click(".CodeMirror-line")
            await page.evaluate(f"SEQ.main.getSourceValue = () => {json.dumps(seq_text)}")
            await page.keyboard.press("a")
            # await asyncio.sleep(0.3)
            svg = await page.evaluate("SEQ.saveAndOpen.generateSvgData()")

            await browser.close()

        elt = ET.fromstring(svg)
        self.replace_alt(elt)
        svg = ET.tostring(elt, encoding="utf-8")

        return svg


    def replace_alt(self, elt):
        prev_alt = None
        for child in elt:
            if prev_alt is not None:
                if child.tag == "{http://www.w3.org/2000/svg}text":
                    m = re.search(r"\[_([^_]*)_(.*)\]", child.text)
                    child.text = f"[{m.group(2)}]"
                    prev_alt.text = m.group(1)
                else:
                    continue

            if child.tag == "{http://www.w3.org/2000/svg}text" and child.text == "alt":
                prev_alt = child
            else:
                prev_alt = None

            self.replace_alt(child)
    
    async def draw_svg(self, seq):
        """
        pass string of sequence
        """
        seq_text = json.loads(seq)
        seq_text = self.draw_sequence_diagram(seq_text)
        print(seq_text)
        return await self.render_sequence_diagram(seq_text)


async def main(seq, path):
    seq_generator = SequenceDiagram()
    svg = await seq_generator.draw_svg(seq)    
    cairosvg.svg2png(bytestring=svg, write_to=path)
    print(f"Image saved as: {image_path}")

if __name__ == "__main__":
    # Accept command-line input for the sequence
    if len(sys.argv) < 2:
        print("Usage: python main.py <sequence>")
    else:
        if len(sys.argv) == 3:
            image_path = sys.argv[2]
        else:
            image_path = "output.png"
        seq = sys.argv[1]
        asyncio.run(main(seq, image_path))