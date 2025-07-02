import xml.etree.ElementTree as ET
from lxml import etree
import ast
from xml.dom import minidom

NAMESPACE = {'xsi': 'http://www.w3.org/2001/XMLSchema-instance'}
ET.register_namespace('xsi', NAMESPACE['xsi'])
xsi_attrib = f"{{{NAMESPACE['xsi']}}}type"


def xml_to_string_prettify(root):
    """
    Return a pretty-printed XML string for the Element.
    """
    return etree.tostring(root, pretty_print=True, xml_declaration=True, encoding='UTF-8').decode('UTF-8')

def xml_to_string(root):
    """
    Return a XML string for the Element.
    """
    return etree.tostring(root, encoding='UTF-8').decode('UTF-8')

def print_xmi(root):
    """
    To Print list of xmi elements
    """
    for ele in root.iter():
        print('element tag: ', ele.tag, ' | element attribute: ', ele.attrib)

def parse_xmi_file(xmi_file_path):
    """
    Parses the XMI file and returns the root element of the XML tree by lxml
    """
    tree = etree.parse(xmi_file_path)
    return tree.getroot()

def parse_xmi_string(xmi_string):
    """
    Convert a pretty-printed XML string back to an ElementTree object and return the root.
    """
    tree = etree.fromstring(xmi_string.encode('UTF-8'))
    return tree

def save_xmi_to_file(xmi_content, file_path):
    """Save XMI content to a file with pretty printing."""
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(xmi_content)
    print(f"XMI content successfully saved to {file_path}")



# search in xml to find tags

   

def find_element_by_tag(root, tag_name):
    """
    Find first children element with specific tag name.
    """
    elements = root.findall(tag_name)
    if len(elements) == 0:
        return None 
    return elements[0]
        

def find_elements_by_tag(root, tag_name):
    """
    Finds all children elements with a specific tag name.
    """
    return root.findall(tag_name)
    

def find_elements_by_tag_orNUll(root, tag_name):
    """
    Finds all children elements with a specific tag name.
    """
    f = root.findall(tag_name)
    if len(f) == 0:
        return None
    return f


def find_element_by_attribute(root, tag_name, attribute_name, attribute_value):
    """
    Finds first element with a specific tag name and attribute value.
    """
    elements = root.findall(f".//{tag_name}[@{attribute_name}='{attribute_value}']", NAMESPACE)
    if len(elements) != 0:
        return elements[0]
    else:
        return None 

def find_elements_by_attribute(root, tag_name, attribute_name, attribute_value):
    """
    Finds elements with a specific tag name and attribute value.
    """
    elements = root.findall(f".//{tag_name}[@{attribute_name}='{attribute_value}']", NAMESPACE)
    if len(elements) != 0:
        return elements
    else:
        return None 



def split_string(input_string):
    result = []
    current = []
    stack = 0

    for char in input_string:
        if char == '.' and stack == 0:
            # End of the current segment
            result.append(''.join(current))
            current = []
        else:
            if char == '(':
                stack += 1
            elif char == ')':
                stack -= 1
            current.append(char)

    # Add the last segment
    if current:
        result.append(''.join(current))

    return result