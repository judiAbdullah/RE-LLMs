import xml.etree.ElementTree as ET
import traceback
import re

from ..process_xmi import *
from ..clean_seq import *


# xsi namespace
NAMESPACE = {'xsi': 'http://www.w3.org/2001/XMLSchema-instance'}
ET.register_namespace('xsi', NAMESPACE['xsi'])
xsi_attrib = f"{{{NAMESPACE['xsi']}}}type"


class pythonSequenceGenerator:
    def __init__(self, cleanseq=False):
        self.cleanseq=cleanseq

    def code_to_seq(self, code):
        """
        if you want code -> seq call this function
        if you want code -> seq with all steps follow functions call bellow
        """
        try:
            tree = self.code_to_ast(code)
            root = self.ast_to_xmi(tree)
            xmlstr = xml_to_string_prettify(root)
            seq = self.generate_sequence(xmlstr, code)
            return seq
        except:
            # import traceback
            # traceback.print_exc()
            return {}

    def code_to_ast(self, code):
        """
        Parse python code to ast
        """
        try:
            tree = ast.parse(code)
            return tree
        except:
            raise ValueError("Syntax Error Occured")

    def ast_to_xmi(self, node, field_name=None):
        """
        Recursively convert AST node to XML element with xsi:type
        using lxml library for more functionality
        """
        if isinstance(node, ast.Constant):  # Python 3.8+
            element = etree.Element(field_name or 'Constant', nsmap=NAMESPACE)
            element.set(f'{{{NAMESPACE["xsi"]}}}type', 'Constant')
            # Include type and value of the constant
            element.set('type', type(node.value).__name__)
            element.set('value', str(node.value))
            return element
        elif isinstance(node, ast.Str):  # Python 3.7 and earlier
            element = etree.Element(field_name or 'Str', nsmap=NAMESPACE)
            element.set(f'{{{NAMESPACE["xsi"]}}}type', 'Str')
            element.set('type', 'str')
            element.set('value', node.s)
            return element
        elif isinstance(node, ast.Num):  # Python 3.7 and earlier
            element = etree.Element(field_name or 'Num', nsmap=NAMESPACE)
            element.set(f'{{{NAMESPACE["xsi"]}}}type', 'Num')
            element.set('type', 'int')  # Assume integer; you may need additional checks for other types like float
            element.set('value', str(node.n))
            return element
        else:
            element_type = type(node).__name__
            element = etree.Element(field_name or element_type, nsmap=NAMESPACE)
            element.set(f'{{{NAMESPACE["xsi"]}}}type', element_type)
            
            for field, value in ast.iter_fields(node):
                if isinstance(value, ast.AST):
                    # Recursively process AST nodes
                    child = self.ast_to_xmi(value, field)
                    element.append(child)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            # Recursively process list items
                            child = self.ast_to_xmi(item, field)
                            element.append(child)
                else:
                    # Handle other values as attributes
                    element.set(field, str(value))
            
            return element
    
    def is_only_subfunction_call(self, code):
        tree = self.code_to_ast(code)
        root = self.ast_to_xmi(tree)
        method_declaration = self.extract_python_method(root)
        method_childrens = [ch for ch in method_declaration.getchildren() if ch.tag == "body"]
        if len(method_childrens) == 2 and method_childrens[0].attrib.get(xsi_attrib) == 'FunctionDef' and method_childrens[1].attrib.get(xsi_attrib) == 'Return':
            return True
        else:
            return False

    def extract_function_def(self, code: str) -> list:
        """
        Extracts function names and their parameter parts from a given Python code as text.

        Args:
            code (str): The Python code as a string.

        Returns:
            list: A list of strings containing the function signature (name and parameters).
        """
        # Match multiline function definitions
        pattern = r'def\s+(\w+)\s*\(([^)]*)\)\s*:'
        matches = re.findall(pattern, code, re.DOTALL)
        signatures = []
        
        for name, params in matches:
            # Clean up parameters to remove excessive whitespace, newlines, and comments
            cleaned_params = re.sub(r'#.*', '', params)  # Remove inline comments
            cleaned_params = re.sub(r'\s+', ' ', cleaned_params).strip()  # Remove newlines and extra spaces
            signatures.append(f"{name}({cleaned_params})")
        
        return signatures

    def replace_escaped_quotes_in_json(self, obj):
        """
        Recursively replace escaped double quotes (\") with single quotes (') in a JSON object.

        Args:
            obj: The JSON object (dict, list, or string).

        Returns:
            The modified JSON object with replaced quotes.
        """
        if isinstance(obj, dict):
            # If it's a dictionary, process each key-value pair
            return {key: self.replace_escaped_quotes_in_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            # If it's a list, process each element
            return [self.replace_escaped_quotes_in_json(item) for item in obj]
        elif isinstance(obj, str):
            # If it's a string, replace \" with '
            return obj.replace('"', "'")
        else:
            # If it's neither, return it as-is (e.g., numbers, booleans, None)
            return obj
    
    def generate_sequence(self, xmi_string, code=None):
        
        """
        Generate sequence in json form by analyzing the function xmi meta model
        """

        try:
            root = parse_xmi_string(xmi_string)
            method_declaration = self.extract_python_method(root)
            # extract parameter
            sequence = []
            parameters_seq, parameters = self.parse_parameters(method_declaration)
            sequence.extend(parameters_seq)
            # extract sequence from function body
            methos_body_sequence = self.parse_method_body(method_declaration)
            sequence.extend(methos_body_sequence)
            formatted_parameters = ', '.join([f"{key}={value}" if value is not None else key for key, value in parameters.items()])
            formatted_parameters = ', '.join([f"{key}" for key, value in parameters.items()])
            funcdef = ''
            if code is not None:
                funcdef = f'def {self.extract_function_def(code)[0]}'
            else:
                funcdef = f"{method_declaration.attrib.get('name')}({formatted_parameters})"
            seqcleaner = SeqCleaner()
            seq = {
                    "title": re.sub(r'\s*=\s*', '=', funcdef),
                    "sequence": seqcleaner.clean_element(sequence) if self.cleanseq else sequence,
                }
            return self.replace_escaped_quotes_in_json(seq)
        except:
            raise ValueError("Error in Sequence Generation Occured")



    def extract_python_method(self, root):
        """
        Extracts method from the XMI file.
        """
        method_declaration = find_elements_by_attribute(root, 'body', '{'+NAMESPACE['xsi']+'}type', 'FunctionDef')
        if method_declaration is not None:
            return method_declaration[0]
        else:
            method_declaration = find_elements_by_attribute(root, 'body', '{'+NAMESPACE['xsi']+'}type', 'AsyncFunctionDef')
            if method_declaration is not None:
                return method_declaration[0]
            else:
                return None 

    def parse_method_body(self, method_declaration):
        """
        Generate the method sequence return in json form
        """
        # get all body stmts
        method_childrens = [ch for ch in method_declaration.getchildren() if ch.tag == "body"]
        seq = []
        for ch in method_childrens:
            seq.extend(self.parse_stmt(ch))
        return seq










    
    def parse_parameters(self, method_declaration):
        """
        Extract the function parameters return as dictionary 
        {arg1:default_value1, .....} if no default_value them None
        """
        method_childrens = method_declaration.getchildren()
        args = None
        for ch in method_childrens:
            if ch.tag == "args":
                args = ch
                break
        if args is None:
            raise Exception("Function declaration parameters not found")
        args_childrens = args.getchildren()
        args_name = [('*' if ar.tag=='vararg' else ('**' if ar.tag=='kwarg' else ''))+ar.attrib.get('arg') for ar in args_childrens if "arg" in ar.tag]
        args_defaults = [(ar.attrib.get('value') if ar.attrib.get('value') else ar.attrib.get('id')) for ar in args_childrens if "defaults" in ar.tag ]
        seq = []
        # for ar in args_childrens:
        #     if "arg" in ar.tag or "kwarg" in ar.tag:
        #         seq.append({"type":"scopedVariable",
        #                     "name":ar.attrib.get('arg')})
        start_def = len(args_name) - len(args_defaults)
        args_return = {key: None for key in args_name[:start_def]}
        args_return.update(zip(args_name[start_def:], args_defaults))

        return seq, args_return



    def parse_stmt(self, stmt):
        """
        Generate the statements sequence return in json form
        """
        stmt_handlers = {
            "FunctionDef": self.process_functiondef,
            "AsyncFunctionDef": self.process_asyncfunctiondef,
            "ClassDef": self.process_classdef,
            "Return": self.process_return,
            "Delete": self.process_delete,
            "Assign": self.process_assign,
            "TypeAlias": self.process_typealias,
            "AugAssign": self.process_augassign,
            "AnnAssign": self.process_annassign,
            "For": self.process_for,
            "AsyncFor": self.process_asyncfor,
            "While": self.process_while,
            "If": self.process_if,
            "With": self.process_with,
            "AsyncWith": self.process_ayncwith,
            "Match": self.process_match,
            "Raise": self.process_raise,
            "Try": self.process_try,
            "TryStar": self.process_trystar,
            "Assert": self.process_assert,
            "Import": self.process_import,
            "ImportFrom": self.process_importfrom,
            "Global": self.process_global,
            "Nonlocal": self.process_nonlocal,
            "Expr": self.process_expr,
            "Pass": self.process_pass,
            "Break": self.process_break,
            "Continue": self.process_continue,
        }
        
        
        
        # re add stmts
        ignore_stmt = ["FunctionDef", 
                    "AsyncFunctionDef", 
                    "ClassDef",
                    "Import",
                    "ImportFrom",
                    "Global",
                    "Nonlocal",
                    ]
        if stmt.attrib.get(xsi_attrib) not in ignore_stmt:
            return stmt_handlers[stmt.attrib.get(xsi_attrib)](stmt)
        else:
            return []

    def process_functiondef(self, stmt):
        pass

    def process_asyncfunctiondef(self, stmt):
        pass

    def process_classdef(self, stmt):
        pass

    def process_return(self, stmt):
        """
        parse return statment
        """
        return_child = find_element_by_tag(stmt, "value")
        # parse depth
        return_seq = []
        if return_child is not None:
            return_seq, return_code_expr = self.parse_expr(return_child)
            # append return call to depth as controlFlow
            return_seq.extend([{"type":"controlFlow", 
                                "name": "return", 
                                "value": return_code_expr
                            }])
            return return_seq
        else:
            return_seq.extend([{"type":"controlFlow", 
                                "name": "return", 
                                "value": None
                            }])
            return return_seq
        
    def process_delete(self, stmt):
        """
        parse delete statment
        """
        delete_child = stmt.getchildren()[0]
        delete_seq, delete_code_expr = self.parse_expr(delete_child)
        delete_code_expr = f"delete {delete_code_expr}"
        delete_seq.extend([{"type":"methodInvocation",
                            "to": [],
                            "method": delete_code_expr
                        }])
        return delete_seq

    def process_assign(self, stmt):
        """
        parse assign statment
        """
        target = find_element_by_tag(stmt, "targets")
        value = find_element_by_tag(stmt, "value")
        target_seq, target_code_expr = self.parse_expr(target)
        value_seq, value_code_expr = self.parse_expr(value)
        seq = []
        seq.extend(target_seq)
        seq.extend(value_seq)
        # if target.attrib.get(xsi_attrib) == "Tuple":
        #     for ch in target.getchildren():
        #         if ch.tag == "elts":
        #             seq.extend([{
        #                 "type":"scopedVariable",
        #                 "name":ch.attrib.get('id')
        #             }])
        # else:
        #     seq.extend([{
        #         "type":"scopedVariable",
        #         "name":target_code_expr
        #     }])

        return seq

    def process_typealias(self, stmt):
        pass

    def process_augassign(self, stmt):
        """
        parse augmented assignment statment
        """
        bin_operator_handlers = {
            "Add": "+",
            "Sub": "-",
            "Mult": "*",
            "MatMult": "@",
            "Div": "/",
            "Mod": "%",
            "Pow": "**",
            "LShift": "<<",
            "RShift": ">>",
            "BitOr": "|",
            "BitXor": "^",
            "BitAnd": "&",
            "FloorDiv": "//"
        }
        target = find_element_by_tag(stmt, "target")
        op = find_element_by_tag(stmt, "op")
        value = find_element_by_tag(stmt, "value")
        target_seq, target_code_expr = self.parse_expr(target)
        value_seq, value_code_expr = self.parse_expr(value)
        seq = []
        seq.extend(target_seq)
        seq.extend(value_seq)
        # if target.attrib.get(xsi_attrib) == "Tuple":
        #     for ch in target.getchildren():
        #         if ch.tag == "elts":
        #             seq.extend([{
        #                 "type":"scopedVariable",
        #                 "name":ch.attrib.get('id')
        #             }])
        # else:
        #     seq.extend([{
        #         "type":"scopedVariable",
        #         "name":target_code_expr
        #     }])
        return seq

    def process_annassign(self, stmt):
        """
        parse annotated assignment statment
        """
        target = find_element_by_tag(stmt, "target")
        annotation = find_element_by_tag(stmt, "annotation")
        value = find_element_by_tag(stmt, "value")
        target_seq, target_code_expr = self.parse_expr(target)
        annotation_seq, annotation_code_expr = self.parse_expr(annotation)
        seq = []
        seq.extend(annotation_seq)
        seq.extend(target_seq)
        if value is not None:
            value_seq, value_code_expr = self.parse_expr(value)
            seq.extend(value_seq)
        else:
            value_code_expr = None
        
        # if target.attrib.get(xsi_attrib) == "Tuple":
        #     for ch in target.getchildren():
        #         if ch.tag == "elts":
        #             seq.extend([{
        #                 "type":"scopedVariable",
        #                 "name":ch.attrib.get('id')
        #             }])
        # else:
        #     seq.extend([{
        #         "type":"scopedVariable",
        #         "name":target_code_expr
        #     }])
        # value could be None check it in sequence draw
        return seq

    def process_for(self, stmt):
        """
        parse for statment
        """
        seq = []
        init_seq = []
        target = find_element_by_tag(stmt, "target")
        iter = find_element_by_tag(stmt, "iter")
        body = find_elements_by_tag(stmt, "body")
        orelse = find_elements_by_tag_orNUll(stmt, "orelse")
        target_seq, target_code_expr = self.parse_expr(target)
        iter_seq, iter_code_expr = self.parse_expr(iter)
        body_seq = []
        for b in body:
            b_seq = self.parse_stmt(b)
            body_seq.extend(b_seq)

        init_seq.extend(target_seq)
        
        init_seq.extend(iter_seq)
        # if target.attrib.get(xsi_attrib) == "Tuple":
        #     for ch in target.getchildren():
        #         if ch.tag == "elts":
        #             init_seq.extend([{
        #                 "type":"scopedVariable",
        #                 "name":ch.attrib.get('id')
        #             }])
        # else:
        #     init_seq.extend([{
        #         "type":"scopedVariable",
        #         "name":target_code_expr
        #     }])
        
        seq.extend(init_seq)
        if orelse is not None:
            orelse_seq = []
            for o in orelse:
                o_seq = self.parse_stmt(o)
                orelse_seq.extend(o_seq)
            seq.extend([{
                "type": "blocks",
                "name": "loop",
                "blocks": [
                    {
                        "guard": target_code_expr + " in " +  iter_code_expr,
                        "contents": body_seq
                    },{
                        "guard": "else",
                        "contents": orelse_seq
                    }
                ]
            }])
        else:
            seq.extend([{
                "type": "blocks",
                "name": "loop",
                "blocks": [
                    {
                        "guard": target_code_expr + " in " +  iter_code_expr,
                        "contents": body_seq
                    }
                ]
            }])
        
        return seq

    def process_asyncfor(self, stmt):
        """
        parse asyncfor statment
        """
        seq = []
        init_seq = []
        target = find_element_by_tag(stmt, "target")
        iter = find_element_by_tag(stmt, "iter")
        body = find_elements_by_tag(stmt, "body")
        orelse = find_elements_by_tag_orNUll(stmt, "orelse")
        target_seq, target_code_expr = self.parse_expr(target)
        iter_seq, iter_code_expr = self.parse_expr(iter)
        body_seq = []
        for b in body:
            b_seq = self.parse_stmt(b)
            body_seq.extend(b_seq)

        init_seq.extend(target_seq)
        
        init_seq.extend(iter_seq)

        # if target.attrib.get(xsi_attrib) == "Tuple":
        #     for ch in target.getchildren():
        #         if ch.tag == "elts":
        #             init_seq.extend([{
        #                 "type":"scopedVariable",
        #                 "name":ch.attrib.get('id')
        #             }])
        # else:
        #     init_seq.extend([{
        #         "type":"scopedVariable",
        #         "name":target_code_expr
        #     }])
        
        seq.extend(init_seq)
        if orelse is not None:
            orelse_seq = []
            for o in orelse:
                o_seq = self.parse_stmt(o)
                orelse_seq.extend(o_seq)
            seq.extend([{
                "type": "blocks",
                "name": "loop",
                "blocks": [
                    {
                        "guard": target_code_expr + " in " +  iter_code_expr,
                        "contents": body_seq
                    },{
                        "guard": "else",
                        "contents": orelse_seq
                    }
                ]
            }])
        else:
            seq.extend([{
                "type": "blocks",
                "name": "loop",
                "blocks": [
                    {
                        "guard": target_code_expr + " in " +  iter_code_expr,
                        "contents": body_seq
                    }
                ]
            }])
        
        return seq

    def process_while(self, stmt):
        """
        Parse While statment
        """
        seq = []
        test = find_element_by_tag(stmt, "test")
        body = find_elements_by_tag(stmt, "body")
        orelse = find_elements_by_tag_orNUll(stmt, "orelse")
        test_seq, test_code_expr = self.parse_expr(test)
        body_seq = []
        seq.extend(test_seq)
        for b in body:
            b_seq = self.parse_stmt(b)
            body_seq.extend(b_seq)
        

        if orelse is not None:
            orelse_seq = []
            for o in orelse:
                o_seq = self.parse_stmt(o)
                orelse_seq.extend(o_seq)
            seq.extend([{
                "type": "blocks",
                "name": "loop",
                "blocks": [
                    {
                        "guard": test_code_expr,
                        "contents": body_seq
                    },{
                        "guard": "else",
                        "contents": orelse_seq
                    }
                ]
            }])
        else:
            seq.extend([{
                "type": "blocks",
                "name": "loop",
                "blocks": [
                    {
                        "guard": test_code_expr,
                        "contents": body_seq
                    }
                ]
            }])
        
        return seq

    def process_if(self, stmt):
        """
        parse If statment
        """
        seq = []
        test = find_element_by_tag(stmt, "test")
        body = find_elements_by_tag(stmt, "body")
        orelse = find_elements_by_tag_orNUll(stmt, "orelse")
        test_seq, test_code_expr = self.parse_expr(test)
        seq.extend(test_seq)
        body_seq = []
        for b in body:
            b_seq = self.parse_stmt(b)
            body_seq.extend(b_seq)
        
        if orelse is not None:
            orelse_seq = []
            for o in orelse:
                o_seq = self.parse_stmt(o)
                orelse_seq.extend(o_seq)
            seq.extend([{
                "type": "blocks",
                "name": "if",
                "blocks": [
                    {
                        "guard": test_code_expr,
                        "contents": body_seq
                    },{
                        "guard": "else",
                        "contents": orelse_seq
                    }
                ]
            }])
        else:
            seq.extend([{
                "type": "blocks",
                "name": "if",
                "blocks": [
                    {
                        "guard": test_code_expr,
                        "contents": body_seq
                    }
                ]
            }])
        
        return seq

    def process_with(self, stmt):
        """
        Parse With statment
        """
        # seq = []
        items = find_elements_by_tag(stmt, "items")
        body = find_elements_by_tag(stmt, "body")
        items_seq = []
        items_code_expr = ""
        for item in items:
            item_seq, item_code_expr = self.parse_withitem(item)
            items_seq.extend(item_seq)
            items_code_expr += item_code_expr + ", "
        if item_code_expr.endswith(", "):
            items_code_expr = items_code_expr[:-2]
        
        body_seq = []
        body_seq.extend(items_seq)
        for b in body:
            b_seq = self.parse_stmt(b)
            body_seq.extend(b_seq)
        # seq.extend([{
        #         "type": "blocks",
        #         "name": "with",
        #         "blocks": [
        #             {
        #                 "guard": items_code_expr,
        #                 "contents": body_seq
        #             }
        #         ]
        #     }])
        # return seq
        return body_seq

    def process_ayncwith(self, stmt):
        """
        Parse AsyncWith statment
        """
        seq = []
        items = find_elements_by_tag(stmt, "items")
        body = find_elements_by_tag(stmt, "body")
        items_seq = []
        items_code_expr = ""
        for item in items:
            item_seq, item_code_expr = self.parse_withitem(item)
            items_seq.extend(item_seq)
            items_code_expr += item_code_expr + ", "
        if item_code_expr.endswith(", "):
            items_code_expr = items_code_expr[:-2]
        
        body_seq = []
        body_seq.extend(items_seq)
        for b in body:
            b_seq = self.parse_stmt(b)
            body_seq.extend(b_seq)

        # seq.extend([{
        #         "type": "blocks",
        #         "name": "with",
        #         "blocks": [
        #             {
        #                 "guard": items_code_expr,
        #                 "contents": body_seq
        #             }
        #         ]
        #     }])
        # return seq
        return body_seq

    def process_match(self, stmt):
        """
        Parse Match statment
        """
        subject = find_element_by_tag(stmt, "subject")
        cases = find_elements_by_tag(stmt, "cases")
        seq = []
        subject_seq, subject_code = self.parse_expr(subject)
        seq.extend(subject_seq)
        cases_seq = []
        for c in cases:
            c_seq = self.parse_match_case(c)
            cases_seq.extend(c_seq)
        seq.extend([{
            "type":"blocks",
            "name":"if",
            "blocks": cases_seq
        }])
        return seq

    def parse_match_case(self, case):
        """
        Parse MatchCase
        """
        pattern = find_element_by_tag(case, "pattern")
        bodys = find_elements_by_tag(case, "body")
        patter_seq, pattern_code = self.parse_pattern(pattern)
        body_seq = []
        for b in bodys:
            b_seq = self.parse_stmt(b)
            body_seq.extend(b_seq)
        seq = [{
            "guard":pattern_code,
            "contents":body_seq
            }]
        return seq

    def process_try(self, stmt):
        """
        parse Try statment
        """
        seq = []
        body = find_elements_by_tag(stmt, "body")
        handlers = find_elements_by_tag(stmt, "handlers")
        orelse = find_elements_by_tag_orNUll(stmt, "orelse")
        finalbody = find_element_by_tag(stmt, "finalbody")

        body_seq = []
        for b in body:
            b_seq = self.parse_stmt(b)
            body_seq.extend(b_seq)
        body_block = [{
            "guard":"",
            "contents":body_seq
        }]
        seq.extend(body_block)
        if handlers is not None:
            handler_seq = [self.parse_excepthandler(h) for h in handlers]
            seq.extend(handler_seq)
        if orelse is not None:
            orelse_seq = []
            for o in orelse:
                o_seq = self.parse_stmt(o)
                orelse_seq.extend(o_seq)

            orelse_block = [{
                "guard":"else",
                "contents":orelse_seq
            }]
            seq.extend(orelse_block)
        if finalbody is not None:
            f_seq = self.parse_stmt(finalbody)
            finally_block = [{
                "guard":"finally",
                "contents":f_seq
            }]
            seq.extend(finally_block)
        
        try_except_block = [{
            "type": "blocks",
            "name": "try",
            "blocks":seq
        }]
        return try_except_block

    def process_trystar(self, stmt):
        """
        parse TryStare statment
        """
        seq = []
        body = find_elements_by_tag(stmt, "body")
        handlers = find_elements_by_tag(stmt, "handlers")
        orelse = find_elements_by_tag_orNUll(stmt, "orelse")
        finalbody = find_element_by_tag(stmt, "finalbody")

        body_seq = []
        for b in body:
            b_seq = self.parse_stmt(b)
            body_seq.extend(b_seq)
        body_block = [{
            "guard":"",
            "contents":body_seq
        }]
        seq.extend(body_block)
        if handlers is not None:
            handler_seq = [self.parse_excepthandler(h) for h in handlers]
            seq.extend(handler_seq)
        if orelse is not None:
            orelse_seq = []
            for o in orelse:
                o_seq = self.parse_stmt(o)
                orelse_seq.extend(o_seq)
            orelse_block = [{
                "guard":"else",
                "contents":orelse_seq
            }]
            seq.extend(orelse_block)
        if finalbody is not None:
            f_seq = self.parse_stmt(finalbody)
            finally_block = [{
                "guard":"finally",
                "contents":f_seq
            }]
            seq.extend(finally_block)
        try_except_block = [{
            "type": "blocks",
            "name": "try",
            "blocks":seq
        }]
        return try_except_block

    def parse_excepthandler(self, handlers):
        """
        return {"guard": "exception", "contents":[body]}
        """
        except_name = ""
        if handlers.attrib.get("name") is not None:
            except_name = handlers.attrib.get("name")
            
        typee = find_element_by_tag(handlers, "type")
        bodys = find_elements_by_tag(handlers, "body")
        t_code = ''
        if typee is not None:
            t_seq, t_code = self.parse_expr(typee)
        seq = []
        for b in bodys:
            b_seq = self.parse_stmt(b)
            seq.extend(b_seq)
        return {
            "guard": "except " + t_code + ("" if except_name == "None" else (" as "+ except_name)),
            "contents": seq
        }

    def process_raise(self, stmt):
        """
        parse Raise statment
        """
        seq = []
        exc = find_element_by_tag(stmt, "exc")
        cause = find_element_by_tag(stmt, "cause")
        if exc is not None and cause is not None:
            exc_seq, exc_code_expr = self.parse_expr(exc)
            cause_seq, cause_code_expr = self.parse_expr(cause)
            seq.extend(exc_seq)
            seq.extend(cause_seq)
            seq.extend([{
                "type": "controlFlow",
                "name": "raise",
                "value": exc_code_expr + " from " + cause_code_expr
            }])
            return seq
        elif exc is not None and cause is None:
            exc_seq, exc_code_expr = self.parse_expr(exc)
            seq.extend(exc_seq)
            seq.extend([{
                "type": "controlFlow",
                "name": "raise",
                "value": exc_code_expr
            }])
            return seq
        elif exc is None and cause is None:
            seq.extend([{
                "type": "controlFlow",
                "name": "raise",
                "value": ""
            }])
            return seq

    def process_assert(self, stmt):
        """
        parse Assert statment
        """
        seq = []
        test = find_element_by_tag(stmt, "test")
        msg = find_element_by_tag(stmt, "msg")
        if msg is not None:
            test_seq, test_code_expr = self.parse_expr(test)
            msg_seq, msg_code_expr = self.parse_expr(msg)
            seq.extend(test_seq)
            seq.extend(msg_seq)
            seq.extend([{
                "type": "blocks",
                "name": "if",
                "blocks": [
                    {
                        "guard": test_code_expr,
                        "contents": [{
                                        "type": "controlFlow",
                                        "name": "raise",
                                        "value": msg_code_expr
                                    }]
                    }
                ]
            }])
            return seq
        else:
            test_seq, test_code_expr = self.parse_expr(test)
            seq.extend(test_seq)
            seq.extend([{
                "type": "blocks",
                "name": "if",
                "blocks": [
                    {
                        "guard": test_code_expr,
                        "contents": [{
                                        "type": "controlFlow",
                                        "name": "raise",
                                        "value": ""
                                    }]
                    }
                ]
            }])
            return seq

    def process_import(self, stmt):
        """
        parse Import statment
        """
        pass

    def process_importfrom(self, stmt):
        """
        parse ImportFrom statment
        """
        pass

    def process_global(self, stmt):
        """
        parse Global statment
        """
        pass

    def process_nonlocal(self, stmt):
        """
        parse NonLocal statment
        """
        pass

    def process_expr(self, stmt):
        """
        parse Expr statment
        """
        value = find_element_by_tag(stmt, "value")
        return self.parse_expr(value)[0]

    def process_pass(self, stmt):
        """
        parse Pass statment
        """
        seq = []
        seq.extend([{
            "type":"controlFlow",
            "name":"pass",
            "value": ""
        }])
        return seq

    def process_break(self, stmt):
        """
        parse Break statment
        """
        seq = []
        seq.extend([{
            "type":"controlFlow",
            "name":"break",
            "value": ""
        }])
        return seq

    def process_continue(self, stmt):
        """
        parse Continue statment
        """
        seq = []
        seq.extend([{
            "type":"controlFlow",
            "name":"continue",
            "value": ""
        }])
        return seq





    # Expression Grammar

    def parse_expr(self, expr):
        """
        Generate the expression sequence return in json form
        """
        expr_handlers = {
            "BoolOp": self.process_boolop,
            "NamedExpr": self.process_namedexpr,
            "BinOp": self.process_binop,
            "UnaryOp": self.process_unaryop,
            "Lambda": self.process_lambda,
            "IfExp": self.process_ifexp,
            "Dict": self.process_dict,
            "Set": self.process_set,
            "ListComp": self.process_listcomp,
            "SetComp": self.process_setcomp,
            "DictComp": self.process_dictcomp,
            "GeneratorExp": self.process_generatorexp,
            "Await": self.process_await,
            "Yield": self.process_yield,
            "YieldFrom": self.process_yieldfrom,
            "Compare": self.process_compare,
            "Call": self.process_call,
            "FormattedValue": self.process_formattedvalue,
            "JoinedStr": self.process_joinedstr,
            "Constant": self.process_constant,
            "Attribute": self.process_attribute,
            "Subscript": self.process_subscript,
            "Starred": self.process_starred,
            "Name": self.process_name,
            "List": self.process_list,
            "Tuple": self.process_tuple,
            "Slice": self.process_slice,
        }
        ignore_expr = []
        if expr is None:
            return [], ""
        if expr.attrib.get(xsi_attrib) not in ignore_expr:
            return expr_handlers[expr.attrib.get(xsi_attrib)](expr)
        else:
            return [], ""

    def process_boolop(self, expr):
        bool_operator_handlers = {
            "And": "and",
            "Or": "or"
        }
        operator = find_element_by_tag(expr, "op")
        boolop_children = [ch for ch in expr.getchildren() if ch.tag != "op"]
        seq = []
        code_expr = ""
        i = 0
        for ch in boolop_children:
            ch_seq, ch_code_exp = self.parse_expr(ch)
            seq.extend(ch_seq)
            code_expr += ch_code_exp + f" {bool_operator_handlers[operator.attrib.get(xsi_attrib)]} "
        code_expr = code_expr[:-(len(bool_operator_handlers[operator.attrib.get(xsi_attrib)])+2)]
        return seq, code_expr

    def process_namedexpr(self, expr):
        """
        Procerss NamedExpr expression
        """
        target = find_element_by_tag(expr, "target")
        value = find_element_by_tag(expr, "value")
        seq = []
        code_expr = ""
        target_seq, target_code_expr = self.parse_expr(target)
        value_seq, value_code_expr = self.parse_expr(value)
        seq.extend(target_seq)
        seq.extend(value_seq)
        code_expr += target_code_expr + ":" + value_code_expr
        return seq, code_expr

    def process_binop(self, expr):
        """
        Generate binary operation seq and code expression
        we have problem here that if there are '(' ')' in the code we can not now from the ast do we have them or not
        if we add them always it will be mistake when we don't have them in the code
        if we didn't add them always it will be mistake when we have them in the code.
        """
        bin_operator_handlers = {
            "Add": "+",
            "Sub": "-",
            "Mult": "*",
            "MatMult": "@",
            "Div": "/",
            "Mod": "%",
            "Pow": "**",
            "LShift": "<<",
            "RShift": ">>",
            "BitOr": "|",
            "BitXor": "^",
            "BitAnd": "&",
            "FloorDiv": "//"
        }
        expr_children = expr.getchildren()
        left, right, op = None, None, None
        for ex in expr_children:
            if ex.tag == "left":
                left = ex
            elif ex.tag == "right":
                right = ex
            elif ex.tag == "op":
                op = ex

        if left.attrib.get(xsi_attrib) == "BinOp" and right.attrib.get(xsi_attrib) == "BinOp":
            left_seq, left_code_expr = self.parse_expr(left)
            right_seq, right_code_expr = self.parse_expr(right)
            left_seq.extend(right_seq)
            seq = left_seq
            code_expr = left_code_expr + f" {bin_operator_handlers[op.attrib.get(xsi_attrib)]} " + right_code_expr
            return seq, code_expr

        elif left.attrib.get(xsi_attrib) == "BinOp" and right.attrib.get(xsi_attrib) != "BinOp":
            left_seq, left_code_expr = self.parse_expr(left)
            right_seq, right_code_expr = self.parse_expr(right)
            left_seq.extend(right_seq)
            seq = left_seq
            code_expr = left_code_expr + f" {bin_operator_handlers[op.attrib.get(xsi_attrib)]} " + right_code_expr
            return seq, code_expr
        
        elif left.attrib.get(xsi_attrib) != "BinOp" and right.attrib.get(xsi_attrib) == "BinOp":
            left_seq, left_code_expr = self.parse_expr(left)
            right_seq, right_code_expr = self.parse_expr(right)
            left_seq.extend(right_seq)
            seq = left_seq
            code_expr = left_code_expr + f" {bin_operator_handlers[op.attrib.get(xsi_attrib)]} " + right_code_expr 
            return seq, code_expr
        
        elif left.attrib.get(xsi_attrib) != "BinOp" and right.attrib.get(xsi_attrib) != "BinOp":
            left_seq, left_code_expr = self.parse_expr(left)
            right_seq, right_code_expr = self.parse_expr(right)
            left_seq.extend(right_seq)
            seq = left_seq
            code_expr = left_code_expr + f" {bin_operator_handlers[op.attrib.get(xsi_attrib)]} " + right_code_expr
            return seq, code_expr

    def process_unaryop(self, expr):
        un_operator_handlers = {
            "Invert": "~",
            "Not": "!",
            "UAdd": "+",
            "USub": "-" 
        }
        seq, code_expr = self.parse_expr(find_element_by_tag(expr, "operand"))
        return seq, un_operator_handlers[find_element_by_tag(expr, "op").attrib.get(xsi_attrib)] + code_expr

    def process_lambda(self, expr):
        parameters = self.parse_parameters(expr)
        body_seq, body_code_expr = self.parse_expr(find_element_by_tag(expr, "body"))
        return [], f"Lambda : {', '.join(parameters[1].keys())} :" + body_code_expr

    def process_ifexp(self, expr):
        """
        parse short if expression
        """
        # parent_seq, parent_code_expr = self.parse_expr(find_element_by_tag(expr.getparent(),"targets"))  
        test_seq, test_code_expr = self.parse_expr(find_element_by_tag(expr, "test"))
        body_seq, body_code_expr = self.parse_expr(find_element_by_tag(expr, "body"))
        orelse_seq, orelse_code_expr = self.parse_expr(find_element_by_tag(expr, "orelse"))
        code_expr = body_code_expr + " if " + test_code_expr + " else " + orelse_code_expr
        seq = []
        seq.extend(test_seq)
        seq.extend([{
            "type": "blocks",
            "name": "if",
            "blocks": [ {
                        "guard":test_code_expr,
                        "contents":body_seq
                        },
                        {
                        "guard":"else",
                        "contents":orelse_seq
                        }
                    ]
        }])
        return seq, code_expr
        
    def process_dict(self, expr):
        """
        parse dictionary creation
        """
        keys = [ch for ch in expr.getchildren() if (ch.tag == "keys" or ch.tag == "key")]
        values = [ch for ch in expr.getchildren() if (ch.tag == "values" or ch.tag == "value")]
        code_expr = "{"
        seq = []
        for i in range(len(keys)):
            key_seq, key_code_expr = self.parse_expr(keys[i])
            value_seq, value_code_expr = self.parse_expr(values[i])
            code_expr += key_code_expr + ":" + value_code_expr + ", "
            seq.extend(key_seq)
            seq.extend(value_seq)
        if code_expr.endswith(", "):
            code_expr = code_expr[:-2]
        code_expr += "}"
        return seq, code_expr

    def process_set(self, expr):
        """
        parse set creation
        """
        elts = find_elements_by_tag(expr, "elts")
        code_expr = "{"
        seq = []
        for e in elts:
            elts_seq, elts_code_expr = self.parse_expr(e)
            code_expr += elts_code_expr +", "
            seq.extend(elts_seq)
        if code_expr.endswith(", "):
            code_expr = code_expr[:-2]
        code_expr = code_expr + "}" 
        return seq, code_expr

    def process_listcomp(self, expr):
        """
        parse ListComp
        """
        elt = find_element_by_tag(expr, "elt")
        generators = find_element_by_tag(expr, "generators")
        elt_seq, elt_code_expr = self.parse_expr(elt)
        target_seq, target_code_expr, iter_seq, iter_code_expr, ifs_seq, ifs_code_expr = self.parse_comprehension(generators)
        seq = []
        seq.extend(iter_seq)
        seq.extend(target_seq)
        code_expr = "[" + elt_code_expr + " for " + target_code_expr + " in " + iter_code_expr
        if ifs_seq == None:
            seq.extend([{
                "type": "blocks",
                "name": "loop",
                "blocks": [
                    {
                        "guard": target_code_expr + " in " + iter_code_expr,
                        "contents": elt_seq
                    }
                ]
            }])
            code_expr += "]"
        else:
            seq.extend([{
                "type": "blocks",
                "name": "loop",
                "blocks": [
                    {
                        "guard": target_code_expr + " in " + iter_code_expr,
                        "contents": [{
                            "type":"blocks",
                            "name":"if",
                            "blocks":[
                                {
                                    "guard":ifs_code_expr,
                                    "contents":elt_seq
                                }
                            ]
                        }]
                    }
                ]
            }])
            code_expr += " if " + ifs_code_expr + "]"
        return seq, code_expr

    def process_setcomp(self, expr):
        """
        parse SetComp
        """
        elt = find_element_by_tag(expr, "elt")
        generators = find_element_by_tag(expr, "generators")
        elt_seq, elt_code_expr = self.parse_expr(elt)
        target_seq, target_code_expr, iter_seq, iter_code_expr, ifs_seq, ifs_code_expr = self.parse_comprehension(generators)
        seq = []
        seq.extend(iter_seq)
        seq.extend(target_seq)
        code_expr = "{" + elt_code_expr + " for " + target_code_expr + " in " + iter_code_expr
        if ifs_seq == None:
            seq.extend([{
                "type": "blocks",
                "name": "loop",
                "blocks": [
                    {
                        "guard": target_code_expr + " in " + iter_code_expr,
                        "contents": elt_seq
                    }
                ]
            }])
            code_expr += "}"
        else:
            seq.extend([{
                "type": "blocks",
                "name": "loop",
                "blocks": [
                    {
                        "guard": target_code_expr + " in " + iter_code_expr,
                        "contents": [{
                            "type":"blocks",
                            "name":"if",
                            "blocks":[
                                {
                                    "guard":ifs_code_expr,
                                    "contents":elt_seq
                                }
                            ]
                        }]
                    }
                ]
            }])
            code_expr += " if " + ifs_code_expr + "}"
        return seq, code_expr

    def process_dictcomp(self, expr):
        """
        parse DictComp
        """
        key = find_element_by_tag(expr, "key")
        value = find_element_by_tag(expr, "value")
        generators = find_element_by_tag(expr, "generators")
        key_seq, key_code_expr = self.parse_expr(key)
        value_seq, value_code_expr = self.parse_expr(value)
        elt_seq = []
        elt_seq.extend(key_seq)
        elt_seq.extend(value_seq)
        target_seq, target_code_expr, iter_seq, iter_code_expr, ifs_seq, ifs_code_expr = self.parse_comprehension(generators)
        seq = []
        seq.extend(iter_seq)
        seq.extend(target_seq)
        code_expr = "{" + key_code_expr + " : " + value_code_expr + " for " + target_code_expr + " in " + iter_code_expr
        if ifs_seq == None:
            seq.extend([{
                "type": "blocks",
                "name": "loop",
                "blocks": [
                    {
                        "guard": target_code_expr + " in " + iter_code_expr,
                        "contents": elt_seq
                    }
                ]
            }])
            code_expr += "}"
        else:
            seq.extend([{
                "type": "blocks",
                "name": "loop",
                "blocks": [
                    {
                        "guard": target_code_expr + " in " + iter_code_expr,
                        "contents": [{
                            "type":"blocks",
                            "name":"if",
                            "blocks":[
                                {
                                    "guard":ifs_code_expr,
                                    "contents":elt_seq
                                }
                            ]
                        }]
                    }
                ]
            }])
            code_expr += " if " + ifs_code_expr + "}"
        return seq, code_expr

    def process_generatorexp(self, expr):
        """
        Parse GenerateorExpr expression
        """
        elt = find_element_by_tag(expr, "elt")
        generators = find_element_by_tag(expr, "generators")
        elt_seq, elt_code_expr = self.parse_expr(elt)
        target_seq, target_code_expr, iter_seq, iter_code_expr, ifs_seq, ifs_code_expr = self.parse_comprehension(generators)
        seq = []
        seq.extend(target_seq)
        seq.extend(iter_seq)
        # seq.extend(iter_seq)
        code_expr = elt_code_expr + " for " + target_code_expr + " in " + iter_code_expr
        if ifs_seq == None:
            seq.extend([{
                "type": "blocks",
                "name": "loop",
                "blocks": [
                    {
                        "guard": target_code_expr + " in " + iter_code_expr,
                        "contents": elt_seq
                    }
                ]
            }])
        else:
            seq.extend([{
                "type": "blocks",
                "name": "loop",
                "blocks": [
                    {
                        "guard": target_code_expr + " in " + iter_code_expr,
                        "contents": [{
                            "type":"blocks",
                            "name":"if",
                            "blocks":[
                                {
                                    "guard":ifs_code_expr,
                                    "contents":elt_seq
                                }
                            ]
                        }]
                    }
                ]
            }])
            code_expr += " if " + ifs_code_expr
        return seq, code_expr

    def process_await(self, expr):
        """
        parse Await
        """
        ch = find_element_by_tag(expr, "value")
        ch_seq, ch_code = self.parse_expr(ch)
        seq = []
        seq.extend(ch_seq)
        seq.extend([{
            "type": "controlFlow",
            "name": "await",
            "value": ch_code
        }])
        return seq, "await " + ch_code

    def process_yield(self, expr):
        yield_child = find_element_by_tag(expr, "value")
        # parse depth
        yield_seq = []
        yield_code_expr = ''
        if yield_child is not None:
            yield_seq, yield_code_expr = self.parse_expr(yield_child)
        # append return call to depth as controlFlow
        yield_seq.extend([{"type":"controlFlow", 
                            "name": "yield", 
                            "value": yield_code_expr
                        }])
        return yield_seq, "yield "+yield_code_expr

    def process_yieldfrom(self, expr):
        yield_child = find_element_by_tag(expr, "value")
        # parse depth
        yield_seq, yield_code_expr = self.parse_expr(yield_child)
        # append return call to depth as controlFlow
        yield_seq.extend([{"type":"controlFlow", 
                            "name": "yield", 
                            "value": yield_code_expr
                        }])
        return yield_seq, "yield "+yield_code_expr

    def process_compare(self, expr):
        cmp_op_handlers = {
            "Eq": "==",
            "NotEq": "!=",
            "Lt": "<",
            "LtE": "<=",
            "Gt": ">",
            "GtE": ">=",
            "Is": "is",
            "IsNot": "is not",
            "In": "in",
            "NotIn": "not in"
        }
        left = find_element_by_tag(expr, "left")
        operator = find_element_by_tag(expr, "ops")
        comparators = find_element_by_tag(expr, "comparators")
        left_seq, left_code_expr = self.parse_expr(left)
        comp_seq, comp_code_expr = self.parse_expr(comparators)
        left_seq.extend(comp_seq)
        seq = left_seq
        code_expr = left_code_expr + " " + cmp_op_handlers[operator.attrib.get(xsi_attrib)] + " " + comp_code_expr
        return seq, code_expr

    def process_call(self, expr):
        func = find_element_by_tag(expr, "func")
        args = find_elements_by_tag_orNUll(expr, "args")
        keywords = find_elements_by_tag_orNUll(expr, "keywords")
        seq = []
        func_args_code_expr = "("
        if args is not None:
            args_parsed_seq = {}
            for ar in args:
                arg_seq, arg_code_expr = self.parse_expr(ar)
                args_parsed_seq[arg_code_expr]=arg_seq
                seq.extend(arg_seq)
                func_args_code_expr += arg_code_expr + ", "
        if keywords is not None:
            keywords_parsed_seq = {}
            for keyword in keywords:
                keyword_seq, keyword_code_expr = self.parse_keyword(keyword)
                keywords_parsed_seq[keyword_code_expr] = keyword_seq
                seq.extend(keyword_seq)
                func_args_code_expr += keyword_code_expr + ", "
        if func_args_code_expr.endswith(", "):
            func_args_code_expr = func_args_code_expr[:-2]
        func_args_code_expr += ")"

        func_seq, func_code_expr = self.parse_expr(func)
        parts = split_string(func_code_expr)
        to = []
        if len(parts) > 1:
            to.extend(parts[:-1])
            func_code_expr = parts[-1]
        
        seq.extend(func_seq)
        func_code_expr += func_args_code_expr
        seq.extend([
            {
                "type":"methodInvocation",
                "to":to,
                "method": func_code_expr
            }
        ])
        if len(to)>0:
            return seq,  '.'.join(to) + "." + func_code_expr
        return seq, func_code_expr
        
    def process_formattedvalue(self, expr):
        """
        Procerss FormattedValue expression
        """
        value = find_element_by_tag(expr, "value")
        value_seq, value_code_expr = self.parse_expr(value)
        if value.attrib.get("type") == "str":
            return value_seq, "{"+value_code_expr[1:-1]+"}"
        else:
            return value_seq, "{"+value_code_expr+"}"

    def process_joinedstr(self, expr):
        """
        Procerss JoineStr expression
        """
        values = find_elements_by_tag(expr, "values")
        seq = []
        code_expr = ""
        for v in values:
            v_seq, v_code = self.parse_expr(v)
            seq.extend(v_seq)
            if v.attrib.get("type") == "str":
                code_expr += v_code[1:-1]
            else:
                code_expr += v_code
        return seq, "f'"+code_expr+"'"

    def process_constant(self, expr):
        """
        Procerss constant expression
        """
        if expr.attrib.get("type") == "str":
            return [], f'"{expr.attrib.get("value")}"'
        elif expr.attrib.get("type") == "int":
            return [], expr.attrib.get('value')
        else:
            return [], expr.attrib.get('value')

    def process_attribute(self, expr):
        """
        Procerss Attribute expression
        """
        value = find_element_by_tag(expr, "value")
        value_seq, value_code_expr = self.parse_expr(value)
        return value_seq, value_code_expr+"."+expr.attrib.get("attr")

    def process_subscript(self, expr):
        """
        Process Subscript expression
        """
        value = find_element_by_tag(expr, "value")
        slic = find_element_by_tag(expr, "slice")
        value_seq, value_code_expr = self.parse_expr(value)
        slice_seq, slice_code_expr = self.parse_expr(slic)
        seq = []
        seq.extend(slice_seq)
        seq.extend(value_seq)
        if slic.attrib.get(xsi_attrib) != "Slice":
            return seq, value_code_expr + "["+slice_code_expr+"]"
        else:
            return seq, value_code_expr + slice_code_expr

    def process_starred(self, expr):
        """
        Process Starred expression
        """
        value = find_element_by_tag(expr, "value")
        seq, code_expr = self.parse_expr(value)
        return seq, "*"+code_expr

    def process_name(self, expr):
        """
        Process Name expression
        """
        return [], expr.attrib.get('id')

    def process_list(self, expr):
        """
        Process List expression
        """
        elts = find_elements_by_tag(expr, "elts")
        seq = []
        code_expr = "["
        for el in elts:
            el_seq, el_code_expr = self.parse_expr(el)
            seq.extend(el_seq)
            code_expr += el_code_expr + ", "
        if code_expr.endswith(", "):
            code_expr = code_expr[:-2]
        code_expr = code_expr + "]"
        return seq, code_expr

    def process_tuple(self, expr):
        """
        Process Tuple expression
        """
        elts = find_elements_by_tag(expr, "elts")
        seq = []
        code_expr = "("
        for el in elts:
            el_seq, el_code_expr = self.parse_expr(el)
            seq.extend(el_seq)
            code_expr += el_code_expr + ", "
        if code_expr.endswith(", "):
            code_expr = code_expr[:-2]
        code_expr = code_expr + ")"
        return seq, code_expr

    def process_slice(self, expr):
        """
        Process Slice expression
        """
        lower = find_element_by_tag(expr, "lower")
        upper = find_element_by_tag(expr, "upper")
        step = find_element_by_tag(expr, "step")
        seq = []
        code_expr = "["
        if lower is not None:
            lower_seq, lower_code_expr = self.parse_expr(lower)
            seq.extend(lower_seq)
            code_expr += lower_code_expr
        if upper is not None:
            upper_seq, upper_code_expr = self.parse_expr(upper)
            seq.extend(upper_seq)
            code_expr += ":"+ upper_code_expr
        if step is not None:
            step_seq, step_code_expr = self.parse_expr(step)
            seq.extend(step_seq)
            code_expr += ":"+ step_code_expr
        code_expr += "]"
        return seq, code_expr



    # items


    def parse_withitem(self, item):
        """
        Process WithItem
        """
        seq = []
        code_expr = "" 
        context_expr = find_element_by_tag(item, "context_expr")
        optional_vars = find_element_by_tag(item, "optional_vars")
        con_expr_seq, con_expr_code = self.parse_expr(context_expr)
        seq.extend(con_expr_seq)
        code_expr += con_expr_code
        if optional_vars is not None:
            ov_seq, ov_code = self.parse_expr(optional_vars)
            seq.extend(ov_seq)
            # seq.extend([{"type":"scopedVariable",
            #             "name":ov_code}])
            code_expr += " as " + ov_code
        return seq, code_expr


    def parse_pattern(self, pattern):
        """
        Parse Pattern
        """
        patterns = None
        match pattern.attrib.get(xsi_attrib):
            case "MatchValue":
                return self.parse_expr(find_element_by_tag(pattern, "value"))
            case "MatchSingleton":
                return [], pattern.attrib.get("value")
            case "MatchSequence":
                patterns = find_elements_by_tag_orNUll(pattern, "patterns")
                if patterns is None:
                    return [], "[]"
                seq = []
                code = "["
                for p in patterns:
                    p_seq, p_code = self.parse_pattern(p)
                    # seq.extend(p_seq)
                    code += p_code + ", "
                if code.endswith(", "):
                    code = code[:-2]
                code = code + ']'
                return seq, code
            case "MatchMapping":
                rest = True if pattern.attrib.get("rest") == "rest" else False
                keys = find_elements_by_tag_orNUll(pattern, "keys")
                patterns = find_elements_by_tag_orNUll(pattern, "patterns")
                if keys is None and patterns is None:
                    return [], "{}"
                seq = []
                code = "{"
                for i in range(len(keys)):
                    k_seq, k_code = self.parse_expr(keys[i])
                    p_seq, p_code = self.parse_pattern(patterns[i])
                    # seq.extend(k_seq)
                    # seq.extend(p_seq)
                    code += k_code + ": " + p_code + ", "
                if rest:
                    code += "**rest, "
                code = code[:-2] + "}"
                return seq, code
            case "MatchClass":
                cls = find_element_by_tag(pattern, "cls")
                patterns = find_elements_by_tag_orNUll(pattern, "patterns")
                kwd_patterns = find_elements_by_tag_orNUll(pattern, "kwd_patterns")
                
                cls_seq, cls_code = self.parse_expr(cls)
                seq = []
                code = cls_code + "("
                if patterns is not None:
                    for p in patterns:
                        p_seq, p_code = self.parse_pattern(p)
                        code += p_code + ", "
                    if(kwd_patterns is None):
                        code = code[:-2]
                if kwd_patterns is not None:
                    for k in kwd_patterns:
                        k_seq, k_code = self.parse_pattern(k)
                        # seq.extend(k_seq)
                        code += k_code + ", "
                    if code.endswith(", "):
                        code = code[:-2]
                code += ")"
                return seq, code
            case "MatchStar":
                pass
            case "MatchAs":
                return [], pattern.attrib.get("name")
            case "MatchOr":
                patterns = find_elements_by_tag_orNUll(pattern, "patterns")
                seq = []
                code = ""
                if patterns is not None:
                    for p in patterns:
                        p_seq, p_code = self.parse_pattern(p)
                        # seq.extend(p_seq)
                        code += p_code + " | "
                    code = code[:-2]
                return seq, code



    def process_String(self, expr):
        return expr.attrib.get('value')

    def process_Int(self, expr):
        return expr.attrib.get('value')

    def process_Identifier(self, expr):
        return expr.attrib.get('value')



    def parse_comprehension(self, comp):
        generators_target = find_element_by_tag(comp, "target")
        generators_iter = find_element_by_tag(comp, "iter")
        generators_ifs = find_elements_by_tag_orNUll(comp, "ifs")
        target_seq, target_code_expr = self.parse_expr(generators_target)
        iter_seq, iter_code_expr = self.parse_expr(generators_iter)
        if generators_ifs is not None:
            ifs_seq = []
            ifs_code_expr = ""
            for ifs in generators_ifs:
                seq, code = self.parse_expr(ifs)
                ifs_seq.extend(seq)
                ifs_code_expr += code + " and "
            ifs_code_expr = ifs_code_expr[:-5]
            return target_seq, target_code_expr, iter_seq, iter_code_expr, ifs_seq, ifs_code_expr
        else:
            return target_seq, target_code_expr, iter_seq, iter_code_expr, None, None

    def parse_keyword(self, keyword):
        """
        Process Keyword
        """
        arg = keyword.attrib.get("arg")
        value = find_element_by_tag(keyword, "value")
        v_seq, v_code = self.parse_expr(value)
        return v_seq, arg+"="+v_code
