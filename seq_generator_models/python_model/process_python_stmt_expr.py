# not need anymore



# import xml.etree.ElementTree as ET
# import traceback
# from ..process_xmi import *
# from process_python_model import *


# # xsi namespace
# NAMESPACE = {'xsi': 'http://www.w3.org/2001/XMLSchema-instance'}
# ET.register_namespace('xsi', NAMESPACE['xsi'])
# xsi_attrib = f"{{{NAMESPACE['xsi']}}}type"


# def parse_parameters(method_declaration):
#     """
#     Extract the function parameters return as dictionary 
#     {arg1:default_value1, .....} if no default_value them None
#     """
#     method_childrens = method_declaration.getchildren()
#     args = None
#     for ch in method_childrens:
#         if ch.tag == "args":
#             args = ch
#             break
#     if args is None:
#         raise Exception("Function declaration parameters not found")
#     args_childrens = args.getchildren()
#     args_name = [ar.attrib.get('arg') for ar in args_childrens if ar.tag == "args"]
#     args_defaults = [ar.attrib.get('value') for ar in args_childrens if ar.tag == "defaults"]
#     len_def = len(args_name) - len(args_defaults)
#     args_return = {key: None for key in args_name[:len_def]}
#     args_return.update(zip(args_name[len_def:], args_defaults))

#     return args_return



# def parse_stmt(stmt):
#     """
#     Generate the statements sequence return in json form
#     """
#     stmt_handlers = {
#         "FunctionDef": process_functiondef,
#         "AsyncFunctionDef": process_asyncfunctiondef,
#         "ClassDef": process_classdef,
#         "Return": process_return,
#         "Delete": process_delete,
#         "Assign": process_assign,
#         "TypeAlias": process_typealias,
#         "AugAssign": process_augassign,
#         "AnnAssign": process_annassign,
#         "For": process_for,
#         "AsyncFor": process_asyncfor,
#         "While": process_while,
#         "If": process_if,
#         "With": process_with,
#         "AsyncWith": process_ayncwith,
#         "Match": process_match,
#         "Raise": process_raise,
#         "Try": process_try,
#         "TryStar": process_trystar,
#         "Assert": process_assert,
#         "Import": process_import,
#         "ImportFrom": process_importfrom,
#         "Global": process_global,
#         "Nonlocal": process_nonlocal,
#         "Expr": process_expr,
#         "Pass": process_pass,
#         "Break": process_break,
#         "Continue": process_continue,
#     }
    
    
    
#     # re add stmts
#     ignore_stmt = ["FunctionDef", 
#                    "AsyncFunctionDef", 
#                    "ClassDef",
#                    "Import",
#                    "ImportFrom",
#                    "Global",
#                    "Nonlocal",
#                    ]
#     if stmt.attrib.get(xsi_attrib) not in ignore_stmt:
#         return stmt_handlers[stmt.attrib.get(xsi_attrib)](stmt)
#     else:
#         return []

# def process_functiondef(stmt):
#     pass

# def process_asyncfunctiondef(stmt):
#     pass

# def process_classdef(stmt):
#     pass

# def process_return(stmt):
#     """
#     parse return statment
#     """
#     return_child = find_element_by_tag(stmt, "value")
#     # parse depth
#     return_seq = []
#     if return_child is not None:
#         return_seq, return_code_expr = parse_expr(return_child)
#         # append return call to depth as controlFlow
#         return_seq.extend([{"type":"controlFlow", 
#                             "name": "return", 
#                             "value": return_code_expr
#                         }])
#         return return_seq
#     else:
#         return_seq.extend([{"type":"controlFlow", 
#                             "name": "return", 
#                             "value": None
#                         }])
#         return return_seq
    
# def process_delete(stmt):
#     """
#     parse delete statment
#     """
#     delete_child = stmt.getchildren()[0]
#     delete_seq, delete_code_expr = parse_expr(delete_child)
#     delete_code_expr = f"delete {delete_code_expr}"
#     delete_seq.extend([{"type":"methodInvokation",
#                         "to": [],
#                         "method": delete_code_expr
#                     }])
#     return delete_seq

# def process_assign(stmt):
#     """
#     parse assign statment
#     """
#     target = find_element_by_tag(stmt, "targets")
#     value = find_element_by_tag(stmt, "value")
#     target_seq, target_code_expr = parse_expr(target)
#     value_seq, value_code_expr = parse_expr(value)
#     seq = []
#     seq.extend(target_seq)
#     seq.extend(value_seq)
#     typee = None
#     if value.attrib.get(xsi_attrib) in ['Constant','String', 'Int', 'Identifier']:
#         typee = "scopedVariable"
#     elif value.attrib.get(xsi_attrib) == "Call":
#         typee = "newInstance"
#     else:
#         typee = "scopedVariable"
#     seq.extend([{
#         "type":typee,
#         "target":target_code_expr,
#         "new_type":value_code_expr
#     }])

#     return seq

# def process_typealias(stmt):
#     pass

# def process_augassign(stmt):
#     """
#     parse augmented assignment statment
#     """
#     bin_operator_handlers = {
#         "Add": "+",
#         "Sub": "-",
#         "Mult": "*",
#         "MatMult": "@",
#         "Div": "/",
#         "Mod": "%",
#         "Pow": "**",
#         "LShift": "<<",
#         "RShift": ">>",
#         "BitOr": "|",
#         "BitXor": "^",
#         "BitAnd": "&",
#         "FloorDiv": "//"
#     }
#     target = find_element_by_tag(stmt, "target")
#     op = find_element_by_tag(stmt, "op")
#     value = find_element_by_tag(stmt, "value")
#     target_seq, target_code_expr = parse_expr(target)
#     value_seq, value_code_expr = parse_expr(value)
#     seq = []
#     seq.extend(target_seq)
#     seq.extend(value_seq)
#     seq.extend([{
#         "type":"augmentedAssignment",
#         "target":target_code_expr,
#         "op":bin_operator_handlers[op.attrib.get(xsi_attrib)],
#         "value":value_code_expr
#     }])
#     return seq

# def process_annassign(stmt):
#     """
#     parse annotated assignment statment
#     """
#     target = find_element_by_tag(stmt, "target")
#     annotation = find_element_by_tag(stmt, "annotation")
#     value = find_element_by_tag(stmt, "value")
#     target_seq, target_code_expr = parse_expr(target)
#     annotation_seq, annotation_code_expr = parse_expr(annotation)
#     seq = []
#     seq.extend(annotation_seq)
#     seq.extend(target_seq)
#     if value is not None:
#         value_seq, value_code_expr = parse_expr(value)
#         seq.extend(value_seq)
#     else:
#         value_code_expr = None
#     # value could be None check it in sequence draw
#     seq.extend([{
#         "type":"newAnnotatedInstance",
#         "name":target_code_expr,
#         "new_type":annotation_code_expr,
#         "value":value_code_expr
#     }])
#     return seq

# def process_for(stmt):
#     """
#     parse for statment
#     """
#     seq = []
#     init_seq = []
#     target = find_element_by_tag(stmt, "target")
#     iter = find_element_by_tag(stmt, "iter")
#     body = find_elements_by_tag(stmt, "body")
#     orelse = find_elements_by_tag(stmt, "orelse")
#     target_seq, target_code_expr = parse_expr(target)
#     iter_seq, iter_code_expr = parse_expr(iter)
#     body_seq = []
#     for b in body:
#         b_seq = parse_stmt(b)
#         body_seq.extend(b_seq)

#     init_seq.extend(target_seq)
#     init_seq.extend([{
#         "type": "scopedVariable",
#         "name": target_code_expr
#     }])
#     init_seq.extend(iter_seq)
    

#     if orelse is not None:
#         orelse_seq = []
#         for o in orelse:
#             o_seq = parse_stmt(o)
#             orelse_seq.extend(o_seq)
#         seq.extend([{
#             "type": "block",
#             "name": "for",
#             "blocks": [
#                 {
#                     "guard": "for init",
#                     "contents": init_seq
#                 },{
#                     "guard": "for "+target_code_expr + " in " +  iter_code_expr,
#                     "contents": body_seq
#                 },{
#                     "guard": "else",
#                     "contents": orelse_seq
#                 }
#             ]
#         }])
#     else:
#         seq.extend([{
#             "type": "block",
#             "name": "for",
#             "blocks": [
#                 {
#                     "guard": "for init",
#                     "contents": init_seq
#                 },{
#                     "guard": "for "+target_code_expr + " in " +  iter_code_expr,
#                     "contents": body_seq
#                 }
#             ]
#         }])
    
#     return seq

# def process_asyncfor(stmt):
#     """
#     parse asyncfor statment
#     """
#     seq = []
#     init_seq = []
#     target = find_element_by_tag(stmt, "target")
#     iter = find_element_by_tag(stmt, "iter")
#     body = find_elements_by_tag(stmt, "body")
#     orelse = find_elements_by_tag(stmt, "orelse")
#     target_seq, target_code_expr = parse_expr(target)
#     iter_seq, iter_code_expr = parse_expr(iter)
#     body_seq = []
#     for b in body:
#         b_seq = parse_stmt(b)
#         body_seq.extend(b_seq)

#     init_seq.extend(target_seq)
#     init_seq.extend([{
#         "type": "scopedVariable",
#         "name": target_code_expr
#     }])
#     init_seq.extend(iter_seq)
    

#     if orelse is not None:
#         orelse_seq = []
#         for o in orelse:
#             o_seq = parse_stmt(o)
#             orelse_seq.extend(o_seq)
#         seq.extend([{
#             "type": "block",
#             "name": "for",
#             "blocks": [
#                 {
#                     "guard": "for init",
#                     "contents": init_seq
#                 },{
#                     "guard": "for "+target_code_expr + " in " +  iter_code_expr,
#                     "contents": body_seq
#                 },{
#                     "guard": "else",
#                     "contents": orelse_seq
#                 }
#             ]
#         }])
#     else:
#         seq.extend([{
#             "type": "block",
#             "name": "for",
#             "blocks": [
#                 {
#                     "guard": "for init",
#                     "contents": init_seq
#                 },{
#                     "guard": "for "+target_code_expr + " in " +  iter_code_expr,
#                     "contents": body_seq
#                 }
#             ]
#         }])
    
#     return seq

# def process_while(stmt):
#     """
#     Parse While statment
#     """
#     seq = []
#     test = find_element_by_tag(stmt, "test")
#     body = find_elements_by_tag(stmt, "body")
#     orelse = find_elements_by_tag(stmt, "orelse")
#     test_seq, test_code_expr = parse_expr(test)
#     body_seq = []
#     body_seq.extend(test_seq)
#     for b in body:
#         b_seq = parse_stmt(b)
#         body_seq.extend(b_seq)
    

#     if orelse is not None:
#         orelse_seq = []
#         for o in orelse:
#             o_seq = parse_stmt(o)
#             orelse_seq.extend(o_seq)
#         seq.extend([{
#             "type": "block",
#             "name": "while",
#             "blocks": [
#                 {
#                     "guard": test_code_expr,
#                     "contents": body_seq
#                 },{
#                     "guard": "else",
#                     "contents": orelse_seq
#                 }
#             ]
#         }])
#     else:
#         seq.extend([{
#             "type": "block",
#             "name": "while",
#             "blocks": [
#                 {
#                     "guard": test_code_expr,
#                     "contents": body_seq
#                 }
#             ]
#         }])
    
#     return seq

# def process_if(stmt):
#     """
#     parse If statment
#     """
#     seq = []
#     test = find_element_by_tag(stmt, "test")
#     body = find_elements_by_tag(stmt, "body")
#     orelse = find_elements_by_tag(stmt, "orelse")
#     test_seq, test_code_expr = parse_expr(test)
#     seq.extend(test_seq)
#     body_seq = []
#     for b in body:
#         b_seq = parse_stmt(b)
#         body_seq.extend(b_seq)
    
#     if orelse is not None:
#         orelse_seq = []
#         for o in orelse:
#             o_seq = parse_stmt(o)
#             orelse_seq.extend(o_seq)
#         seq.extend([{
#             "type": "block",
#             "name": "if",
#             "blocks": [
#                 {
#                     "guard": test_code_expr,
#                     "contents": body_seq
#                 },{
#                     "guard": "else",
#                     "contents": orelse_seq
#                 }
#             ]
#         }])
#     else:
#         seq.extend([{
#             "type": "block",
#             "name": "if",
#             "blocks": [
#                 {
#                     "guard": test_code_expr,
#                     "contents": body_seq
#                 }
#             ]
#         }])
    
#     return seq

# def process_with(stmt):
#     """
#     Parse With statment
#     """
#     seq = []
#     items = find_elements_by_tag(stmt, "items")
#     body = find_elements_by_tag(stmt, "body")
#     items_seq = []
#     items_code_expr = ""
#     for item in items:
#         item_seq, item_code_expr = parse_withitem(item)
#         items_seq.extend(item_seq)
#         items_code_expr += item_code_expr + ", "
#     items_code_expr = items_code_expr[:-2]
    
#     body_seq = []
#     body_seq.extend(items_seq)
#     for b in body:
#         b_seq = parse_stmt(b)
#         body_seq.extend(b_seq)
#     seq.extend([{
#             "type": "block",
#             "name": "with",
#             "blocks": [
#                 {
#                     "guard": items_code_expr,
#                     "contents": body_seq
#                 }
#             ]
#         }])
#     return seq

# def process_ayncwith(stmt):
#     """
#     Parse AsyncWith statment
#     """
#     seq = []
#     items = find_elements_by_tag(stmt, "items")
#     body = find_elements_by_tag(stmt, "body")
#     items_seq = []
#     items_code_expr = ""
#     for item in items:
#         item_seq, item_code_expr = parse_withitem(item)
#         items_seq.extend(item_seq)
#         items_code_expr += item_code_expr + ", "
#     items_code_expr = items_code_expr[:-2]
    
#     body_seq = []
#     body_seq.extend(item_seq)
#     for b in body:
#         b_seq = parse_stmt(b)
#         body_seq.extend(b_seq)

#     seq.extend([{
#             "type": "block",
#             "name": "with",
#             "blocks": [
#                 {
#                     "guard": items_code_expr,
#                     "contents": body_seq
#                 }
#             ]
#         }])
#     return seq

# def process_match(stmt):
#     """
#     Parse Match statment
#     """
#     subject = find_element_by_tag(stmt, "subject")
#     cases = find_elements_by_tag(stmt, "cases")
#     seq = []
#     subject_seq, subject_code = parse_expr(subject)
#     seq.extend(subject_seq)
#     cases_seq = []
#     for c in cases:
#         c_seq = parse_match_case(c)
#         cases_seq.extend(c_seq)
#     seq.extend([{
#         "type":"blocks",
#         "name":"match",
#         "blocks": cases_seq
#     }])
#     return seq

# def process_try(stmt):
#     """
#     parse Try statment
#     """
#     seq = []
#     body = find_elements_by_tag(stmt, "body")
#     handlers = find_elements_by_tag(stmt, "handlers")
#     orelse = find_element_by_tag(stmt, "orelse")
#     finalbody = find_element_by_tag(stmt, "finalbody")

#     body_seq = []
#     for b in body:
#         b_seq = parse_stmt(b)
#         body_seq.extend(b_seq)
#     body_block = [{
#         "guard":None,
#         "contents":body_seq
#     }]
#     seq.extend(body_block)
#     if handlers is not None:
#         handler_seq = [parse_excepthandler(h) for h in handlers]
#         seq.extend(handler_seq)
#     if orelse is not None:
#         oe_seq = parse_stmt(orelse)
#         orelse_block = [{
#             "guard":"else",
#             "contents":oe_seq
#         }]
#         seq.extend(orelse_block)
#     if finalbody is not None:
#         f_seq = parse_stmt(finalbody)
#         finally_block = [{
#             "guard":"finally",
#             "contents":f_seq
#         }]
#         seq.extend(finally_block)
    
#     try_except_block = [{
#         "type": "blocks",
#         "name": "try",
#         "blocks":seq
#     }]
#     return try_except_block

# def process_trystar(stmt):
#     """
#     parse TryStare statment
#     """
#     seq = []
#     body = find_elements_by_tag(stmt, "body")
#     handlers = find_elements_by_tag(stmt, "handlers")
#     orelse = find_element_by_tag(stmt, "orelse")
#     finalbody = find_element_by_tag(stmt, "finalbody")

#     body_seq = []
#     for b in body:
#         b_seq = parse_stmt(b)
#         body_seq.extend(b_seq)
#     body_block = [{
#         "guard":None,
#         "contents":body_seq
#     }]
#     seq.extend(body_block)
#     if handlers is not None:
#         handler_seq = [parse_excepthandler(h) for h in handlers]
#         seq.extend(handler_seq)
#     if orelse is not None:
#         oe_seq = parse_stmt(orelse)
#         orelse_block = [{
#             "guard":"else",
#             "contents":oe_seq
#         }]
#         seq.extend(orelse_block)
#     if finalbody is not None:
#         f_seq = parse_stmt(finalbody)
#         finally_block = [{
#             "guard":"finally",
#             "contents":f_seq
#         }]
#         seq.extend(finally_block)
#     try_except_block = [{
#         "type": "blocks",
#         "name": "try",
#         "blocks":seq
#     }]
#     return try_except_block

# def process_raise(stmt):
#     """
#     parse Raise statment
#     """
#     seq = []
#     exc = find_element_by_tag(stmt, "exc")
#     cause = find_element_by_tag(stmt, "cause")
#     if exc is not None and cause is not None:
#         exc_seq, exc_code_expr = parse_expr(exc)
#         cause_seq, cause_code_expr = parse_expr(cause)
#         seq.extend(exc_seq)
#         seq.extend(cause_seq)
#         seq.extend([{
#             "type": "controlFlow",
#             "name": "raise",
#             "value": exc_code_expr + " from " + cause_code_expr
#         }])
#         return seq
#     elif exc is not None and cause is None:
#         exc_seq, exc_code_expr = parse_expr(exc)
#         seq.extend(exc_seq)
#         seq.extend([{
#             "type": "controlFlow",
#             "name": "raise",
#             "value": exc_code_expr
#         }])
#         return seq
#     elif exc is None and cause is None:
#         seq.extend([{
#             "type": "controlFlow",
#             "name": "raise",
#             "value": ""
#         }])
#         return seq

# def process_assert(stmt):
#     """
#     parse Assert statment
#     """
#     seq = []
#     test = find_element_by_tag(stmt, "test")
#     msg = find_element_by_tag(stmt, "msg")
#     if msg is not None:
#         test_seq, test_code_expr = parse_expr(test)
#         msg_seq, msg_code_expr = parse_expr(msg)
#         seq.extend(test_seq)
#         seq.extend(msg_seq)
#         seq.extend([{
#             "type": "block",
#             "name": "if",
#             "blocks": [
#                 {
#                     "guard": test_code_expr,
#                     "contents": [{
#                                     "type": "controlFlow",
#                                     "name": "raise",
#                                     "value": msg_code_expr
#                                 }]
#                 }
#             ]
#         }])
#         return seq
#     else:
#         test_seq, test_code_expr = parse_expr(test)
#         seq.extend(test_seq)
#         seq.extend([{
#             "type": "block",
#             "name": "if",
#             "blocks": [
#                 {
#                     "guard": test_code_expr,
#                     "contents": [{
#                                     "type": "controlFlow",
#                                     "name": "raise",
#                                     "value": ""
#                                 }]
#                 }
#             ]
#         }])
#         return seq

# def process_import(stmt):
#     """
#     parse Import statment
#     """
#     pass

# def process_importfrom(stmt):
#     """
#     parse ImportFrom statment
#     """
#     pass

# def process_global(stmt):
#     """
#     parse Global statment
#     """
#     pass

# def process_nonlocal(stmt):
#     """
#     parse NonLocal statment
#     """
#     pass

# def process_expr(stmt):
#     """
#     parse Expr statment
#     """
#     value = find_element_by_tag(stmt, "value")
#     return parse_expr(value)[0]

# def process_pass(stmt):
#     """
#     parse Pass statment
#     """
#     seq = []
#     seq.extend([{
#         "type":"controlFlow",
#         "name":"pass",
#         "value": ""
#     }])
#     return seq

# def process_break(stmt):
#     """
#     parse Break statment
#     """
#     seq = []
#     seq.extend([{
#         "type":"controlFlow",
#         "name":"break",
#         "value": ""
#     }])
#     return seq

# def process_continue(stmt):
#     """
#     parse Continue statment
#     """
#     seq = []
#     seq.extend([{
#         "type":"controlFlow",
#         "name":"continue",
#         "value": ""
#     }])
#     return seq





# # Expression Grammar

# def parse_expr(expr):
#     """
#     Generate the expression sequence return in json form
#     """
#     expr_handlers = {
#         "BoolOp": process_boolop,
#         "NamedExpr": process_namedexpr,
#         "BinOp": process_binop,
#         "UnaryOp": process_unaryop,
#         "Lambda": process_lambda,
#         "IfExp": process_ifexp,
#         "Dict": process_dict,
#         "Set": process_set,
#         "ListComp": process_listcomp,
#         "SetComp": process_setcomp,
#         "DictComp": process_dictcomp,
#         "GeneratorExp": process_generatorexp,
#         "Await": process_await,
#         "Yield": process_yield,
#         "YieldFrom": process_yieldfrom,
#         "Compare": process_compare,
#         "Call": process_call,
#         "FormattedValue": process_formattedvalue,
#         "JoinedStr": process_joinedstr,
#         "Constant": process_constant,
#         "Attribute": process_attribute,
#         "Subscript": process_subscript,
#         "Starred": process_starred,
#         "Name": process_name,
#         "List": process_list,
#         "Tuple": process_tuple,
#         "Slice": process_slice,
#     }
#     ignore_expr = []
#     if expr.attrib.get(xsi_attrib) not in ignore_expr:
#         return expr_handlers[expr.attrib.get(xsi_attrib)](expr)
#     else:
#         return [], ""

# def process_boolop(expr):
#     bool_operator_handlers = {
#         "And": "and",
#         "Or": "or"
#     }
#     operator = find_element_by_tag(expr, "op")
#     boolop_children = [ch for ch in expr.getchildren() if ch.tag != "op"]
#     seq = []
#     code_expr = ""
#     i = 0
#     for ch in boolop_children:
#         ch_seq, ch_code_exp = parse_expr(ch)
#         seq.extend(ch_seq)
#         code_expr += ch_code_exp + f" {bool_operator_handlers[operator.attrib.get(xsi_attrib)]} "
#     code_expr = code_expr[:-(len(bool_operator_handlers[operator.attrib.get(xsi_attrib)])+2)]
#     return seq, code_expr

# def process_namedexpr(expr):
#     """
#     Procerss NamedExpr expression
#     """
#     target = find_element_by_tag(expr, "target")
#     value = find_element_by_tag(expr, "value")
#     seq = []
#     code_expr = ""
#     target_seq, target_code_expr = parse_expr(target)
#     value_seq, value_code_expr = parse_expr(value)
#     seq.extend(target_seq)
#     seq.extend(value_seq)
#     code_expr += target_code_expr + ":" + value_code_expr
#     return seq, code_expr

# def process_binop(expr):
#     """
#     Generate binary operation seq and code expression
#     """
#     bin_operator_handlers = {
#         "Add": "+",
#         "Sub": "-",
#         "Mult": "*",
#         "MatMult": "@",
#         "Div": "/",
#         "Mod": "%",
#         "Pow": "**",
#         "LShift": "<<",
#         "RShift": ">>",
#         "BitOr": "|",
#         "BitXor": "^",
#         "BitAnd": "&",
#         "FloorDiv": "//"
#     }
#     expr_children = expr.getchildren()
#     left, right, op = None, None, None
#     for ex in expr_children:
#         if ex.tag == "left":
#             left = ex
#         elif ex.tag == "right":
#             right = ex
#         elif ex.tag == "op":
#             op = ex

#     if left.attrib.get(xsi_attrib) == "BinOp" and right.attrib.get(xsi_attrib) == "BinOp":
#         left_seq, left_code_expr = parse_expr(left)
#         right_seq, right_code_expr = parse_expr(right)
#         left_seq.extend(right_seq)
#         seq = left_seq
#         code_expr = "("+left_code_expr + f") {bin_operator_handlers[op.attrib.get(xsi_attrib)]} (" + right_code_expr +")"
#         return seq, code_expr

#     elif left.attrib.get(xsi_attrib) == "BinOp" and right.attrib.get(xsi_attrib) != "BinOp":
#         left_seq, left_code_expr = parse_expr(left)
#         right_seq, right_code_expr = parse_expr(right)
#         left_seq.extend(right_seq)
#         seq = left_seq
#         code_expr = "("+left_code_expr + f") {bin_operator_handlers[op.attrib.get(xsi_attrib)]} " + right_code_expr
#         return seq, code_expr
    
#     elif left.attrib.get(xsi_attrib) != "BinOp" and right.attrib.get(xsi_attrib) == "BinOp":
#         left_seq, left_code_expr = parse_expr(left)
#         right_seq, right_code_expr = parse_expr(right)
#         left_seq.extend(right_seq)
#         seq = left_seq
#         code_expr = left_code_expr + f" {bin_operator_handlers[op.attrib.get(xsi_attrib)]} (" + right_code_expr +")"
#         return seq, code_expr
    
#     elif left.attrib.get(xsi_attrib) != "BinOp" and right.attrib.get(xsi_attrib) != "BinOp":
#         left_seq, left_code_expr = parse_expr(left)
#         right_seq, right_code_expr = parse_expr(right)
#         left_seq.extend(right_seq)
#         seq = left_seq
#         code_expr = "("+ left_code_expr + f" {bin_operator_handlers[op.attrib.get(xsi_attrib)]} " + right_code_expr + ")"
#         return seq, code_expr

# def process_unaryop(expr):
#     un_operator_handlers = {
#         "Invert": "~",
#         "Not": "!",
#         "UAdd": "+",
#         "USub": "-" 
#     }
#     seq, code_expr = parse_expr(find_element_by_tag(expr, "operand"))
#     return seq, un_operator_handlers[find_element_by_tag(expr, "op").attrib.get(xsi_attrib)] + code_expr

# def process_lambda(expr):
#     parameters = parse_parameters(expr)
#     body_seq, body_code_expr = parse_expr(find_element_by_tag(expr, "body"))
#     return [], f"Lambda : {', '.join(parameters.keys())} :" + body_code_expr

# def process_ifexp(expr):
#     """
#     parse short if expression
#     """
#     # parent_seq, parent_code_expr = parse_expr(find_element_by_tag(expr.getparent(),"targets"))  
#     test_seq, test_code_expr = parse_expr(find_element_by_tag(expr, "test"))
#     body_seq, body_code_expr = parse_expr(find_element_by_tag(expr, "body"))
#     orelse_seq, orelse_code_expr = parse_expr(find_element_by_tag(expr, "orelse"))
#     code_expr = body_code_expr + " if " + test_code_expr + " else " + orelse_code_expr
#     seq = [{
#         "type": "blocks",
#         "name": "if",
#         "blocks": [ {
#                     "guard":test_code_expr,
#                     "contents":body_seq
#                     },
#                     {
#                     "guard":"else",
#                     "contents":orelse_seq
#                     }
#                 ]
#     }]
#     return seq, code_expr
    
# def process_dict(expr):
#     """
#     parse dictionary creation
#     """
#     keys = [ch for ch in expr.getchildren() if ch.tag != "keys"]
#     values = [ch for ch in expr.getchildren() if ch.tag != "values"]
#     code_expr = "{"
#     seq = []
#     for i in range(len(keys)):
#         key_seq, key_code_expr = parse_expr(keys[i])
#         value_seq, value_code_expr = parse_expr(values[i])
#         code_expr += key_code_expr + ":" + value_code_expr + ", "
#         seq.extend(key_seq)
#         seq.extend(value_seq)
#     code_expr = code_expr[:-2]
#     code_expr += "}"
#     return seq, code_expr

# def process_set(expr):
#     """
#     parse set creation
#     """
#     elts = find_elements_by_tag(expr, "elts")
#     code_expr = "{"
#     seq = []
#     for e in elts:
#         elts_seq, elts_code_expr = parse_expr(e)
#         code_expr += elts_code_expr +", "
#         seq.extend(elts_seq)
#     code_expr = code_expr[:-2] + "}" 
#     return seq, code_expr

# def process_listcomp(expr):
#     """
#     parse ListComp
#     """
#     elt = find_element_by_tag(expr, "elt")
#     generators = find_element_by_tag(expr, "generators")
#     elt_seq, elt_code_expr = parse_expr(elt)
#     target_seq, target_code_expr, iter_seq, iter_code_expr, ifs_seq, ifs_code_expr = parse_comprehension(generators)
#     seq = []
#     elt_seq.extend(iter_seq)
#     # seq.extend(iter_seq)
#     code_expr = "[" + elt_code_expr + " for " + target_code_expr + " in " + iter_code_expr
#     if ifs_seq == None:
#         seq.extend([{
#             "type": "block",
#             "name": "for",
#             "blocks": [
#                 {
#                     "guard": "for init",
#                     "contents": target_seq
#                 },{
#                     "guard": target_code_expr + " in " + iter_code_expr,
#                     "contents": elt_seq
#                 }
#             ]
#         }])
#         code_expr += "]"
#     else:
#         seq.extend([{
#             "type": "block",
#             "name": "for",
#             "blocks": [
#                 {
#                     "guard": "for init",
#                     "contents": target_seq
#                 },{
#                     "guard": target_code_expr + " in " + iter_code_expr,
#                     "contents": [{
#                         "type":"blocks",
#                         "name":"if",
#                         "blocks":[
#                             {
#                                 "guard":ifs_code_expr,
#                                 "contents":elt_seq
#                             }
#                         ]
#                     }]
#                 }
#             ]
#         }])
#         code_expr += " if " + ifs_code_expr + "]"
#     return seq, code_expr

# def process_setcomp(expr):
#     """
#     parse SetComp
#     """
#     elt = find_element_by_tag(expr, "elt")
#     generators = find_element_by_tag(expr, "generators")
#     elt_seq, elt_code_expr = parse_expr(elt)
#     target_seq, target_code_expr, iter_seq, iter_code_expr, ifs_seq, ifs_code_expr = parse_comprehension(generators)
#     seq = []
#     elt_seq.extend(iter_seq)
#     # seq.extend(iter_seq)
#     code_expr = "{" + elt_code_expr + " for " + target_code_expr + " in " + iter_code_expr
#     if ifs_seq == None:
#         seq.extend([{
#             "type": "block",
#             "name": "for",
#             "blocks": [
#                 {
#                     "guard": "for init",
#                     "contents": target_seq
#                 },{
#                     "guard": target_code_expr + " in " + iter_code_expr,
#                     "contents": elt_seq
#                 }
#             ]
#         }])
#         code_expr += "}"
#     else:
#         seq.extend([{
#             "type": "block",
#             "name": "for",
#             "blocks": [
#                 {
#                     "guard": "for init",
#                     "contents": target_seq
#                 },{
#                     "guard": target_code_expr + " in " + iter_code_expr,
#                     "contents": [{
#                         "type":"blocks",
#                         "name":"if",
#                         "blocks":[
#                             {
#                                 "guard":ifs_code_expr,
#                                 "contents":elt_seq
#                             }
#                         ]
#                     }]
#                 }
#             ]
#         }])
#         code_expr += " if " + ifs_code_expr + "}"
#     return seq, code_expr

# def process_dictcomp(expr):
#     """
#     parse DictComp
#     """
#     key = find_element_by_tag(expr, "key")
#     value = find_element_by_tag(expr, "value")
#     generators = find_element_by_tag(expr, "generators")
#     key_seq, key_code_expr = parse_expr(key)
#     value_seq, value_code_expr = parse_expr(value)
#     elt_seq = []
#     elt_seq.extend(key_seq)
#     elt_seq.extend(value_seq)
#     target_seq, target_code_expr, iter_seq, iter_code_expr, ifs_seq, ifs_code_expr = parse_comprehension(generators)
#     seq = []
#     elt_seq.extend(iter_seq)
#     # seq.extend(iter_seq)
#     code_expr = "{" + key_code_expr + " : " + value_code_expr + " for " + target_code_expr + " in " + iter_code_expr
#     if ifs_seq == None:
#         seq.extend([{
#             "type": "block",
#             "name": "for",
#             "blocks": [
#                 {
#                     "guard": "for init",
#                     "contents": target_seq
#                 },{
#                     "guard": target_code_expr + " in " + iter_code_expr,
#                     "contents": elt_seq
#                 }
#             ]
#         }])
#         code_expr += "}"
#     else:
#         seq.extend([{
#             "type": "block",
#             "name": "for",
#             "blocks": [
#                 {
#                     "guard": "for init",
#                     "contents": target_seq
#                 },{
#                     "guard": target_code_expr + " in " + iter_code_expr,
#                     "contents": [{
#                         "type":"blocks",
#                         "name":"if",
#                         "blocks":[
#                             {
#                                 "guard":ifs_code_expr,
#                                 "contents":elt_seq
#                             }
#                         ]
#                     }]
#                 }
#             ]
#         }])
#         code_expr += " if " + ifs_code_expr + "}"
#     return seq, code_expr

# def process_generatorexp(expr):
#     """
#     Parse GenerateorExpr expression
#     """
#     elt = find_element_by_tag(expr, "elt")
#     generators = find_element_by_tag(expr, "generators")
#     elt_seq, elt_code_expr = parse_expr(elt)
#     target_seq, target_code_expr, iter_seq, iter_code_expr, ifs_seq, ifs_code_expr = parse_comprehension(generators)
#     seq = []
#     elt_seq.extend(iter_seq)
#     # seq.extend(iter_seq)
#     code_expr = elt_code_expr + " for " + target_code_expr + " in " + iter_code_expr
#     if ifs_seq == None:
#         seq.extend([{
#             "type": "block",
#             "name": "for",
#             "blocks": [
#                 {
#                     "guard": "for init",
#                     "contents": target_seq
#                 },{
#                     "guard": iter_code_expr,
#                     "contents": elt_seq
#                 }
#             ]
#         }])
#     else:
#         seq.extend([{
#             "type": "block",
#             "name": "for",
#             "blocks": [
#                 {
#                     "guard": "for init",
#                     "contents": target_seq
#                 },{
#                     "guard": iter_code_expr,
#                     "contents": [{
#                         "type":"blocks",
#                         "name":"if",
#                         "blocks":[
#                             {
#                                 "guard":ifs_code_expr,
#                                 "contents":elt_seq
#                             }
#                         ]
#                     }]
#                 }
#             ]
#         }])
#         code_expr += " if " + ifs_code_expr
#     return seq, code_expr

# def process_await(expr):
#     """
#     parse Await
#     """
#     ch = find_element_by_tag(expr, "value")
#     ch_seq, ch_code = parse_expr(ch)
#     seq = []
#     seq.extend(ch_seq)
#     seq.extend([{
#         "type": "controlFlow",
#         "name": "await",
#         "value": ch_code
#     }])
#     return seq, "await " + ch_code

# def process_yield(expr):
#     yield_child = find_element_by_tag(expr, "value")
#     # parse depth
#     yield_seq, yield_code_expr = parse_expr(yield_child)
#     # append return call to depth as controlFlow
#     yield_seq.extend([{"type":"controlFlow", 
#                         "name": "return", 
#                         "value": yield_code_expr
#                     }])
#     return yield_seq, "yield "+yield_code_expr

# def process_yieldfrom(expr):
#     yield_child = find_element_by_tag(expr, "value")
#     # parse depth
#     yield_seq, yield_code_expr = parse_expr(yield_child)
#     # append return call to depth as controlFlow
#     yield_seq.extend([{"type":"controlFlow", 
#                         "name": "return", 
#                         "value": yield_code_expr
#                     }])
#     return yield_seq, "yield "+yield_code_expr

# def process_compare(expr):
#     cmp_op_handlers = {
#         "Eq": "==",
#         "NotEq": "!=",
#         "Lt": "<",
#         "LtE": "<=",
#         "Gt": ">",
#         "GtE": ">=",
#         "Is": "is",
#         "IsNot": "is not",
#         "In": "in",
#         "NotIn": "not in"
#     }
#     left = find_element_by_tag(expr, "left")
#     operator = find_element_by_tag(expr, "ops")
#     comparators = find_element_by_tag(expr, "comparators")
#     left_seq, left_code_expr = parse_expr(left)
#     comp_seq, comp_code_expr = parse_expr(comparators)
#     left_seq.extend(comp_seq)
#     seq = left_seq
#     code_expr = left_code_expr + " " + cmp_op_handlers[operator.attrib.get(xsi_attrib)] + " " + comp_code_expr
#     return seq, code_expr

# def process_call(expr):
#     func = find_element_by_tag(expr, "func")
#     args = find_elements_by_tag(expr, "args")
#     keywords = find_elements_by_tag(expr, "keywords")
#     seq = []
#     func_args_code_expr = "("
#     if args is not None:
#         args_parsed_seq = {}
#         for ar in args:
#             arg_seq, arg_code_expr = parse_expr(ar)
#             args_parsed_seq[arg_code_expr]=arg_seq
#             seq.extend(arg_seq)
#             func_args_code_expr += arg_code_expr + ", "
#         func_args_code_expr = func_args_code_expr[:-2]
#     if keywords is not None:
#         keywords_parsed_seq = {}
#         for keyword in keywords:
#             keyword_seq, keyword_code_expr = parse_keyword(keyword)
#             keywords_parsed_seq[keyword_code_expr] = keyword_seq
#             seq.extend(keyword_seq)
#             func_args_code_expr += keyword_code_expr + ", "
#         func_args_code_expr = func_args_code_expr[:-2]
#     func_args_code_expr += ")"

#     func_seq, func_code_expr = parse_expr(func)
#     parts = func_code_expr.rsplit('.',1)
#     to = []
#     if len(parts) > 1:
#         to.extend([parts[0]])
#         func_code_expr = parts[1]
    
#     seq.extend(func_seq)
#     func_code_expr += func_args_code_expr
#     seq.extend([
#         {
#             "type":"methodInvocation",
#             "to":to,
#             "method": func_code_expr
#         }
#     ])
#     if len(to)>0:
#         return seq,  to[0] + "." + func_code_expr
#     return seq, func_code_expr
    
# def process_formattedvalue(expr):
#     """
#     Procerss FormattedValue expression
#     """
#     value = find_element_by_tag(expr, "value")
#     value_seq, value_code_expr = parse_expr(value)
#     if value.attrib.get("type") == "str":
#         return value_seq, "{"+value_code_expr[1:-1]+"}"
#     else:
#         return value_seq, "{"+value_code_expr+"}"

# def process_joinedstr(expr):
#     """
#     Procerss JoineStr expression
#     """
#     values = find_elements_by_tag(expr, "values")
#     seq = []
#     code_expr = ""
#     for v in values:
#         v_seq, v_code = parse_expr(v)
#         seq.extend(v_seq)
#         if v.attrib.get("type") == "str":
#             code_expr += v_code[1:-1]
#         else:
#             code_expr += v_code
#     return seq, "f'"+code_expr+"'"

# def process_constant(expr):
#     """
#     Procerss constant expression
#     """
#     if expr.attrib.get("type") == "str":
#         return [], f'"{expr.attrib.get("value")}"'
#     elif expr.attrib.get("type") == "int":
#         return [], expr.attrib.get('value')

# def process_attribute(expr):
#     """
#     Procerss Attribute expression
#     """
#     value = find_element_by_tag(expr, "value")
#     value_seq, value_code_expr = parse_expr(value)
#     return value_seq, value_code_expr+"."+expr.attrib.get("attr")

# def process_subscript(expr):
#     """
#     Process Subscript expression
#     """
#     value = find_element_by_tag(expr, "value")
#     slic = find_element_by_tag(expr, "slice")
#     value_seq, value_code_expr = parse_expr(value)
#     slice_seq, slice_code_expr = parse_expr(slic)
#     seq = []
#     seq.extend(slice_seq)
#     seq.extend(value_seq)
#     if slic.attrib.get(xsi_attrib) != "Slice":
#         return seq, value_code_expr + "["+slice_code_expr+"]"
#     else:
#         return seq, value_code_expr + slice_code_expr

# def process_starred(expr):
#     """
#     Process Starred expression
#     """
#     value = find_element_by_tag(expr, "value")
#     seq, code_expr = parse_expr(value)
#     return seq, "*"+code_expr

# def process_name(expr):
#     """
#     Process Name expression
#     """
#     return [], expr.attrib.get('id')

# def process_list(expr):
#     """
#     Process List expression
#     """
#     elts = find_elements_by_tag(expr, "elts")
#     seq = []
#     code_expr = "["
#     for el in elts:
#         el_seq, el_code_expr = parse_expr(el)
#         seq.extend(el_seq)
#         code_expr += el_code_expr + ", "
#     code_expr = code_expr[:-2] + "]"
#     return seq, code_expr

# def process_tuple(expr):
#     """
#     Process Tuple expression
#     """
#     elts = find_elements_by_tag(expr, "elts")
#     seq = []
#     code_expr = "("
#     for el in elts:
#         el_seq, el_code_expr = parse_expr(el)
#         seq.extend(el_seq)
#         code_expr += el_code_expr + ", "
#     code_expr = code_expr[:-2] + ")"
#     return seq, code_expr

# def process_slice(expr):
#     """
#     Process Slice expression
#     """
#     lower = find_element_by_tag(expr, "lower")
#     upper = find_element_by_tag(expr, "upper")
#     step = find_element_by_tag(expr, "step")
#     seq = []
#     code_expr = "["
#     if lower is not None:
#         lower_seq, lower_code_expr = parse_expr(lower)
#         seq.extend(lower_seq)
#         code_expr += lower_code_expr
#     if upper is not None:
#         upper_seq, upper_code_expr = parse_expr(upper)
#         seq.extend(upper_seq)
#         code_expr += ":"+ upper_code_expr
#     if step is not None:
#         step_seq, step_code_expr = parse_expr(step)
#         seq.extend(step_seq)
#         code_expr += ":"+ step_code_expr
#     code_expr += "]"
#     return seq, code_expr



# # items


# def parse_withitem(item):
#     """
#     Process WithItem
#     """
#     seq = []
#     code_expr = "" 
#     context_expr = find_element_by_tag(item, "context_expr")
#     optional_vars = find_element_by_tag(item, "optional_vars")
#     con_expr_seq, con_expr_code = parse_expr(context_expr)
#     seq.extend(con_expr_seq)
#     code_expr += con_expr_code
#     if optional_vars is not None:
#         ov_seq, ov_code = parse_expr(optional_vars)
#         seq.extend(ov_seq)
#         code_expr += " as " + ov_code
#     return seq, code_expr

# def parse_excepthandler(handlers):
#     """
#     return {"guard": "exception", "contents":[body]}
#     """
#     except_name = ""
#     if handlers.attrib.get("name") is not None:
#         except_name = handlers.attrib.get("name")
#     typee = find_element_by_tag(handlers, "type")
#     bodys = find_elements_by_tag(handlers, "body")
#     t_seq, t_code = parse_expr(typee)
#     seq = []
#     for b in bodys:
#         b_seq = parse_stmt(b)
#         seq.extend(b_seq)
#     return {
#         "guard": "except " + t_code + ("" if except_name == "" else (" as "+ except_name)),
#         "contents": seq
#     }
    


# def parse_match_case(case):
#     """
#     Parse MatchCase
#     """
#     pattern = find_element_by_tag(case, "pattern")
#     bodys = find_elements_by_tag(case, "body")
#     patter_seq, pattern_code = parse_pattern(pattern)
#     body_seq = []
#     for b in bodys:
#         b_seq = parse_stmt(b)
#         body_seq.extend(b_seq)
#     seq = [{
#         "guard":"case "+pattern_code + ":",
#         "contents":body_seq
#         }]
#     return seq

# def parse_pattern(pattern):
#     """
#     Parse Pattern
#     """
#     patterns = None
#     match pattern.attrib.get(xsi_attrib):
#         case "MatchValue":
#             return parse_expr(find_element_by_tag(pattern, "value"))
#         case "MatchSingleton":
#             return [], pattern.attrib.get("value")
#         case "MatchSequence":
#             patterns = find_elements_by_tag(pattern, "patterns")
#             if patterns is None:
#                 return [], "[]"
#             seq = []
#             code = "["
#             for p in patterns:
#                 p_seq, p_code = parse_pattern(p)
#                 # seq.extend(p_seq)
#                 code += p_code + ", "
#             code = code[:-2] + ']'
#             return seq, code
#         case "MatchMapping":
#             rest = True if pattern.attrib.get("rest") == "rest" else False
#             keys = find_elements_by_tag(pattern, "keys")
#             patterns = find_elements_by_tag(pattern, "patterns")
#             if keys is None and patterns is None:
#                 return [], "{}"
#             seq = []
#             code = "{"
#             for i in range(len(keys)):
#                 k_seq, k_code = parse_expr(keys[i])
#                 p_seq, p_code = parse_pattern(patterns[i])
#                 # seq.extend(k_seq)
#                 # seq.extend(p_seq)
#                 code += k_code + ": " + p_code + ", "
#             if rest:
#                 code += "**rest, "
#             code = code[:-2] + "}"
#             return seq, code
#         case "MatchClass":
#             cls = find_element_by_tag(pattern, "cls")
#             patterns = find_elements_by_tag(pattern, "patterns")
#             kwd_patterns = find_elements_by_tag(pattern, "kwd_patterns")
            
#             cls_seq, cls_code = parse_expr(cls)
#             seq = []
#             code = cls_code + "("
#             if patterns is not None:
#                 for p in patterns:
#                     p_seq, p_code = parse_pattern(p)
#                     code += p_code + ", "
#                 if(kwd_patterns is None):
#                     code = code[:-2]
#             if kwd_patterns is not None:
#                 for k in kwd_patterns:
#                     k_seq, k_code = parse_pattern(k)
#                     # seq.extend(k_seq)
#                     code += k_code + ", "
#                 code = code[:-2]
#             code += ")"
#             return seq, code
#         case "MatchStar":
#             pass
#         case "MatchAs":
#             return [], pattern.attrib.get("name")
#         case "MatchOr":
#             patterns = find_elements_by_tag(pattern, "patterns")
#             seq = []
#             code = ""
#             if patterns is not None:
#                 for p in patterns:
#                     p_seq, p_code = parse_pattern(p)
#                     # seq.extend(p_seq)
#                     code += p_code + " | "
#                 code = code[:-2]
#             return seq, code



# def process_String(expr):
#     return expr.attrib.get('value')

# def process_Int(expr):
#     return expr.attrib.get('value')

# def process_Identifier(expr):
#     return expr.attrib.get('value')



# def parse_comprehension(comp):
#     generators_target = find_element_by_tag(comp, "target")
#     generators_iter = find_element_by_tag(comp, "iter")
#     generators_ifs = find_elements_by_tag(comp, "ifs")
#     target_seq, target_code_expr = parse_expr(generators_target)
#     iter_seq, iter_code_expr = parse_expr(generators_iter)
#     if generators_ifs != None:
#         ifs_seq = []
#         ifs_code_expr = ""
#         for ifs in generators_ifs:
#             seq, code = parse_expr(ifs)
#             ifs_seq.extend(seq)
#             ifs_code_expr += code + " and "
#         ifs_code_expr = ifs_code_expr[:-5]
#         return target_seq, target_code_expr, iter_seq, iter_code_expr, ifs_seq, ifs_code_expr
#     else:
#         return target_seq, target_code_expr, iter_seq, iter_code_expr, None, None

# def parse_keyword(keyword):
#     """
#     Process Keyword
#     """
#     arg = keyword.attrib.get("arg")
#     value = find_element_by_tag(keyword, "value")
#     v_seq, v_code = parse_expr(value)
#     return v_seq, arg+"="+v_code
