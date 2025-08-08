import re
import sys
from pathlib import Path
parent_parent_folder = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_parent_folder))

from ..process_xmi import *
from ..clean_seq import *

class javaSequenceGenerator:
    def __init__(self, cleanseq=False):
        self.cleanseq = cleanseq

    def generate_sequence(self, xmi_string):
        """
        Generate sequence in json form by analyzing the function xmi meta model
        """
        try:
            root = parse_xmi_string(xmi_string)
            method = root.xpath(f'.//MethodDeclaration')
            if len(method) == 0:
                method = root.xpath(f'.//ConstructorDeclaration')
            method_declaration = method[0]
            modifiers = []
            name = None
            Type = None
            parameters = []
            exceptions = []
            for ch in method_declaration.getchildren():
                if ch.tag == "Modifier":
                    modifiers.append(ch)
                elif ch.tag == "SimpleName":
                    name = ch
                elif ch.tag == "Parameter":
                    pass
                elif ch.tag == "BlockStmt":
                    pass
                elif "Exception" in ch.attrib.get("value"):
                    exceptions.append(ch)
                else:
                    Type = ch

            # extract parameter
            seq = []
            parameters_seq, parameters_code = self.process_Parameters(method_declaration)
            seq.extend(parameters_seq)
            # extract sequence from function body
            body_sequence = self.process_method_body(method_declaration)
            seq.extend(body_sequence)
            
            seqcleaner = SeqCleaner()
            title = (f"{''.join(m.attrib.get('value') for m in modifiers)}"
                    + f"{('' if Type is None else (Type.attrib.get('value')+str(' ')))}"
                    + f"{name.attrib.get('value')}"
                    + f"({parameters_code})"
                    + ('' if len(exceptions) == 0 else ' throws '+', '.join(e.attrib.get('value') for e in exceptions))
            )
            return {
                    "title": re.sub(r'\s*=\s*', '=', title),
                    "sequence": seqcleaner.clean_element(seq) if self.cleanseq else seq,
                }
        except Exception as e:
            # import traceback
            # traceback.print_exc()
            return {}
        
    def process_Parameters(self, method_declaration):
        """
        Extract the function parameters return sequence and code string
        """
        parameters = find_elements_by_tag(method_declaration, "Parameter")
        rec_parameters = find_elements_by_tag(method_declaration, "ReceiverParameter")
        if len(parameters) == 0:
            return [],""
        parameters_seq = []
        parameter_code = ""
        if len(parameters) != 0:
            for p in parameters:
                p_seq, p_code = self.process_Parameter(p)
                # parameters_seq.extend(p_seq)
                parameter_code += p_code + ", "
        if len(rec_parameters) != 0:
            for p in rec_parameters:
                p_seq, p_code = self.process_receiver_parameter(p)
                # parameters_seq.extend(p_seq)
                parameter_code += p_code + ", "
        if len(parameter_code) >= 2:
            parameter_code = parameter_code[:-2]
        return parameters_seq, parameter_code

    def process_method_body(self, method_declaration):
        """
        Generate the method sequence return in json form
        """
        # get all body stmts
        body = find_element_by_tag(method_declaration, "BlockStmt")
        body_seq = self.process_stmt(body)
        return body_seq

    def process_Parameter(self, parameter):
        modifier = None
        Type = None
        Identifier = None
        for p in parameter.getchildren():
            if "Type" in p.tag:
                Type = p
            elif p.tag == "Modifier":
                modifier = p
            elif p.tag == "SimpleName":
                Identifier = p
        seq = [{"type":"scopedVariable",
                "name":Identifier.attrib.get("value")}]
        return seq, parameter.attrib.get("value")

    def process_receiver_parameter(self, rec_parameter):
        modifier = None
        Type = None
        Identifier = None
        for p in rec_parameter.getchildren():
            if "Type" in p.tag:
                Type = p
            elif p.tag == "Modifier":
                modifier = p
            elif "Name" in p.tag:
                Identifier = p
        seq = [{"type":"scopedVariable",
                "name":Identifier.attrib.get("value")}]
        return seq, rec_parameter.attrib.get("value")

    def process_stmt(self, stmt, lable=None):
        """
        Generate the statements sequence return in json form
        """
        stmt_handlers = {
            "BlockStmt": self.process_BlockStmt,
            "AssertStmt": self.process_AssertStmt,
            "BreakStmt": self.process_BreakStmt,
            "ContinueStmt": self.process_ContinueStmt,
            "DoStmt": self.process_DoStmt,
            "EmptyStmt": self.process_EmptyStmt,
            "ExplicitConstructorInvocationStmt": self.process_ExplicitConstructorInvocationStmt,
            "ForEachStmt": self.process_ForEachStmt,
            "ForStmt": self.process_ForStmt,
            "IfStmt": self.process_IfStmt,
            "LabeledStmt": self.process_LabeledStmt,
            "LocalClassDeclarationStmt": self.process_LocalClassDeclarationStmt,
            "LocalRecordDeclarationStmt": self.process_LocalRecordDeclarationStmt,
            "ReturnStmt": self.process_ReturnStmt,
            "SwitchStmt": self.process_SwitchStmt,
            "SynchronizedStmt": self.process_SynchronizedStmt,
            "ThrowStmt": self.process_ThrowStmt,
            "TryStmt": self.process_TryStmt,
            "UnparsableStmt": self.process_UnparsableStmt,
            "WhileStmt": self.process_WhileStmt,
            "YieldStmt": self.process_YieldStmt,

            "ExpressionStmt": self.process_ExpressionStmt,
        }
        if stmt is None:
            return []
        if stmt.tag in stmt_handlers:
            return stmt_handlers[stmt.tag](stmt, lable)
        elif 'Expr' in stmt.tag:
            return self.process_expr(stmt)
        else: return []
        

    def process_BlockStmt(self, blockStmt, label=None):
        seq = []
        for stmt in blockStmt.getchildren():
            seq.extend(self.process_stmt(stmt, label))
        return seq

    def process_AssertStmt(self, asrtStmt, label=None):
        return [{"type":"controlFlow", "name": "assert", "value": asrtStmt.attrib.get('value')}]

    def extrack_break(self, breakstring):
        start_index = breakstring.find('break')  # Find the position of 'break'
        if start_index != -1:  # If 'break' is found
            result = breakstring[start_index + len('break'):].strip().rstrip(';')  # Extract text after 'break' and remove semicolon
            return result
        
    def process_BreakStmt(self, breakstmt, label=None):
        """
        ast generator has problem it add the comment into vlaue
        """
        return [{"type":"controlFlow", "name": "break", "value": self.extrack_break(breakstmt.attrib.get('value'))}]

    def extrack_continue(self, continuestring):
        start_index = continuestring.find('continue')  # Find the position of 'break'
        if start_index != -1:  # If 'break' is found
            result = continuestring[start_index + len('continue'):].strip().rstrip(';')  # Extract text after 'break' and remove semicolon
            return result
        
    def process_ContinueStmt(self, continuestmt, label=None):
        return [{"type":"controlFlow", "name": "continue", "value": self.extrack_continue(continuestmt.attrib.get('value'))}]

    def process_DoStmt(self, dostmt, label=None):
        stmt = None
        expr = None
        for element in dostmt.getchildren():
            if element.tag.endswith('Stmt'):
                stmt = element
            elif element.tag.endswith('Expr'):
                expr = element
        body_seq = self.process_stmt(stmt)
        return [{
                "type": "blocks",
                "name": (label+":" if label is not None else '')+"loop",
                "blocks": [
                    {
                        "guard": expr.attrib.get('value'),
                        "contents": body_seq
                    }
                ]
                }
            ]

    def process_EmptyStmt(self, emptystmt, label=None):
        return []

    def process_ExplicitConstructorInvocationStmt(self, ecstmt, label=None):
        seq = []
        for element in ecstmt.getchildren():
            par_seq, par_code = self.process_expr(element)
            seq.extend(par_seq)
        seq.extend([{
            "type":"methodInvocation",
            "To":[],
            "method": ecstmt.atttrib.get('value').rstrip(";")
        }])
    
    def process_ExpressionStmt(self, expr_stmt, label=None):
        return self.process_expr(expr_stmt.getchildren()[0])

    def process_ForEachStmt(self, foreachstmt, label=None):
        seq = []
        varibaledeclaration = None
        expression = None
        blockstmt = None
        for element in foreachstmt.getchildren():
            if element.tag == 'VariableDeclarationExpr':
                varibaledeclaration = element
            elif  element.tag =='BlockStmt':
                blockstmt = element
            else:
                expression = element
        seq.extend(self.process_expr(expression))
        seq.extend(self.process_expr(varibaledeclaration))
        blockseq = self.process_stmt(blockstmt)
        seq.extend([{
            "type": "blocks",
            "name": (label+":" if label is not None else '')+"loop",
            "blocks":[{
                "guard": self.findBracketContent(foreachstmt.attrib.get('value')),
                "contents":blockseq
            }]
        }])
        return seq

    def process_ForStmt(self, forstmt, label=None):
        seq = []
        blockstmt = None
        for element in forstmt.getchildren():
            if element.tag == 'BlockStmt':
                blockstmt = element
            else:
                seq.extend(self.process_expr(element))
        blockseq = self.process_stmt(blockstmt)
        seq.extend([{
            "type": "blocks",
            "name": (label+":" if label is not None else '')+"loop",
            "blocks":[{
                "guard":self.findBracketContent(forstmt.attrib.get('value')),
                "contents":blockseq
                }]
            
        }])
        return seq
            

    def process_IfStmt(self, block, label=None):
        if_child = block.getchildren()
        expression = if_child[0]
        ifblock = if_child[1]
        seq = []
        seq.extend(self.process_expr(expression))
        if len(if_child) > 2:
            elseseq = self.process_stmt(if_child[2])
            seq.extend([{
                "type": "blocks",
                "name": (label+":" if label is not None else '')+"if",
                "blocks": [
                    {"guard": expression.attrib.get('value'),
                    "contents":self.process_stmt(ifblock)},
                    {"guard": "else",
                    "contents":elseseq}
                ]
            }])
        else:
            seq.extend([{
                "type": "blocks",
                "name": (label+":" if label is not None else '')+"if",
                "blocks": [
                    {"guard": expression.attrib.get('value'),
                    "contents":self.process_stmt(ifblock)},
                ]
            }])
        return seq



    def process_LabeledStmt(self, labelstmt, label=None):
        simplename = None
        stmt = None
        for element in labelstmt.getchildren():
            if element.tag == 'SimpleName':
                simplename = element
            else:
                stmt = element
        return self.process_stmt(stmt, lable=simplename.attrib.get('value'))

    def process_LocalClassDeclarationStmt(self, lcds, label=None):
        """
        we don't need to process class declaration for sequence
        in term of declaration no difference between local or somewhere else
        """
        return []
    
    def process_LocalRecordDeclarationStmt(self, lrds, label=None):
        """
        we don't need to process class declaration for sequence
        in term of declaration no difference between local or somewhere else
        """
        return []
    
    def process_ReturnStmt(self, returnstmt, label=None):
        if len(returnstmt.getchildren()) == 0:
            return [{
                "type": "controlFlow",
                "name": "return",
                "value": "",
            }]
        rstmt = returnstmt.getchildren()[0]
        seq = []
        seq.extend(self.process_stmt(rstmt))
        seq.extend([{
            "type": "controlFlow",
            "name": "return",
            "value": rstmt.attrib.get('value'),
        }])
        return seq
    
    def process_SwitchStmt(self, switchstmt, label=None):
        seq = []
        cases = []
        for element in switchstmt.getchildren():
            if element.tag == 'SwitchEntry':
                cases.extend(self.process_SwitchEntry(element))
            else:
                seq.extend(self.process_expr(element))
        seq.extend([{
            "type": "blocks",
            "name": (label+":" if label is not None else '')+"if",
            "blocks": cases
        }])
        return seq
    
    def process_SwitchEntry(self, switchentry, label=None):
        seq = []
        if 'default' in switchentry.attrib.get('value'):
            for element in switchentry.getchildren():
                seq.extend(self.process_stmt(element))
            return [{
                "guard":"default",
                "contents":seq
            }]
        else:
            expression = []
            for element in switchentry.getchildren():
                if element.tag.endswith('Expr'):
                    expression.append(element.attrib.get('value'))
                else:
                    seq.extend(self.process_stmt(element))
            return [{
                "guard":",".join(expression),
                "contents":seq
            }]
    
    def process_SynchronizedStmt(self, syncstmt, label=None):
        expression = None
        blockstmt = None
        for element in syncstmt.getchildren():
            if element.tag == 'BlockStmt':
                blockstmt = element
            else:
                expression = element
        seq = []
        seq.extend(self.process_expr(expression))
        syncstmts = []
        for element in blockstmt.getchildren():
            syncstmts.extend(self.process_stmt(element))
        seq.extend([{
            "type": "blocks",
            "name": (label+":" if label is not None else '')+"synchronized",
            "blocks": [{
                "guard": expression.attrib.get('value'),
                "contents":syncstmts
            }]
        }])
        return seq

    def process_ThrowStmt(self, throwstmt, label=None):
        expression = throwstmt.getchildren()[0]
        seq= []
        seq.extend(self.process_expr(expression))
        seq.extend([{
            "type":"controlFlow", 
            "name": "assert", 
            "value": expression.attrib.get('value')
        }])
        return seq
    
    def process_TryStmt(self, trystmt, label=None):
        """
        expression seq could be joined with block since no condition check exist
        """
        blockseq = []
        catchseq = []
        finallyCheck = False
        for element in trystmt.getchildren():
            if element.tag == 'CatchClause':
                catchseq.extend(self.process_CatchClause(element))
            elif element.tag == 'BlockStmt':
                if finallyCheck is False:
                    blockseq.extend(self.process_stmt(element))
                    finallyCheck = True
                else:
                    catchseq.extend([{
                        'guard': 'finally',
                        'contents': self.process_BlockStmt(element)
                    }])
            else:
                blockseq.extend(self.process_expr(element))
        blocks = [{
            "guard": "",
            "contents": blockseq
        }]
        blocks.extend(catchseq)
        seq = [{
            "type": "blocks",
            "name": (label+":" if label is not None else '')+"try",
            "blocks": blocks
        }]
        return seq

    def process_CatchClause(self, catchstmt, label=None):
        parameter = find_element_by_tag(catchstmt, "Parameter")
        parseq, parcode = self.process_Parameter(parameter)
        seq = []
        blockstmt = find_element_by_tag(catchstmt, "BlockStmt")
        seq.extend(self.process_BlockStmt(blockstmt))
        returnseq = [{
            "guard": "catch ("+ parcode + ")",
            "contents": seq
        }]
        return returnseq 

    def process_UnparsableStmt(self, unpstmt, label=None):
        return []
    
    def process_WhileStmt(self, whilestmt, label=None):
        seq = []
        expression = None
        stmt = None
        for element in whilestmt.getchildren():
            # change to str.ends with
            if element.tag.endswith('Expr'):
                expression = element
            elif element.tag.endswith('Stmt'):
                stmt = element
        seq.extend(self.process_expr(expression))
        seq.extend([{
            "type": "blocks",
            "name": (label+":" if label is not None else '')+"loop",
            "blocks":[{
                "guard": expression.attrib.get('value'),
                "contents":self.process_stmt(stmt)
            }]
        }])
        return seq
        
    def process_YieldStmt(self, yieldstmt, label=None):
        """
        parser doesn't see this element
        """
        return []




    def process_expr(self, expr):
        expr_handlers = {
            "ArrayAccessExpr":self.process_ArrayAccessExpr ,
            "VariableDeclarationExpr":self.process_VariableDeclarationExpr ,
            "ArrayCreationExpr":self.process_ArrayCreationExpr ,
            "ArrayInitializerExpr":self.process_ArrayInitializerExpr ,
            "AssignExpr":self.process_AssignExpr ,
            "BinaryExpr":self.process_BinaryExpr ,
            "CastExpr":self.process_CastExpr ,
            "ClassExpr":self.process_ClassExpr ,
            "ConditionalExpr":self.process_ConditionalExpr ,
            "EnclosedExpr":self.process_EnclosedExpr ,
            "FieldAccessExpr":self.process_FieldAccessExpr ,
            "InstanceOfExpr":self.process_InstanceOfExpr ,
            "LambdaExpr":self.process_LambdaExpr ,
            "LiteralExpr":self.process_LiteralExpr ,
            "MethodCallExpr":self.process_MethodCallExpr ,
            "MethodReferenceExpr":self.process_MethodReferenceExpr ,
            "NameExpr":self.process_NameExpr ,
            "ObjectCreationExpr":self.process_ObjectCreationExpr ,
            "SuperExpr":self.process_SuperExpr ,
            "ThisExpr":self.process_ThisExpr ,
            "TypeExpr":self.process_TypeExpr ,
            "UnaryExpr":self.process_UnaryExpr ,
            "SwitchExpr":self.process_SwitchExpr,
        }
        if expr is None: 
            return []
        if expr.tag in expr_handlers:
            return expr_handlers[expr.tag](expr)
        else: return []

    def process_ArrayAccessExpr(self, arrayexpr):
        seq = []
        for element in arrayexpr.getchildren():
            seq.extend(self.process_expr(element))
        return seq

    def process_VariableDeclarationExpr(self, expr):
        seq = []
        for element in expr.getchildren():
            if element.tag == 'VariableDeclarator':
                seq.extend(self.process_VariableDeclarator(element))
            else:
                pass
                
        return seq

    def process_VariableDeclarator(self, expr):
        seq = []
        vartype = None
        simple_name = None
        expression = None
        for element in expr.getchildren():
            if element.tag == "SimpleName":
                simple_name = element
            elif element.tag.endswith('Type'):
                vartype = element
            elif element.tag.endswith('Expr'):
                expression = element
        if expression is not None:
            object_creation = self.process_expr(expression)
            if len(object_creation) > 2:
                seq.extend(object_creation[:-1])
            seq.extend([{
                "type":"newInstance",
                "name":simple_name.attrib.get('value'),
                "new_type":vartype.attrib.get('value')
            }])
        return seq

    def process_ArrayCreationExpr(self, arraycreationexpr):
        seq = []
        arraytype = None
        arraycreationlevel = []
        arrayinitializerexpr = None
        for element in arraycreationexpr.getchildren():
            if element.tag.endswith('Type'):
                arraytype = element
            elif element.tag == 'ArrayCreationLevel':
                arraycreationlevel.append(element)
            elif element.tag == 'ArrayInitializerExpr':
                arrayinitializerexpr = element
        if arrayinitializerexpr is not None:
            seq.extend(self.process_ArrayInitializerExpr(arrayinitializerexpr))
        for el in arraycreationlevel:
            seq.extend(self.process_ArrayCreationLevel(el))
        seq.extend(self.process_TypeExpr(arraytype))
        seq.extend([{
            "type":"newInstance",
            "new_type":arraytype.attrib.get('value')
        }])
        return seq

    def process_ArrayCreationLevel(self,arraycreationlevel):
        seq = []
        for element in arraycreationlevel.getchildren():
            seq.extend(self.process_expr(element))
        return seq
    
    def process_ArrayInitializerExpr(self, arrayinitializerexpr):
        seq = []
        for element in arrayinitializerexpr.getchildren():
            seq.extend(self.process_expr(element))
        return seq
    
    def process_AssignExpr(self, assignexpr):
        seq = []
        for element in assignexpr.getchildren():
            if element.tag != "NameExpr":
                seq.extend(self.process_expr(element))
        return seq
    
    def process_SwitchExpr(self, switchexpr):
        return self.process_SwitchStmt(switchexpr)

    def process_BinaryExpr(self, binaryexpr):
        seq = []
        for element in binaryexpr.getchildren():
            seq.extend(self.process_expr(element))
        return seq
    
    def process_CastExpr(self, castexpr):
        """
        does not contain seq information in casting it self so we process only the expression
        """
        seq = []
        for element in castexpr.getchildren():
            if element.tag.endswith('Expr'):
                seq.extend(self.process_expr(element))
        return seq

    def process_ClassExpr(self, classexpre):
        """no process need"""
        return []

    def process_ConditionalExpr(self, conditionalexpr):
        children = conditionalexpr.getchildren()
        seq = []
        seq.extend(self.process_expr(children[0]))
        seq.extend([{
                    "type": "blocks",
                    "name": "if",
                    "blocks": [{"guard": children[0].attrib.get('value'),
                                "contents":self.process_expr(children[1])
                                },
                                {"guard": "else",
                                "contents":self.process_expr(children[2])
                                }]
                }])
        return seq

    def process_EnclosedExpr(self, encloseexpr):
        return self.process_expr(encloseexpr.getchildren()[0])

    def process_FieldAccessExpr(self, fieldexpr):
        seq = []
        simplename = None
        expr = None
        for element in fieldexpr.getchildren():
            if element.tag == "SimpleName":
                simplename = element
            else:
                expr = element
                seq.extend(self.process_expr(element))
        return seq
    
    def process_LambdaExpr(self, lambdaexpr):
        """
        Lambda expression does not excute any statment by just define so we do not add it to seq
        """
        return []
        

    def process_LiteralExpr(self, litralexpr):
        """Litral expression does not contain any seq information it just a value to be stored"""
        return []

    def process_MethodCallExpr(self, methodcallexpr):
        isnameaccessed = False
        objectaccess = None
        funcName = None
        funcParameters = []
        for element in methodcallexpr.getchildren():
            if element.tag == 'SimpleName':
                funcName = element
                isnameaccessed = True
            else:
                if isnameaccessed:
                    funcParameters.append(element)
                else:
                    objectaccess = element
        seq = []
        if objectaccess is not None:
            seq.extend(self.process_expr(objectaccess))
        for par in funcParameters:
            seq.extend(self.process_expr(par))
        
        seq.extend([{
            "type":"methodInvocation",
            "to":split_string(objectaccess.attrib.get('value')) if objectaccess is not None else [],
            "method":(funcName.attrib.get('value')
                      +'('
                      +self.findBracketContent(methodcallexpr.attrib.get('value')[methodcallexpr.attrib.get('value').find(funcName.attrib.get('value')):])
                      +')'),
        }])
        return seq

    def process_MethodReferenceExpr(self, expr):
        """
        it just assign a function to other referance to be used no sequence information
        """
        return []

    def process_NameExpr(self, nameexpr):
        return []
    
    def process_ObjectCreationExpr(self, objectcreationexpr):
        seq = []
        objecttype = None
        for element in objectcreationexpr.getchildren():
            if element.tag.endswith('Type'):
                objecttype = element
            else:
                seq.extend(self.process_expr(element))
        seq.extend([{
            "type": "newInstance",
            "new_type": objecttype.attrib.get('value')
        }])
        return seq

    def process_InstanceOfExpr(self, expr):
        """
        it does not call any functions just comparing condition no need to be added to seq
        """
        return []
    
        
    def process_SuperExpr(self, expr):
        return []
    
    def process_ThisExpr(self, expr):
        return []
    
    def process_TypeExpr(self, expr):
        return []

    def process_UnaryExpr(self, expr):
        for element in expr.getchildren():
            if element.tag.endswith('Expr'):
                Uexpr = element
        return self.process_expr(Uexpr)


    


    def findBracketContent(self, input_string):
        # Find the first opening parenthesis
        start = input_string.find('(')
        if start != -1:
            # Use a counter to track parentheses
            count = 1
            end = start + 1
            while end < len(input_string) and count > 0:
                if input_string[end] == '(':
                    count += 1  # Increment count for nested opening parenthesis
                elif input_string[end] == ')':
                    count -= 1  # Decrement count for closing parenthesis
                end += 1

            # If the parentheses were balanced
            if count == 0:
                return input_string[start + 1:end - 1].strip()

        return ''  # If no valid arguments are found